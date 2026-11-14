"""Обучение своего детектора (задача B4). Только на Kaggle (GPU), локально допустима лишь короткая отладка.

Запуск одной ячейкой:
  python tracks/det/train.py --config tracks/det/configs/ft_yolo_v1.yaml \
      --data <каталог датасета build_dataset.py> --weights <ts-models>/yolo11m.pt
Возобновление после обрыва сессии (last.pt из вывода предыдущей версии ноутбука):
  python tracks/det/train.py --config ... --data ... --weights ... --resume <путь>/last.pt

Результат в ART/models/<name>/: weights/last.pt (контрольная точка каждой эпохи), итоговые веса <name>.pt,
карточка <name>.json, results.csv. Итоговые веса это last.pt после заданного числа эпох: aux-holdout
служит только для наблюдения за обучением, модель по нему не выбирается.
Сеть не используется: YOLO_OFFLINE, sync=False, plots=False, явный путь к весам.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import yaml

import common
from common import CLASS_SETS, OV_EXTRA, TRAIN_DATES
import fetch_weights

from ts import SEED, paths


def gray_trainer(gray_p: float, seed: int):
    """Тренер Ultralytics, который с вероятностью `gray_p` переводит картинку батча в серый (под ИК-кадры).

    Серый считается после сборки батча (мозаика уже применена), поэтому картинка становится серой целиком,
    как настоящий ИК-кадр. Генератор случайных чисел свой, с сидом, от устройства не зависит.
    """
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer

    class GrayTrainer(DetectionTrainer):
        _gray_rng = None

        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            if gray_p <= 0:
                return batch
            if self._gray_rng is None:
                self._gray_rng = torch.Generator().manual_seed(seed)
            image = batch['img']
            pick = (torch.rand(image.shape[0], generator=self._gray_rng) < gray_p).to(image.device)
            if pick.any():
                weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                image[pick] = (image[pick] * weights).sum(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            return batch

    return GrayTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='каталог датасета (images, labels, card.json)')
    parser.add_argument('--weights', type=Path, required=True, help='стартовые COCO-веса, явный путь')
    parser.add_argument('--resume', type=Path, help='last.pt прерванного обучения')
    parser.add_argument('--device', default=None, help="одно устройство: '0', 'cpu', 'mps'")
    parser.add_argument('--hours', type=float, help='потолок времени обучения в часах (аргумент time Ultralytics)')
    parser.add_argument('--epochs', type=int, help='отладка: переопределить число эпох')
    parser.add_argument('--imgsz', type=int, help='отладка: переопределить imgsz')
    parser.add_argument('--batch', type=int, help='переопределить размер батча')
    parser.add_argument('--fraction', type=float, help='отладка: доля обучающей выборки')
    parser.add_argument('--name', help='отладка: другое имя каталога в ART/models')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    name = args.name or cfg['name']
    train_args = dict(cfg['train'])
    for key in ['epochs', 'imgsz', 'batch', 'fraction']:
        if getattr(args, key) is not None:
            train_args[key] = getattr(args, key)
    if args.hours:
        train_args['time'] = args.hours
    if train_args.get('seed') != SEED or not train_args.get('deterministic') or train_args.get('plots'):
        raise SystemExit('Конфиг обязан задавать seed 419, deterministic: true и plots: false.')
    if train_args.get('flipud', 0.0) != 0.0:
        raise SystemExit('Вертикальное отражение запрещено задачей B4 (flipud должен быть 0).')

    data_dir = args.data.resolve()
    data_card = json.loads((data_dir / 'card.json').read_text(encoding='utf-8'))
    base = CLASS_SETS['v1']
    classes = list(data_card['classes'])
    allowed = base if cfg['classes'] == 'v1' else base + [c for c in OV_EXTRA if c in classes]
    if data_card.get('manual_labels_used') or classes != allowed:
        raise SystemExit(f"Датасет не подходит: ручные метки или классы {classes} вместо {allowed} (режим {cfg['classes']}).")
    if args.weights.name != cfg['base_weights']:
        raise SystemExit(f"Конфиг требует стартовые веса {cfg['base_weights']}, передан {args.weights.name}.")
    base_model = fetch_weights.model_ref(args.weights)      # сверяет sha256 с зафиксированным

    ultralytics_version = common.setup_ultralytics()
    common.seed_everything()
    import torch

    device = args.device or common.default_device()
    save_dir = paths.art('models', name)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = save_dir / 'data.yaml'
    data_yaml.write_text(yaml.safe_dump({'path': str(data_dir), 'train': 'images/train', 'val': 'images/holdout',
                                         'names': dict(enumerate(classes))}, allow_unicode=True), encoding='utf-8')
    trainer = gray_trainer(float(cfg.get('gray_p', 0.0)), SEED)
    final = save_dir / f'{name}.pt'
    if final.exists():
        raise SystemExit(f'{final} уже существует. Веса неизменяемы: смените name в конфиге.')

    started = time.time()
    if args.resume:
        last = save_dir / 'weights' / 'last.pt'
        last.parent.mkdir(parents=True, exist_ok=True)
        if args.resume.resolve() != last.resolve():
            shutil.copyfile(args.resume, last)
        model = common.load_model(last)
        model.train(resume=True, trainer=trainer, device=device)
    else:
        model = common.load_model(args.weights)
        model.train(trainer=trainer, data=str(data_yaml), project=str(save_dir.parent), name=name, exist_ok=True,
                    device=device, augmentations=[], **train_args)
    elapsed = time.time() - started

    last = save_dir / 'weights' / 'last.pt'
    shutil.copyfile(last, final)
    results = pd.read_csv(save_dir / 'results.csv')
    results.columns = [c.strip() for c in results.columns]
    card = {
        'name': name, 'kind': 'model', 'owner': common.OWNER, 'created_utc': common.utc_now(),
        'git_sha': common.git_sha(), 'command': common.command_line(),
        'file': final.name, 'sha256': common.file_sha256(final), 'license': fetch_weights.LICENSE,
        'base_model': base_model,
        'train_dataset': {'name': data_card['name'], 'git_sha': data_card.get('git_sha'),
                          'boxes_by_class': data_card['boxes_by_class'], 'images': data_card['images'],
                          'holdout_frames': data_card['holdout_frames'], 'pseudo_dates': data_card['pseudo_dates']},
        'manual_labels_used': False,
        'uses_train_counters': False,
        'classes': classes,
        'config': cfg | {'train': train_args},
        'epochs_done': int(results['epoch'].max()),
        'last_epoch': {k: float(v) for k, v in results.iloc[-1].items()},
        'weights_choice': 'last.pt после заданного числа эпох, по holdout модель не выбиралась',
        'resumed': bool(args.resume),
        'runtime': {'device': common.device_name(device), 'seconds_this_run': round(elapsed, 1),
                    'ultralytics': ultralytics_version, 'torch': torch.__version__},
    }
    trained_on = sorted(set(data_card['pseudo_dates']) & set(TRAIN_DATES))
    if trained_on:
        card['trained_on_dates'] = trained_on      # раздел 8.6: для такого источника действует правило 12.4
    final.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n{name}: эпох {card["epochs_done"]}, веса {final}, карточка {final.with_suffix(".json").name}')
    print(f'время этого запуска: {elapsed / 3600:.2f} ч на {common.device_name(device)} (для учёта квоты GPU)')


if __name__ == '__main__':
    main()
