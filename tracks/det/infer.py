"""Детекции в формате 8.1 контракта: COCO-источники без обучения и свой детектор.

Источники:
  coco_person  YOLO11x (COCO), класс person на полном кадре, imgsz 1280, все 1718 кадров (часть B6)
  coco_crop    YOLO11x (COCO), кропы top ×4 всех строк rows_ext, классы COCO -> словарь 8.2 таблицей
  ft_yolo      свой детектор (B4): кропы top ×4 всех строк rows_ext и полный кадр для людей

Запуск (одна ячейка Kaggle на источник, сеть не нужна):
  python tracks/det/infer.py --source coco_person --weights <ts-models>/yolo11x.pt
  python tracks/det/infer.py --source coco_crop   --weights <ts-models>/yolo11x.pt
  python tracks/det/infer.py --source ft_yolo     --weights <ART>/models/ft_yolo_v1/ft_yolo_v1.pt

Результат: ART/dets/<source>__v<N>.parquet и карточка, в конце ts.check.check_dets.
Прогон возобновляемый: порции лежат в ART/dets/_parts/<source>__v<N>/, повторный запуск той же
команды пропускает готовые порции. Отладка на train-днях: --subset train (карточка coverage=train_only).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import CONF_MIN, FT_CLASSES, NMS_IOU, TRAIN_DATES, PartStore
import fetch_weights

from ts import io, paths
from ts.crops import crop_table, to_frame_coords
from ts.rows import load_rows_ext

# Классы COCO -> словарь 8.2. Счётные: cup, bowl, phone, bag, paper. Негативы: laptop, bottle, vase.
COCO_CROP_MAP = {
    'cup': 'cup', 'wine glass': 'cup',
    'bowl': 'bowl',
    'cell phone': 'phone',
    'handbag': 'bag', 'backpack': 'bag',
    'book': 'paper',
    'laptop': 'laptop', 'bottle': 'bottle', 'vase': 'vase',
}
COCO_PERSON_MAP = {'person': 'person'}


@dataclass(frozen=True)
class Pass:
    """Один проход модели: что подаётся на вход и какие классы забираются."""
    name: str                   # имя каталога порций
    view: str                   # 'full' (кадр целиком) или зона стола для ts.crops.crop_table
    imgsz: int                  # длинная сторона входа модели
    class_map: dict             # имя класса модели -> класс словаря 8.2
    scale: float = 1.0          # увеличение кропа


PROFILES: dict[str, list[Pass]] = {
    'coco_person': [Pass('full', 'full', 1280, COCO_PERSON_MAP)],
    'coco_crop': [Pass('top', 'top', 1024, COCO_CROP_MAP, scale=4.0)],
    'ft_yolo': [Pass('top', 'top', 1024, {c: c for c in FT_CLASSES if c != 'person'}, scale=4.0),
                Pass('full', 'full', 1280, {'person': 'person'})],
}
CHUNK = {'full': 128, 'top': 512}


def units_of(view: str, subset: str) -> pd.DataFrame:
    """Единицы работы прохода: кадры (для `full`) или строки rows_ext (для кропов), в устойчивом порядке."""
    if view == 'full':
        frames = io.load_frames()
        if subset == 'train':
            frames = frames[frames['date_utc'].isin(TRAIN_DATES)]
        ids = sorted(frames['frame_id'])
        return pd.DataFrame({'unit_id': ids, 'frame_id': ids})
    rows = load_rows_ext()
    if subset == 'train':
        rows = rows[rows['split'] == 'train']
    rows = rows.sort_values(['src_frame_id', 'row_id', 'dt'], kind='stable').reset_index(drop=True)
    rows['unit_id'] = rows['row_id'] + '|' + rows['dt'].astype(str)
    rows['frame_id'] = rows['src_frame_id']      # детекция лежит на кадре, с которого взят кроп
    return rows[['unit_id', 'frame_id', 'row_id', 'dt', 'slot', 'x1', 'y1', 'x2', 'y2']]


def load_unit(unit, p: Pass):
    """Картинка для модели и CropMeta (None для полного кадра)."""
    if p.view == 'full':
        return io.read_image(unit.frame_id), None
    return crop_table(unit.frame_id, (unit.x1, unit.y1, unit.x2, unit.y2), kind=p.view, scale=p.scale,
                      slot=int(unit.slot))


def decode(result, unit, meta, p: Pass, names: dict) -> list[tuple]:
    """Результат Ultralytics для одной картинки -> сырые детекции в координатах кадра.

    Координаты кропа переводятся через `to_frame_coords`, боксы обрезаются по кадру. После перевода
    классов в словарь 8.2 делается NMS по классу (IoU 0.5) внутри этой картинки: два класса COCO могут
    отображаться в один класс словаря. Между кропами дедупликации нет, её делает `ts.assign` (8.1).
    """
    boxes = result.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4)
    conf = result.boxes.conf.cpu().numpy().astype(float).clip(0.0, 1.0)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    if meta is not None:
        boxes = to_frame_coords(boxes, meta)
    boxes = common.clip_to_frame(boxes)
    classes = np.array([p.class_map[names[c]] for c in cls], dtype=object)
    ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (conf >= CONF_MIN)
    boxes, conf, classes = boxes[ok], conf[ok], classes[ok]
    keep = common.nms_per_class(boxes, conf, classes, NMS_IOU)
    row_id = None if meta is None else unit.row_id
    dt = None if meta is None else int(unit.dt)
    return [(unit.frame_id, classes[i], float(conf[i]), *boxes[i].tolist(), row_id, dt) for i in keep]


def run_pass(model, p: Pass, units: pd.DataFrame, store: PartStore, args) -> None:
    """Посчитать недостающие порции прохода."""
    names = model.names
    index = {v: k for k, v in names.items()}
    missing = sorted(set(p.class_map) - set(index))
    if missing:
        raise SystemExit(f'В весах {args.weights} нет классов {missing}. Это не те веса для источника {args.source}.')
    class_ids = sorted(index[name] for name in p.class_map)
    chunk = args.chunk or CHUNK[p.name]
    n_parts = math.ceil(len(units) / chunk)
    todo = [i for i in range(n_parts) if not store.is_done(i)]
    print(f'[{args.source}/{p.name}] единиц: {len(units)}, порций: {n_parts}, осталось: {len(todo)}, '
          f'imgsz {p.imgsz}, классов модели: {len(class_ids)}', flush=True)
    started, done_units = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for part in todo:
            part_units = units.iloc[part * chunk:(part + 1) * chunk]
            rows: list[tuple] = []
            for start in range(0, len(part_units), args.batch):
                batch = list(part_units.iloc[start:start + args.batch].itertuples(index=False))
                loaded = list(pool.map(lambda u: load_unit(u, p), batch))
                results = model.predict([image for image, _ in loaded], imgsz=p.imgsz, conf=CONF_MIN, iou=NMS_IOU,
                                        classes=class_ids, agnostic_nms=False, max_det=args.max_det,
                                        device=args.device, quantize=16 if args.half else 32, verbose=False)
                for unit, (_, meta), result in zip(batch, loaded, results):
                    rows.extend(decode(result, unit, meta, p, names))
            store.save(part, common.raw_table(rows), part_units['unit_id'].tolist())
            done_units += len(part_units)
            rate = (time.time() - started) / done_units
            left = sum(len(units.iloc[i * chunk:(i + 1) * chunk]) for i in todo if i > part)
            print(f'[{args.source}/{p.name}] порция {part + 1}/{n_parts}: детекций {len(rows)}, '
                  f'{rate * 1000:.0f} мс на единицу, осталось около {left * rate / 60:.1f} мин', flush=True)


def model_refs(weights: Path) -> tuple[list[str], dict]:
    """Поле `models` карточки и поля, которые наследуются из карточки своих весов (если она есть)."""
    if weights.name in fetch_weights.WEIGHTS:
        return [fetch_weights.model_ref(weights)], {}
    inherited = {}
    card_path = weights.with_suffix('.json')
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding='utf-8'))
        inherited = {k: card[k] for k in ['manual_labels_used', 'trained_on_dates', 'train_dataset', 'base_model']
                     if k in card}
    return [f'{weights.name} sha256:{common.file_sha256(weights)}'], inherited


def log_time() -> str:
    now = datetime.now(timezone.utc) + timedelta(hours=3)
    return f"{['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'][now.weekday()]} {now:%d.%m %H:%M}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, choices=sorted(PROFILES))
    parser.add_argument('--weights', type=Path, required=True, help='явный путь к файлу весов, автоскачивания нет')
    parser.add_argument('--version', type=int, default=1, help='N в имени <source>__v<N>')
    parser.add_argument('--subset', choices=['all', 'train'], default='all',
                        help='train: только кадры 25–26.10, отладка, карточка coverage=train_only')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--chunk', type=int, default=0, help='единиц в порции (по умолчанию 128 кадров или 512 кропов)')
    parser.add_argument('--workers', type=int, default=4, help='потоков чтения и нарезки кропов')
    parser.add_argument('--device', default=None, help="устройство Ultralytics: '0', 'cpu', 'mps'")
    parser.add_argument('--half', action='store_true', help='FP16 на GPU (быстрее, по умолчанию FP32)')
    parser.add_argument('--max-det', type=int, default=300)
    args = parser.parse_args()

    common.require_data()
    ultralytics_version = common.setup_ultralytics()
    common.seed_everything()
    import torch

    args.device = args.device or common.default_device()
    weights_sha = common.file_sha256(args.weights) if args.weights.is_file() else ''
    model = common.load_model(args.weights)
    models, inherited = model_refs(args.weights)
    name = f'{args.source}__v{args.version}'
    if (paths.ART / 'dets' / f'{name}.parquet').exists():
        raise SystemExit(f'{name}.parquet уже существует. Артефакты неизменяемы: поднимите --version.')
    parts_root = paths.art('dets', '_parts') / (name if args.subset == 'all' else f'{name}__{args.subset}')

    started = time.time()
    raw, frames, rows_done, passes_info = [], set(), {}, []
    for p in PROFILES[args.source]:
        units = units_of(p.view, args.subset)
        chunk = args.chunk or CHUNK[p.name]
        fingerprint = {'source': args.source, 'pass': p.name, 'view': p.view, 'scale': p.scale, 'imgsz': p.imgsz,
                       'class_map': p.class_map, 'conf': CONF_MIN, 'iou': NMS_IOU, 'half': args.half,
                       'max_det': args.max_det, 'weights_sha256': weights_sha, 'ultralytics': ultralytics_version,
                       'chunk': chunk, 'units': {'n': len(units), 'sha1': common.sha1_ids(units['unit_id'])}}
        store = PartStore(parts_root / p.name, fingerprint)
        run_pass(model, p, units, store, args)
        table, done = store.load(math.ceil(len(units) / chunk))
        if sorted(done) != sorted(units['unit_id']):
            raise SystemExit(f'[{args.source}/{p.name}] обработано {len(done)} единиц из {len(units)}')
        raw.append(table)
        frames |= set(units['frame_id'])
        if p.view != 'full':
            rows_done = {'n': len(units), 'sha1': common.sha1_ids(units['unit_id'])}
        passes_info.append({'pass': p.name, 'view': p.view, 'scale': p.scale, 'imgsz': p.imgsz,
                            'class_map': p.class_map, 'units': len(units), 'detections': len(table)})
    elapsed = time.time() - started

    raw = [t for t in raw if len(t)]
    dets = common.finalize_dets(pd.concat(raw, ignore_index=True) if raw else pd.DataFrame(), args.source)
    by_class = {k: int(v) for k, v in dets['class_name'].value_counts().sort_index().items()}
    views = ', '.join(f"{i['view']} imgsz {i['imgsz']}" + (f" ×{i['scale']:g}" if i['view'] != 'full' else '')
                      for i in passes_info)
    card = common.dets_card(
        models=models,
        inputs=['frames.csv'] + (['rows_ext.parquet'] if rows_done else []),
        coverage='all' if args.subset == 'all' else 'train_only',
        frame_ids=frames,
        notes=(f'{views}; conf от {CONF_MIN}; NMS по классу IoU {NMS_IOU} внутри одной картинки (Ultralytics по '
               'классам модели, затем по классам словаря 8.2); между кропами дедупликации нет (8.1); '
               'sha1 это sha1 от отсортированных id через перевод строки'),
        rows_processed=rows_done or None,
        passes=passes_info,
        detections={'n': len(dets), 'frames_with_detections': int(dets['frame_id'].nunique()), 'by_class': by_class},
        params={'conf': CONF_MIN, 'iou': NMS_IOU, 'half': args.half, 'batch': args.batch, 'max_det': args.max_det,
                'seed': common.SEED},
        runtime={'device': common.device_name(args.device), 'seconds': round(elapsed, 1),
                 'ultralytics': ultralytics_version, 'torch': torch.__version__},
        kaggle_dataset_version=None,
        **inherited,
    )
    path = common.write_dets_artifact(dets, args.source, args.version, card)

    print(f'\n{name}: {len(dets)} детекций на {dets["frame_id"].nunique()} кадрах из {len(frames)} обработанных')
    print('классы: ' + ', '.join(f'{k} {v}' for k, v in by_class.items()))
    print(f'файл: {path}\nкарточка: {path.with_suffix(".json")}\nts.check dets: OK')
    print(f'время этого запуска: {elapsed / 60:.1f} мин на {common.device_name(args.device)} (для учёта квоты GPU)')
    print('\nСтрока для experiments/LOG.md (метрики пустые, пока источник не оценён через ts.cv):')
    print(f'| {log_time()} | B | {name} | {args.source}: {views} | | | | | | | | | | '
          f'{len(dets)} детекций ({", ".join(f"{k} {v}" for k, v in by_class.items())}); '
          f'`{common.command_line()}`; `python -m ts.check dets {path.name}`: OK |')


if __name__ == '__main__':
    main()
