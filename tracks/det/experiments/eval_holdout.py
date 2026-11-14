"""Recall своего детектора на aux-holdout: абсолютные числа для сравнения версий без траты сабмитов.

Запуск (6 кадров, хватает CPU):
  python tracks/det/eval_holdout.py --weights <ART>/models/ft_yolo_v1/ft_yolo_v1.pt [--tta] [--device cpu]

Модель применяется к holdout-кадрам так же, как в infer.py: кропы top ×4 каждого стола (все классы,
кроме person) и полный кадр (person). Детекции одного объекта из соседних кропов сливаются по правилу
9.2.6. Бокс aux считается найденным, если с ним по тому же правилу совпала детекция его класса с
confidence не ниже порога. Пороги 0.25 и 0.5 фиксированы, по holdout ничего не подбирается.

Ограничения: разметка aux неполная, поэтому «лишние» детекции это не ошибки, а справочное число;
сравнения с прод-детектором нет (его детекций для aux-кадров в пакете нет); кадров всего 6.
Результат печатается и пишется в <каталог весов>/holdout__<имя весов>[__tta].json.
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # модули build_dataset, common, infer лежат в tracks/det
import build_dataset
import common
import infer

from ts import io
from ts.check import CLASS_TO_COUNTER
from ts.geometry import zone
from ts.slots import slot_of

THRESHOLDS = (0.25, 0.5)
BOX = ['x1', 'y1', 'x2', 'y2']


def predict_frame(model, frame_id: str, tables: np.ndarray, small: dict, args) -> pd.DataFrame:
    """Детекции одного aux-кадра в координатах кадра: кропы top для малых классов и полный кадр для person."""
    names = model.names
    index = {v: k for k, v in names.items()}
    rows = []
    passes = [(infer.Pass('full', 'full', 1280, {'person': 'person'}), [types.SimpleNamespace(frame_id=frame_id)])]
    if small:
        units = [types.SimpleNamespace(frame_id=frame_id, row_id=f't{i}', dt=0, slot=slot_of((t[0] + t[2]) / 2),
                                       x1=t[0], y1=t[1], x2=t[2], y2=t[3]) for i, t in enumerate(tables)]
        passes.append((infer.Pass('top', 'top', 1024, small, scale=4.0), units))
    for p, units in passes:
        if not units or not set(p.class_map) <= set(index):
            continue
        class_ids = sorted(index[c] for c in p.class_map)
        for start in range(0, len(units), args.batch):
            batch = units[start:start + args.batch]
            loaded = [infer.load_unit(u, p) for u in batch]
            results = infer.predict_boxes(model, [image for image, _ in loaded], p, class_ids, args)
            for unit, (_, meta), result in zip(batch, loaded, results):
                rows.extend(infer.decode(result, unit, meta, p, names))
    table = common.raw_table(rows)
    kept = []
    for _, group in table.groupby('class_name', sort=True):       # один объект в кропах двух столов
        group = group.sort_values('confidence', ascending=False, kind='stable')
        boxes = group[BOX].to_numpy(float)
        chosen: list[int] = []
        for i in range(len(group)):
            if not build_dataset.same_object(boxes[i], boxes[chosen]).any():
                chosen.append(i)
        kept.append(group.iloc[chosen])
    return pd.concat(kept, ignore_index=True) if kept else table


def match(truth: np.ndarray, preds: np.ndarray) -> tuple[int, int]:
    """Жадное сопоставление по убыванию confidence: (найдено боксов aux, детекций без пары)."""
    free = np.ones(len(truth), dtype=bool)
    extra = 0
    for box in preds:
        hit = np.flatnonzero(build_dataset.same_object(box, truth) & free)
        if len(hit):
            free[hit[0]] = False
        else:
            extra += 1
    return int((~free).sum()), extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--tta', action='store_true', help='те же отражение и второй масштаб, что в infer.py --tta')
    parser.add_argument('--device', default=None)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--max-det', type=int, default=300)
    args = parser.parse_args()

    common.require_data()
    common.setup_ultralytics()
    common.seed_everything()
    args.device = args.device or common.default_device()
    model = common.load_model(args.weights)
    classes = [c for c in model.names.values() if c in CLASS_TO_COUNTER]
    small = {c: c for c in classes if c != 'person'}

    aux = io.load_aux_boxes()
    aux_frames = pd.read_csv(common.paths.DATA / 'aux_frames.csv')
    holdout, _ = build_dataset.pick_holdout(aux, aux_frames)
    card_path = args.weights.with_suffix('.json')
    if card_path.is_file():
        trained_holdout = json.loads(card_path.read_text(encoding='utf-8')).get('train_dataset', {}).get('holdout_frames')
        if trained_holdout is not None and sorted(trained_holdout) != holdout:
            raise SystemExit('Holdout в карточке весов не совпал с расчётным: эти веса могли видеть holdout-кадры.')

    found = {c: {t: 0 for t in THRESHOLDS} for c in classes}
    extra = {c: {t: 0 for t in THRESHOLDS} for c in classes}
    total = {c: 0 for c in classes}
    outside = {c: 0 for c in classes}
    for frame_id in holdout:
        of_frame = aux[aux['frame_id'] == frame_id]
        tables = of_frame.loc[of_frame['class_name'] == 'table', BOX].to_numpy(float)
        tops = np.array([zone(t, 'top', slot_of((t[0] + t[2]) / 2)) for t in tables]).reshape(-1, 4)
        dets = predict_frame(model, frame_id, tables, small, args)
        for c in classes:
            truth = of_frame.loc[of_frame['class_name'] == c, BOX].to_numpy(float).reshape(-1, 4)
            total[c] += len(truth)
            if c != 'person' and len(truth):
                cx, cy = (truth[:, 0] + truth[:, 2]) / 2, (truth[:, 1] + truth[:, 3]) / 2
                inside = ((cx[:, None] >= tops[None, :, 0]) & (cx[:, None] <= tops[None, :, 2]) &
                          (cy[:, None] >= tops[None, :, 1]) & (cy[:, None] <= tops[None, :, 3])).any(axis=1)
                outside[c] += int((~inside).sum())
            mine = dets[dets['class_name'] == c].sort_values('confidence', ascending=False, kind='stable')
            for t in THRESHOLDS:
                hit, more = match(truth, mine.loc[mine['confidence'] >= t, BOX].to_numpy(float).reshape(-1, 4))
                found[c][t] += hit
                extra[c][t] += more

    report = {'weights': args.weights.name, 'sha256': common.file_sha256(args.weights), 'tta': args.tta,
              'holdout_frames': holdout, 'thresholds': list(THRESHOLDS), 'git_sha': common.git_sha(),
              'command': common.command_line(), 'classes': {}}
    print(f'{args.weights.name}{" + TTA" if args.tta else ""}, holdout: {len(holdout)} aux-кадров')
    print(f"{'класс':<15}{'боксов aux':>11}{'вне зон top':>13}" + ''.join(
        f"{'recall@' + str(t):>13}{'лишних@' + str(t):>13}" for t in THRESHOLDS))
    for c in classes:
        report['classes'][c] = {'aux_boxes': total[c], 'outside_top_zones': outside[c],
                                **{f'found@{t}': found[c][t] for t in THRESHOLDS},
                                **{f'recall@{t}': (round(found[c][t] / total[c], 4) if total[c] else None) for t in THRESHOLDS},
                                **{f'unmatched@{t}': extra[c][t] for t in THRESHOLDS}}
        cells = ''.join((f"{found[c][t]:>6}/{total[c]:<3}{(found[c][t] / total[c] if total[c] else float('nan')):>4.2f}"
                         f"{extra[c][t]:>13}") for t in THRESHOLDS)
        print(f'{c:<15}{total[c]:>11}{outside[c]:>13}{cells}')
    out = args.weights.parent / f'holdout__{args.weights.stem}{"__tta" if args.tta else ""}.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'отчёт: {out}')


if __name__ == '__main__':
    main()
