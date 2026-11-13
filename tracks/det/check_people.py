"""Проверка двух типов пропусков people (задача B6): находит ли источник людей нужного стола.

По умолчанию две train-строки из tracks/core/PEOPLE_ERRORS.md:
  q_c5311f5b4593b510f144  слот 6, второй человек у правого края кадра (истина 2);
  q_1d3cfd4459d2c4fef90f  ночной ИК-кадр im_cf13426b6f011b9cb35c (истина 5).
Только проверка: эти кадры в обучение не идут, пороги под них не подбираются.

Запуск:
  python tracks/det/check_people.py --sources prod coco_person__v1 [ft_yolo__v1]
Печатает по каждой строке и источнику людей, привязанных `ts.assign` (правила R1 без фильтра площади),
и всех кандидатов person в зоне seat. Листы с боксами: report/sheets/people_<frame_id>.jpg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw

import common

from ts import io
from ts.assign import assign
from ts.features import load_dets
from ts.geometry import zone
from ts.rows import load_rows_ext

ROWS = ['q_c5311f5b4593b510f144', 'q_1d3cfd4459d2c4fef90f']
COLORS = ['#ff4d4d', '#00e5ff', '#ffd60a', '#c77dff']
SCALE = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', nargs='+', default=['prod', 'coco_person__v1'])
    parser.add_argument('--rows', nargs='+', default=ROWS, help='row_id train-строк')
    parser.add_argument('--out', type=Path, default=common.ROOT / 'report' / 'sheets')
    args = parser.parse_args()

    common.require_data()
    cfg = yaml.safe_load((common.ROOT / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
    labels = io.load_labels().set_index('row_id')
    rows = load_rows_ext()
    rows = rows[(rows['dt'] == 0) & rows['frame_id'].isin(rows.loc[rows['row_id'].isin(args.rows), 'frame_id'])]
    wrong = sorted(set(args.rows) - set(labels.index))
    if wrong:
        raise SystemExit(f'Не train-строки: {wrong}')
    args.out.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, Image.Image] = {}
    lines = []
    for k, source in enumerate(args.sources):
        dets = load_dets(source)
        people = dets[(dets['class_name'] == 'person') & dets['frame_id'].isin(rows['frame_id'])]
        linked = assign(people, rows, cfg)
        for row_id in args.rows:
            r = rows[rows['row_id'] == row_id].iloc[0]
            seat = zone((r.x1, r.y1, r.x2, r.y2), 'seat', int(r.slot))
            d = people[people['frame_id'] == r.frame_id]
            cx, cy = (d['x1'] + d['x2']) / 2, (d['y1'] + d['y2']) / 2
            near = d[(cx >= seat[0]) & (cx <= seat[2]) & (cy >= seat[1]) & (cy <= seat[3])]
            mine = linked[linked['row_id'] == row_id] if len(linked) else linked
            conf = ', '.join(f'{c:.2f}' for c in sorted(mine['confidence'], reverse=True)) if len(mine) else 'нет'
            lines.append({'row_id': row_id, 'frame_id': r.frame_id, 'slot': int(r.slot), 'истина': int(labels.at[row_id, 'people']),
                          'источник': source, 'привязано': len(mine), 'confidence привязанных': conf,
                          'person с центром в seat': len(near),
                          'из них conf >= 0.4': int((near['confidence'] >= 0.4).sum())})
            sheet = sheets.get(r.frame_id)
            if sheet is None:
                image = io.read_image(r.frame_id)
                sheet = sheets[r.frame_id] = image.resize((image.width * SCALE, image.height * SCALE), Image.Resampling.LANCZOS)
                ImageDraw.Draw(sheet).rectangle([v * SCALE for v in (r.x1, r.y1, r.x2, r.y2)], outline='#22c55e', width=3)
            draw = ImageDraw.Draw(sheet)
            for p in d.itertuples():
                box = [p.x1 * SCALE - k, p.y1 * SCALE - k, p.x2 * SCALE + k, p.y2 * SCALE + k]
                draw.rectangle(box, outline=COLORS[k % len(COLORS)], width=2)
                draw.text((box[0] + 3, box[1] + 3 + 14 * k), f'{source} {p.confidence:.2f}', fill=COLORS[k % len(COLORS)])
    table = pd.DataFrame(lines).sort_values(['row_id', 'источник'])
    print(table.to_string(index=False))
    for frame_id, sheet in sheets.items():
        path = args.out / f'people_{frame_id}.jpg'
        sheet.save(path, quality=90)
        print('лист:', path)


if __name__ == '__main__':
    main()
