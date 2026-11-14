"""Контактные листы train-кадров: рамки столов, слоты, прод-детекции, ручные счётчики.

Запуск:
  python tools/contact_sheet.py                 # все 39 train-кадров -> report/sheets/
  python tools/contact_sheet.py --frames im_... # выбранные кадры
  python tools/contact_sheet.py --context       # ещё и соседи ±30 с (без счётчиков)

Работает только с кадрами train-дней (25–26.10). Тестовые кадры отклоняются: по правилам соревнования их нельзя размечать вручную.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import ROOT

from ts import COUNTS, io

SCALE = 2
TRAIN_DATES = {'2026-10-25', '2026-10-26'}
CLASS_COLORS = {'person': '#ff4d4d', 'hand': '#ffd60a', 'cup': '#00e5ff', 'tray': '#c77dff', 'napkin': '#aaaaaa'}
TABLE_COLOR, OCCUPIED_COLOR = '#22c55e', '#ff9f1c'


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ['DejaVuSans.ttf', 'Arial.ttf', 'Helvetica.ttc']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def draw_sheet(frame_id: str, frames: pd.DataFrame, rows: pd.DataFrame, dets: pd.DataFrame,
               labels: pd.DataFrame | None, title: str = '') -> Image.Image:
    """Собрать лист одного кадра. `rows`: рамки столов, `labels`: счётчики или None для соседа."""
    font, small = load_font(22), load_font(15)
    image = io.read_image(frame_id)
    image = image.resize((image.width * SCALE, image.height * SCALE), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for d in dets.itertuples():
        if d.class_name == 'table':
            continue
        color = CLASS_COLORS.get(d.class_name, 'white')
        box = [d.x1 * SCALE, d.y1 * SCALE, d.x2 * SCALE, d.y2 * SCALE]
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(0, box[1] - 17)), f'{d.class_name} {d.confidence:.2f}', fill=color, font=small,
                  stroke_width=2, stroke_fill='black')
    lines = []
    for r in rows.sort_values('x1').itertuples():
        counters = None if labels is None else labels[labels.row_id == r.row_id]
        occupied = counters is not None and len(counters) and int(counters.people.iloc[0]) > 0
        color = OCCUPIED_COLOR if occupied else TABLE_COLOR
        box = [r.x1 * SCALE, r.y1 * SCALE, r.x2 * SCALE, r.y2 * SCALE]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[3] + 2), f'slot {r.slot} ({r.table_id})', fill=color, font=font,
                  stroke_width=2, stroke_fill='black')
        if counters is not None and len(counters):
            values = ' '.join(f'{c}={int(counters[c].iloc[0])}' for c in COUNTS)
            lines.append(f'slot {r.slot} / {r.table_id} / {r.row_id}:  {values}')
        else:
            lines.append(f'slot {r.slot} / {r.table_id}: ручных счётчиков нет (соседний кадр)')
    info = frames.loc[frame_id]
    head = (f'{title}{frame_id}   {info.timestamp} UTC   {info.hour_msk:02d} ч МСК   '
            f'прод-детекции: ' + ', '.join(f'{k} {v}' for k, v in dets[dets.class_name != 'table']
                                            .class_name.value_counts().items()))
    line_h = 30
    canvas = Image.new('RGB', (image.width, image.height + line_h * (len(lines) + 2) + 10), 'white')
    canvas.paste(image, (0, 0))
    text = ImageDraw.Draw(canvas)
    text.text((10, image.height + 6), head, fill='black', font=font)
    text.text((10, image.height + 6 + line_h), 'ручные счётчики по столам (слева направо):', fill='#555555', font=font)
    for i, line in enumerate(lines):
        text.text((10, image.height + 6 + line_h * (i + 2)), line, fill='black', font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', nargs='*', help='id train-кадров, по умолчанию все 39')
    parser.add_argument('--context', action='store_true', help='добавить листы соседних кадров ±30 с')
    parser.add_argument('--out', type=Path, default=ROOT / 'report' / 'sheets')
    args = parser.parse_args()

    frames = io.load_frames().set_index('frame_id')
    targets = io.load_targets().set_index('frame_id')
    tables, labels, dets = io.load_tables(), io.load_labels(), io.load_prod_dets()
    train_ids = targets.index[targets.split == 'train'].tolist()
    chosen = args.frames or train_ids
    wrong = [f for f in chosen if f not in train_ids]
    if wrong:
        raise SystemExit(f'Не train-кадры (тестовые кадры этим инструментом не открываются): {wrong}')
    args.out.mkdir(parents=True, exist_ok=True)
    by_frame = dict(tuple(dets.groupby('frame_id')))
    done = 0
    for frame_id in chosen:
        rows = tables[tables.frame_id == frame_id]
        jobs = [(frame_id, labels, f'train_{frame_id}.jpg', '')]
        if args.context:
            for n in targets.at[frame_id, 'context']:
                dt = int(round((frames.at[n, 'ts'] - frames.at[frame_id, 'ts']).total_seconds() / 30)) * 30
                jobs.append((n, None, f'train_{frame_id}__ctx{dt:+d}.jpg', f'сосед {dt:+d} с кадра {frame_id}: '))
        for src, lab, filename, title in jobs:
            if frames.at[src, 'date_utc'] not in TRAIN_DATES:
                raise SystemExit(f'Кадр {src} не из train-дней')
            sheet = draw_sheet(src, frames, rows, by_frame.get(src, dets.iloc[:0]), lab, title)
            sheet.save(args.out / filename, quality=90)
            done += 1
    print(f'Листов: {done}, каталог: {args.out}')


if __name__ == '__main__':
    main()
