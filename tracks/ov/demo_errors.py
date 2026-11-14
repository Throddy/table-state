"""C6: лист «сложные кадры с ошибками» по train-кадрам, в оформлении `preview.py` организаторов.

Запуск:
  python tracks/ov/demo_errors.py artifacts/preds/COMBO.csv      # report/demo_errors.jpg и report/demo_errors.csv

Нужен файл предсказаний со всеми 231 train-строками (out-of-fold). Файл сабмита не подходит: в нём только тест.
Берутся train-кадры с наибольшей суммой абсолютных ошибок по пяти счётчикам, под кадром для каждого стола
предсказание (pred) и разметка (true). Кадры выбирает правило, а не человек. Тестовые строки файла не
используются: разметки для них нет. Подписи латиницей: шрифт с кириллицей есть не везде.

Очередь оператора (`report/demo_queue.*`) собирает трек A, этот скрипт её не строит.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from contact_sheet import load_font  # noqa: E402  (добавляет корень репозитория в sys.path)

from ts import COUNTS, io, paths  # noqa: E402

REPORT = paths.ROOT / 'report'
WIDTH, HEIGHT, IMAGE_H, SCALE = 960, 650, 540, 0.75   # как в preview.py
WRONG, RIGHT = '#db2777', '#10b981'


def pick_error_frames(pred: pd.DataFrame, labels: pd.DataFrame, count: int) -> tuple[list[str], pd.DataFrame]:
    """Train-кадры с наибольшей суммой абсолютных ошибок по счётчикам и таблица строк этих кадров."""
    rows = labels.merge(pred[['row_id'] + COUNTS], on='row_id', suffixes=('_true', ''), validate='one_to_one')
    if len(rows) != len(labels):
        raise SystemExit(f'В файле предсказаний train-строк {len(rows)} из {len(labels)}: нужен файл со всеми '
                         'train-строками (out-of-fold), файл сабмита не подходит')
    rows['abs_error'] = sum((rows[c] - rows[f'{c}_true']).abs() for c in COUNTS)
    worst = rows.groupby('frame_id')['abs_error'].sum().reset_index().sort_values(
        ['abs_error', 'frame_id'], ascending=[False, True]).frame_id.tolist()[:count]
    return worst, rows[rows.frame_id.isin(worst)]


def counts_text(row, suffix: str = '') -> str:
    return '/'.join(f'{getattr(row, c + suffix):g}' for c in COUNTS)


def sheet(frame_ids: list[str], rows: pd.DataFrame, timestamps: pd.Series) -> Image.Image:
    """Лист 2 кадра в ряд: рамки столов (с ошибкой выделены) и подписи `pred` и `true` под кадром."""
    font = load_font(16)
    canvas = Image.new('RGB', (2 * WIDTH, ((len(frame_ids) + 1) // 2) * HEIGHT), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, frame_id in enumerate(frame_ids):
        image = io.read_image(frame_id).convert('RGB').resize((WIDTH, IMAGE_H))
        pen = ImageDraw.Draw(image)
        tables = rows[rows.frame_id == frame_id].sort_values('table_id')
        for r in tables.itertuples():
            box, color = [r.x1 * SCALE, r.y1 * SCALE, r.x2 * SCALE, r.y2 * SCALE], WRONG if r.abs_error > 0 else RIGHT
            pen.rectangle(box, outline=color, width=3)
            pen.text((box[0], max(0, box[1] - 20)), r.table_id, fill=color, font=font, stroke_width=1,
                     stroke_fill='black')
        x, y = (i % 2) * WIDTH, (i // 2) * HEIGHT
        canvas.paste(image, (x, y))
        draw.text((x + 6, y + IMAGE_H + 5), str(timestamps[frame_id]), fill='black', font=font)
        for j, r in enumerate(tables.itertuples()):
            draw.text((x + 6 + (j % 2) * 475, y + IMAGE_H + 28 + (j // 2) * 22),
                      f"{r.table_id}: pred {counts_text(r)}  true {counts_text(r, '_true')}", fill='black', font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pred', type=Path, help='файл предсказаний с 231 train-строкой (out-of-fold)')
    parser.add_argument('--count', type=int, default=2, help='кадров на листе')
    parser.add_argument('--out', type=Path, default=REPORT / 'demo_errors.jpg')
    args = parser.parse_args()

    tables = io.load_tables()[['row_id', 'frame_id', 'table_id', 'x1', 'y1', 'x2', 'y2']]
    labels = io.load_labels()[['row_id'] + COUNTS].merge(tables, on='row_id', validate='one_to_one')
    frame_ids, rows = pick_error_frames(pd.read_csv(args.pred), labels, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet(frame_ids, rows, io.load_frames().set_index('frame_id')['timestamp']).save(args.out, quality=90)
    columns = ['row_id', 'frame_id', 'table_id'] + COUNTS + [f'{c}_true' for c in COUNTS] + ['abs_error']
    rows.sort_values(['frame_id', 'table_id'])[columns].to_csv(args.out.with_suffix('.csv'), index=False)
    print(f'train-кадры с наибольшей ошибкой: {frame_ids}')
    print(f'лист -> {args.out}, строки -> {args.out.with_suffix(".csv")} (порядок счётчиков: {"/".join(COUNTS)})')


if __name__ == '__main__':
    main()
