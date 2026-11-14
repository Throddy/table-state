"""Демо очереди оператора на предсказаниях COMBO (на основе preview.py организаторов).

Запуск: python tracks/core/demo_queue.py [--pred COMBO]
Пишет report/demo_queue.jpg (шесть кадров) и report/demo_queue.csv (очередь подсказок по всем строкам).

Подсказки те же, что в preview.py: `occupied` при occ_prob >= 0.5; иначе `check_belongings`, если есть вещи;
иначе `check_dishes`, если есть чашки или посуда. Это применение прогнозов, а не новые метки.

Кадры выбираются правилом, без просмотра: в тесте самый «активный» кадр (сумма счётчиков) днём (11–17 МСК),
вечером (19–23 МСК), на ИК, и кадр с наибольшим числом подсказок на свободных столах; из train два кадра
с известными ошибками (out-of-fold предсказание рядом с ручными счётчиками).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from _bootstrap import ROOT  # noqa: E402
from contact_sheet import load_font  # noqa: E402

from ts import COUNTS, io, paths  # noqa: E402
from ts.rows import load_rows_ext  # noqa: E402

COLORS = {'occupied': '#ffb000', 'check_belongings': '#db2777', 'check_dishes': '#db2777', 'no_prompt': '#10b981'}
HINT_RU = {'occupied': 'занят', 'check_belongings': 'проверить вещи', 'check_dishes': 'убрать посуду', 'no_prompt': 'свободен'}
TRAIN_ERROR_FRAMES = ['im_cf13426b6f011b9cb35c', 'im_d59660ef92145770924d']   # ИК-компания из пяти; слот 6 у края кадра
TILE_W, TILE_H, CAPTION_H = 1280, 720, 196


def queue(pred: pd.DataFrame) -> pd.DataFrame:
    rows = load_rows_ext()
    rows = rows[rows['dt'] == 0][['row_id', 'frame_id', 'split', 'table_id', 'slot', 'x1', 'y1', 'x2', 'y2', 'hour_msk', 'is_ir']]
    q = rows.merge(pred, on='row_id', validate='one_to_one')
    q['operator_hint'] = np.select(
        [q.occ_prob >= .5, (q.occ_prob < .5) & (q['items'] >= .5), (q.occ_prob < .5) & ((q.cups + q.dishes) >= .5)],
        ['occupied', 'check_belongings', 'check_dishes'], default='no_prompt')
    return q


def pick_frames(q: pd.DataFrame) -> list[tuple[str, str]]:
    test = q[q.split == 'test']
    frames = test.groupby('frame_id').agg(activity=('people', 'sum'), hour=('hour_msk', 'first'), ir=('is_ir', 'first'))
    frames['activity'] = test.groupby('frame_id')[COUNTS].sum().sum(axis=1)
    frames['free_hints'] = test[test.operator_hint.isin(['check_belongings', 'check_dishes'])].groupby('frame_id').size()
    frames = frames.fillna(0).sort_index()

    def top(mask: pd.Series, column: str = 'activity') -> str:
        return frames[mask].sort_values(column, ascending=False, kind='stable').index[0]

    chosen = [('тест, день', top((frames.ir == 0) & frames.hour.between(11, 17))),
              ('тест, вечер', top((frames.ir == 0) & frames.hour.between(19, 23))),
              ('тест, ИК-кадр', top(frames.ir == 1))]
    used = {f for _, f in chosen}
    chosen.append(('тест, подсказки на свободных столах', top(~frames.index.isin(used), 'free_hints')))
    chosen += [('train, разбор ошибки: компания на ИК-кадре', TRAIN_ERROR_FRAMES[0]),
               ('train, разбор ошибки: стол у края кадра', TRAIN_ERROR_FRAMES[1])]
    return chosen


def tile(title: str, frame_id: str, q: pd.DataFrame, labels: pd.DataFrame, frames: pd.DataFrame) -> Image.Image:
    font, small = load_font(24), load_font(20)
    image = io.read_image(frame_id).resize((TILE_W, TILE_H))
    draw = ImageDraw.Draw(image)
    rows = q[q.frame_id == frame_id].sort_values('x1')
    for r in rows.itertuples():
        color = COLORS[r.operator_hint]
        draw.rectangle([r.x1, r.y1, r.x2, r.y2], outline=color, width=4)
        draw.text((r.x1 + 4, max(0, r.y1 - 30)), f'{r.slot}: {HINT_RU[r.operator_hint]}', fill=color, font=font,
                  stroke_width=3, stroke_fill='black')
    canvas = Image.new('RGB', (TILE_W, TILE_H + CAPTION_H), 'white')
    canvas.paste(image, (0, 0))
    text = ImageDraw.Draw(canvas)
    stamp = frames.loc[frame_id]
    text.text((8, TILE_H + 4), f'{title}   {stamp.ts.tz_convert("Europe/Moscow"):%d.%m %H:%M} МСК   {frame_id}', fill='black', font=font)
    for j, r in enumerate(rows.itertuples()):
        line = f'стол {r.slot}: ' + ' '.join(f'{c[:3]} {getattr(r, c):g}' for c in COUNTS) + f'  P={r.occ_prob:.2f}'
        if r.row_id in labels.index:
            truth = labels.loc[r.row_id]
            line += '   | истина: ' + ' '.join(f'{int(truth[c])}' for c in COUNTS)
        text.text((8 + (j % 2) * 640, TILE_H + 38 + (j // 2) * 26), line, fill='black', font=small)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred', default='COMBO', help='ART/preds/<имя>.csv, все 4964 строки (train out-of-fold)')
    parser.add_argument('--out', type=Path, default=ROOT / 'report' / 'demo_queue.jpg')
    args = parser.parse_args()
    pred = io.read_csv_exact(paths.ART / 'preds' / f'{args.pred}.csv')
    q = queue(pred)
    labels = io.load_labels().set_index('row_id')
    frames = io.load_frames().set_index('frame_id')
    chosen = pick_frames(q)
    tiles = [tile(title, frame_id, q, labels, frames) for title, frame_id in chosen]
    canvas = Image.new('RGB', (2 * TILE_W, ((len(tiles) + 1) // 2) * (TILE_H + CAPTION_H)), 'white')
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % 2) * TILE_W, (i // 2) * (TILE_H + CAPTION_H)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=85)
    shown = {frame_id: title for title, frame_id in chosen}
    table = q[(q.split == 'test') | q.frame_id.isin(shown)].copy()
    table['shown_as'] = table.frame_id.map(shown).fillna('')
    table = table.sort_values(['frame_id', 'slot'])[['row_id', 'frame_id', 'split', 'table_id', 'slot'] + COUNTS
                                                    + ['occ_prob', 'operator_hint', 'shown_as']]
    table.to_csv(args.out.with_suffix('.csv'), index=False)
    test = table[table.split == 'test']
    print(f'{args.out}: {len(tiles)} кадров; {args.out.with_suffix(".csv").name}: {len(table)} строк')
    print('подсказки в тесте:', test.operator_hint.value_counts().to_dict())
    for title, frame_id in chosen:
        print(f'  {title}: {frame_id}')


if __name__ == '__main__':
    main()
