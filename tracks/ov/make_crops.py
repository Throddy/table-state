"""C1: кропы столов для 231 train-строки и контактный лист непустых столов.

Запуск: python tracks/ov/make_crops.py [--force]

Пишет в ART/crops/:
  ov_train__v1/top/<row_id>.png    зона top, увеличение ×4 (ts.crops.crop_table, Lanczos)
  ov_train__v1/wide/<row_id>.png   зона wide, увеличение ×2.5
  ov_train__v1/sheet_p<N>.jpg      контактный лист: строки train с ненулевыми счётчиками
  ov_train__v1.parquet             индекс: строка × вид кропа, область и масштаб для обратного пересчёта
  ov_train__v1.json                карточка (раздел 8.6)

Только целевые train-кадры (dt = 0, 25–26.10). PNG без потерь: файл совпадает с тем, что вернёт
`crop_table`. Готовый файл нужного размера не перезаписывается (перезапись: --force).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from _bootstrap import git_sha  # noqa: E402  (добавляет корень репозитория в sys.path)
from contact_sheet import load_font  # noqa: E402

from ts import COUNTS, io, paths  # noqa: E402
from ts.crops import CropMeta, crop_table, to_crop_coords  # noqa: E402
from ts.rows import load_rows_ext  # noqa: E402

NAME = 'ov_train__v1'
KINDS = {'top': 4.0, 'wide': 2.5}   # вид кропа -> увеличение (раздел 9.1)
SHEET_COLS, SHEET_ROWS = 2, 2       # ячеек на странице контактного листа
CAPTION_H, GAP = 58, 12
BOX_COLOR = '#ffd60a'


def train_rows() -> pd.DataFrame:
    """231 строка train с `dt = 0`, сверенная с `train_labels.csv`."""
    rows = load_rows_ext()
    rows = rows[(rows.split == 'train') & (rows.dt == 0)].reset_index(drop=True)
    labels = io.load_labels()
    if rows.row_id.duplicated().any() or set(rows.row_id) != set(labels.row_id):
        raise SystemExit('train-строки rows_ext не совпадают с train_labels.csv')
    if not (rows.src_frame_id == rows.frame_id).all() or set(rows.date_utc) - set(labels.date_utc):
        raise SystemExit('в выборке есть кадры не из train-дней')
    return rows


def make_crops(rows: pd.DataFrame, out_dir: Path, force: bool) -> tuple[pd.DataFrame, dict, int]:
    """Сохранить кропы. Возвращает индекс, `CropMeta` по ключу `(row_id, kind)` и число записанных файлов."""
    records, metas, written = [], {}, 0
    for kind in KINDS:
        (out_dir / kind).mkdir(parents=True, exist_ok=True)
    for r in rows.itertuples():
        box = (r.x1, r.y1, r.x2, r.y2)
        for kind, scale in KINDS.items():
            crop, meta = crop_table(r.src_frame_id, box, kind=kind, scale=scale, slot=r.slot)
            path = out_dir / kind / f'{r.row_id}.png'
            if force or not path.is_file() or Image.open(path).size != crop.size:
                crop.save(path)
                written += 1
            metas[r.row_id, kind] = meta
            records.append({
                'row_id': r.row_id, 'frame_id': r.frame_id, 'src_frame_id': r.src_frame_id, 'dt': r.dt,
                'date_utc': r.date_utc, 'table_id': r.table_id, 'slot': r.slot, 'is_ir': r.is_ir,
                'x1': r.x1, 'y1': r.y1, 'x2': r.x2, 'y2': r.y2,
                'kind': kind, 'scale': scale, 'file': f'{out_dir.name}/{kind}/{path.name}',
                'crop_x0': meta.x0, 'crop_y0': meta.y0, 'crop_x1': meta.x1, 'crop_y1': meta.y1,
                'sx': meta.sx, 'sy': meta.sy, 'width': meta.width, 'height': meta.height,
            })
    return pd.DataFrame(records), metas, written


def load_marked(path: Path, meta: CropMeta) -> Image.Image:
    """Сохранённый кроп с тонкой рамкой целевого стола (рисуется только на листе)."""
    image = Image.open(path).convert('RGB')
    x1, y1, x2, y2 = to_crop_coords(meta.table_box, meta)[0]
    ImageDraw.Draw(image).rectangle([x1, y1, x2 - 1, y2 - 1], outline=BOX_COLOR, width=1)
    return image


def make_sheets(rows: pd.DataFrame, metas: dict, out_dir: Path) -> int:
    """Контактный лист строк с ненулевыми счётчиками: кропы top и wide, подпись со счётчиками train."""
    labels = io.load_labels().set_index('row_id')
    frames = io.load_frames().set_index('frame_id')
    busy = rows[rows.row_id.map(labels[COUNTS].sum(axis=1)) > 0]
    busy = busy.assign(ts=busy.frame_id.map(frames.ts)).sort_values(['ts', 'slot', 'row_id'])
    sizes = {kind: [(metas[row_id, kind].width, metas[row_id, kind].height) for row_id in busy.row_id]
             for kind in KINDS}
    top_w, wide_w = (max(w for w, _ in sizes[kind]) for kind in ('top', 'wide'))
    cell_w = top_w + GAP + wide_w
    cell_h = CAPTION_H + max(h for kind in KINDS for _, h in sizes[kind])
    font = load_font(20)

    for old in out_dir.glob('sheet_p*.jpg'):
        old.unlink()
    per_page = SHEET_COLS * SHEET_ROWS
    n_pages = -(-len(busy) // per_page)
    for page in range(n_pages):
        canvas = Image.new('RGB', (SHEET_COLS * (cell_w + GAP) + GAP, SHEET_ROWS * (cell_h + GAP) + GAP), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(busy.iloc[page * per_page:(page + 1) * per_page].itertuples()):
            x = GAP + (i % SHEET_COLS) * (cell_w + GAP)
            y = GAP + (i // SHEET_COLS) * (cell_h + GAP)
            lab = labels.loc[r.row_id]
            when = (r.ts + pd.Timedelta(hours=3)).strftime('%d.%m %H:%M')
            draw.text((x, y), f'{r.row_id}  slot {r.slot} ({r.table_id})  {when} MSK'
                              f'{"  IR" if r.is_ir else ""}  {r.frame_id}', fill='black', font=font)
            draw.text((x, y + 26), '  '.join(f'{c}={int(lab[c])}' for c in COUNTS)
                      + f'   occupied={int(lab.occupied)}', fill='#b00020', font=font)
            canvas.paste(load_marked(out_dir / 'top' / f'{r.row_id}.png', metas[r.row_id, 'top']),
                         (x, y + CAPTION_H))
            canvas.paste(load_marked(out_dir / 'wide' / f'{r.row_id}.png', metas[r.row_id, 'wide']),
                         (x + top_w + GAP, y + CAPTION_H))
        canvas.save(out_dir / f'sheet_p{page + 1}.jpg', quality=92)
    return len(busy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='перезаписать уже готовые кропы')
    args = parser.parse_args()

    rows = train_rows()
    out_dir = paths.art('crops', NAME)
    index, metas, written = make_crops(rows, out_dir, args.force)
    for kind in KINDS:
        files = {p.stem for p in (out_dir / kind).glob('*.png')}
        if files != set(rows.row_id):
            raise SystemExit(f'{kind}: файлы кропов не совпадают со списком строк ({len(files)} из {len(rows)})')
    index_path = paths.art('crops', f'{NAME}.parquet')
    index.to_parquet(index_path, index=False)
    n_busy = make_sheets(rows, metas, out_dir)

    frame_ids = sorted(rows.src_frame_id.unique())
    sha, _ = git_sha()
    card = {
        'name': NAME, 'kind': 'crops', 'owner': 'C',
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'command': 'python tracks/ov/make_crops.py',
        'models': [],
        'inputs': ['rows_ext.parquet', 'train_labels.csv'],
        'coverage': 'train_only',
        'frames_processed': {'n': len(frame_ids), 'sha1': hashlib.sha1('\n'.join(frame_ids).encode()).hexdigest()},
        'rows': len(rows),
        'crops': {kind: {'scale': scale, 'n': int((index.kind == kind).sum())} for kind, scale in KINDS.items()},
        'manual_labels_used': False,
        'uses_train_counters': False,
        'notes': 'ts.crops.crop_table без резкости, PNG без потерь; train-счётчики только в подписях '
                 'контактного листа sheet_p*.jpg (строки с ненулевыми счётчиками)',
    }
    index_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')
    print(f'{NAME}: строк {len(rows)}, кропов {len(index)} (записано {written}, пропущено {len(index) - written}), '
          f'на контактном листе {n_busy} строк -> {out_dir}')


if __name__ == '__main__':
    main()
