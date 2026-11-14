"""Картинки для слайдов защиты, только train-кадры (25–26.10): tracks/det/figs/*.jpg.

Запуск: python tracks/det/make_figs.py
  hands_*.jpg   столешница ×4: руки прод-детектора (жёлтые) и ft_yolo__v1 с confidence >= 0.15 (голубые);
  error_*.jpg   два типичных случая из ERRORS.md.
Подписи к картинкам: tracks/det/REPORT.md.
"""
from __future__ import annotations

import pandas as pd
from PIL import ImageDraw, ImageFont

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # модуль common лежит в tracks/det
import common

from ts import io, paths
from ts.crops import crop_region, to_crop_coords
from ts.geometry import zone
from ts.rows import load_rows_ext

OUT = common.ROOT / 'tracks' / 'det' / 'figs'
SCALE = 4.0
FT_MIN = 0.15
COLORS = {'prod': '#ffd60a', 'ft_yolo': '#00e5ff'}
BOX = ['x1', 'y1', 'x2', 'y2']
# (имя файла, кадр, слоты, классы, источники)
FIGS = [
    ('hands_1_slot1_laptop', 'im_88fd0d1f0f8d2daba005', [1], ['hand'], ['prod', 'ft_yolo']),
    ('hands_2_slot1_evening', 'im_a2d4da367c833c6a05d4', [1], ['hand'], ['prod', 'ft_yolo']),
    ('hands_3_slots1_2', 'im_f37ded0672c3a357918a', [1, 2], ['hand'], ['prod', 'ft_yolo']),
    ('error_1_glass_empty_table', 'im_817c518154e6821104e9', [6], ['cup'], ['prod', 'ft_yolo']),
    ('error_2_ir_group', 'im_cf13426b6f011b9cb35c', [2], ['person'], ['prod']),
]


def font(size: int):
    for name in ['DejaVuSans.ttf', 'Arial.ttf', 'Helvetica.ttc']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    common.require_data()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows_ext()
    rows = rows[(rows['dt'] == 0) & (rows['split'] == 'train')]
    dets = {'prod': io.load_prod_dets(), 'ft_yolo': pd.read_parquet(paths.ART / 'dets' / 'ft_yolo__v1.parquet')}
    dets['ft_yolo'] = dets['ft_yolo'][dets['ft_yolo']['confidence'] >= FT_MIN]
    for name, frame_id, slots, classes, sources in FIGS:
        tables = rows[(rows['frame_id'] == frame_id) & rows['slot'].isin(slots)]
        if tables.empty:
            raise SystemExit(f'{frame_id}: нет строк слотов {slots} (это не train-кадр?)')
        zones = [zone((t.x1, t.y1, t.x2, t.y2), 'wide' if 'person' in classes else 'top', int(t.slot)) for t in tables.itertuples()]
        region = (min(z[0] for z in zones) - 12, min(z[1] for z in zones) - 30, max(z[2] for z in zones) + 12, max(z[3] for z in zones) + 12)
        crop, meta = crop_region(io.read_image(frame_id), region, 2.5 if 'person' in classes else SCALE, frame_id=frame_id)
        draw = ImageDraw.Draw(crop)
        for t in tables.itertuples():
            draw.rectangle(to_crop_coords([[t.x1, t.y1, t.x2, t.y2]], meta)[0].tolist(), outline='#22c55e', width=2)
        for k, source in enumerate(sources):
            d = dets[source]
            d = d[(d['frame_id'] == frame_id) & d['class_name'].isin(classes)]
            for r in d.itertuples():
                box = to_crop_coords([[r.x1, r.y1, r.x2, r.y2]], meta)[0]
                if box[2] < 0 or box[3] < 0 or box[0] > crop.width or box[1] > crop.height:
                    continue
                box = [box[0] - 2 * k, box[1] - 2 * k, box[2] + 2 * k, box[3] + 2 * k]
                draw.rectangle(box, outline=COLORS[source], width=3)
                draw.text((box[0], max(0, box[1] - 20 - 0 * k) if k == 0 else box[3] + 2), f'{source} {r.confidence:.2f}',
                          fill=COLORS[source], font=font(17), stroke_width=2, stroke_fill='black')
        crop.save(OUT / f'{name}.jpg', quality=92)
        print(f'{name}.jpg: {crop.size}, {frame_id}, слоты {slots}')


if __name__ == '__main__':
    main()
