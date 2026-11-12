"""Кропы зон стола с обратным пересчётом координат (раздел 7.5)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from . import io
from .geometry import clip_box, zone


@dataclass(frozen=True)
class CropMeta:
    """Связь координат кропа и кадра: `x_frame = x_crop / sx + x0`.

    `x0, y0, x1, y1`: вырезанная область кадра в пикселях (целые, внутри кадра).
    `sx, sy`: фактическое увеличение по осям. `width, height`: размер кропа.
    `table_box`: рамка стола в координатах кадра.
    """
    frame_id: str
    kind: str
    slot: int | None
    x0: int
    y0: int
    x1: int
    y1: int
    sx: float
    sy: float
    width: int
    height: int
    table_box: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        """Словарь для сохранения в json или в колонки таблицы."""
        return asdict(self)


def crop_region(image: Image.Image, region: Sequence[float], scale: float = 4.0,
                frame_id: str = '', kind: str = 'custom', slot: int | None = None,
                table_box: Sequence[float] | None = None) -> tuple[Image.Image, CropMeta]:
    """Вырезать область уже загруженного кадра и увеличить её (Lanczos).

    Область обрезается по кадру и округляется наружу до целых пикселей.
    """
    x0f, y0f, x1f, y1f = clip_box(region, image.width, image.height)
    x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
    x1, y1 = int(math.ceil(x1f)), int(math.ceil(y1f))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'Пустая область кропа: {tuple(region)}')
    width = max(1, int(round((x1 - x0) * scale)))
    height = max(1, int(round((y1 - y0) * scale)))
    crop = image.crop((x0, y0, x1, y1)).resize((width, height), Image.Resampling.LANCZOS)
    box = tuple(float(v) for v in (table_box if table_box is not None else region))
    meta = CropMeta(frame_id=frame_id, kind=kind, slot=None if slot is None else int(slot),
                    x0=x0, y0=y0, x1=x1, y1=y1, sx=width / (x1 - x0), sy=height / (y1 - y0),
                    width=width, height=height, table_box=box)
    return crop, meta


def crop_table(frame_id: str, box: Sequence[float], kind: str = 'top', scale: float = 4.0,
               slot: int | None = None) -> tuple[Image.Image, CropMeta]:
    """Кроп зоны `kind` стола `box` с кадра `frame_id`, увеличенный в `scale` раз.

    Узкий кроп: `kind='top', scale=4.0`. Широкий: `kind='wide', scale=2.5`.
    Для строк `rows_ext` передавайте `src_frame_id` и рамку `x1, y1, x2, y2`.
    """
    image = io.read_image(frame_id)
    return crop_region(image, zone(box, kind, slot), scale, frame_id=frame_id,
                       kind=kind, slot=slot, table_box=box)


def to_frame_coords(boxes_in_crop: np.ndarray, meta: CropMeta) -> np.ndarray:
    """Боксы `(n, 4)` `x1, y1, x2, y2` из пикселей кропа в пиксели кадра."""
    b = np.asarray(boxes_in_crop, dtype=float).reshape(-1, 4)
    return b / np.array([meta.sx, meta.sy, meta.sx, meta.sy]) + np.array([meta.x0, meta.y0, meta.x0, meta.y0])


def to_crop_coords(boxes_in_frame: np.ndarray, meta: CropMeta) -> np.ndarray:
    """Обратное к `to_frame_coords`: боксы кадра в пиксели кропа (без обрезки)."""
    b = np.asarray(boxes_in_frame, dtype=float).reshape(-1, 4)
    return (b - np.array([meta.x0, meta.y0, meta.x0, meta.y0])) * np.array([meta.sx, meta.sy, meta.sx, meta.sy])
