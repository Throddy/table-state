"""Зоны стола top / seat / wide."""
from __future__ import annotations

from typing import Sequence

from . import FRAME_H, FRAME_W

Box = tuple[float, float, float, float]

# Расширение рамки в долях её ширины (left, right) и высоты (up, down).
ZONE_PARAMS: dict[str, dict[str, float]] = {
    'top': {'left': 0.08, 'right': 0.08, 'up': 0.08, 'down': 0.08},
    'seat': {'left': 0.45, 'right': 0.45, 'up': 1.10, 'down': 0.90},
    'wide': {'left': 0.45, 'right': 0.45, 'up': 1.10, 'down': 0.90},
}
# Зоны, которые обрезаются по кадру уже при вычислении.
CLIPPED_KINDS = {'wide'}
# Единственное место для параметров, зависящих от слота: {slot: {kind: {side: value}}}.
SLOT_OVERRIDES: dict[int, dict[str, dict[str, float]]] = {}


def clip_box(box: Sequence[float], width: float = FRAME_W, height: float = FRAME_H) -> Box:
    """Обрезать бокс `x1, y1, x2, y2` по границам кадра."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return (min(max(x1, 0.0), width), min(max(y1, 0.0), height),
            min(max(x2, 0.0), width), min(max(y2, 0.0), height))


def zone(box: Sequence[float], kind: str, slot: int | None = None) -> Box:
    """Зона стола в пикселях кадра.

    `box`: рамка стола `x1, y1, x2, y2`. `kind`: `'top'` (столешница, рамка
    +8% во все стороны), `'seat'` (где сидят люди) или `'wide'` (`seat`,
    обрезанная по кадру; кроп для VLM и людей). `slot` выбирает коэффициенты
    из `SLOT_OVERRIDES`, если они заданы. `top` и `seat` могут выходить за
    кадр: это зоны для геометрии, по кадру их обрезает `ts.crops`.
    """
    if kind not in ZONE_PARAMS:
        raise ValueError(f"Неизвестная зона '{kind}', допустимы: {sorted(ZONE_PARAMS)}")
    params = dict(ZONE_PARAMS[kind])
    if slot is not None:
        params.update(SLOT_OVERRIDES.get(int(slot), {}).get(kind, {}))
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    out = (x1 - params['left'] * w, y1 - params['up'] * h,
           x2 + params['right'] * w, y2 + params['down'] * h)
    return clip_box(out) if kind in CLIPPED_KINDS else out
