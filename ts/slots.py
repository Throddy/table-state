"""Физические места столов (слоты) по координате центра рамки."""
from __future__ import annotations

import bisect

SLOT_EDGES = [0, 275, 450, 690, 925, 1140, 1300]
N_SLOTS = len(SLOT_EDGES) - 1


def slot_of(cx: float) -> int:
    """Номер слота 1..6 по `cx` центра рамки стола.

    Интервалы полуоткрытые `[edge_i, edge_{i+1})`. Значения вне диапазона
    прижимаются к крайним слотам.
    """
    slot = bisect.bisect_right(SLOT_EDGES, float(cx))
    return min(max(slot, 1), N_SLOTS)
