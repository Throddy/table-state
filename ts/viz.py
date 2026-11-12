"""Общие визуализации для отчёта и разбора ошибок. Пока заглушка.

Контактные листы train-кадров уже делает `tools/contact_sheet.py`.
"""
from __future__ import annotations

import pandas as pd
from PIL import Image


def draw_rows(frame_id: str, rows: pd.DataFrame, dets: pd.DataFrame | None = None,
              pred: pd.DataFrame | None = None) -> Image.Image:
    """Кадр с рамками столов, детекциями и предсказаниями по строкам."""
    raise NotImplementedError('ts.viz.draw_rows: задача A7')


def error_sheet(pred: pd.DataFrame, labels: pd.DataFrame, top: int = 6) -> Image.Image:
    """Лист самых ошибочных train-строк (кроп, истина, предсказание)."""
    raise NotImplementedError('ts.viz.error_sheet: задача A7')
