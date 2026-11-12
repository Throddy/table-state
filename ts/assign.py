"""Привязка объектов к столам (разделы 7.6 и 9). Задача A3, пока заглушка."""
from __future__ import annotations

import pandas as pd


def assign(dets: pd.DataFrame, rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Привязать детекции одного источника к строкам `rows_ext`.

    `dets`: детекции формата 8.1. `rows`: строки `rows_ext` (формат 8.3).
    `cfg`: параметры зон и порогов из `configs/default.yaml`.
    Возвращает длинную таблицу `row_id, dt, det_id, counter, weight`, где
    каждый объект привязан не больше чем к одному столу в пределах кадра
    (правила 9.2 для малых объектов и 9.3 для людей).
    """
    raise NotImplementedError('ts.assign.assign: задача A3')
