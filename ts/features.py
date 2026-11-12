"""Сборка признаков строк из источников (раздел 7.7). Задачи A4–A5, пока заглушка."""
from __future__ import annotations

import pandas as pd


def build_features(sources: list[str], cfg: dict) -> pd.DataFrame:
    """Признаки всех 4964 строк по списку источников.

    `sources`: имена артефактов `dets/*` и `rowfeats/*`, прошедших `ts.check`.
    Возвращает таблицу с индексом `row_id` (4964 строки).
    """
    raise NotImplementedError('ts.features.build_features: задачи A4–A5')
