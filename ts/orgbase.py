"""Колонки бейзлайна организаторов (`baseline.py` из каталога данных, режим calibrated) без изменений.

Нужны, пока своя колонка не доказала преимущество заменой на Public (раздел 12.5 контракта).
Код бейзлайна не копируется: модуль импортируется из `TS_DATA` как есть.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import pandas as pd

from . import PRED_COLUMNS, paths


@lru_cache(maxsize=1)
def _module():
    if str(paths.DATA) not in sys.path:
        sys.path.insert(0, str(paths.DATA))
    import baseline
    return baseline


@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    """Признаки бейзлайна (`baseline.make_features`) по всем 4964 строкам, индекс `row_id`."""
    return _module().make_features(paths.DATA)


def calibrated(train_rows: pd.DataFrame, predict_ids: list[str]) -> pd.DataFrame:
    """Предсказание режима calibrated: подгонка на `train_rows`, ответ для `predict_ids`.

    Возвращает `row_id`, пять счётчиков и `occ_prob`. Значения `occ_prob` и `items`
    между запусками могут отличаться на величину порядка 1e-16 (порядок сложения
    в ExtraTrees при `n_jobs=2`), счётчики детектора воспроизводятся точно.
    """
    pred, _ = _module().calibrated_prediction(features(), train_rows, list(predict_ids))
    return pred.reset_index()[PRED_COLUMNS]
