"""Контекст ±30 секунд (разделы 7.7 и 10). Задача A4, пока заглушка."""
from __future__ import annotations

import pandas as pd


def aggregate(assigned: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Свернуть привязанные объекты в признаки строки.

    `assigned`: выход `ts.assign.assign`. Возвращает по одной строке на
    `row_id` с колонками
    `<source>__<counter>__{cur,prev,next,med,max,min,conf_max,conf_sum}`.
    Отсутствующий сосед не заменяется нулями.
    """
    raise NotImplementedError('ts.temporal.aggregate: задача A4')
