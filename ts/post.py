"""Постобработка предсказаний по флагам `post` конфига (раздел 11.3). Задача A5, пока заглушка."""
from __future__ import annotations

import pandas as pd


def apply(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Применить включённые правила: `hands_need_people`, `hands_cap`, `round_counts`, `clip`, `empty_night`.

    Правила по `slot`, `frame_id`, `row_id` и времени суток запрещены без
    проверки на обоих фолдах.
    """
    raise NotImplementedError('ts.post.apply: задача A5')
