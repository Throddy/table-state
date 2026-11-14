"""Метрика кейса: обёртка над копией score.py организаторов."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import COUNTS, PRED_COLUMNS
from .vendor import score as vscore


def score_frame(labels: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Оценить предсказания на строках `labels`.

    `labels`: строки с `row_id` и пятью счётчиками (как `train_labels.csv`).
    `pred`: таблица с `row_id`, пятью счётчиками и `occ_prob`; лишние строки и
    колонки допустимы. Возвращает словарь: `score`, `occupancy_ap`, `rows`,
    `occupied`, а также `quality`, `abs_error`, `objects` по классам.
    Считает `vendor/score.breakdown`, формула организаторов не дублируется.
    """
    if isinstance(pred.index, pd.Index) and pred.index.name == 'row_id' and 'row_id' not in pred:
        pred = pred.reset_index()
    result = vscore.breakdown(labels, pred[PRED_COLUMNS])
    counters = result['counters']
    return {
        'score': result['score'],
        'occupancy_ap': result['occupancy_ap'],
        'rows': result['rows'],
        'occupied': result['occupied_tables'],
        'quality': {c: counters[c]['quality'] for c in COUNTS},
        'abs_error': {c: counters[c]['absolute_error'] for c in COUNTS},
        'objects': {c: counters[c]['objects'] for c in COUNTS},
    }


def fast_score(y: np.ndarray, p: np.ndarray, prob: np.ndarray) -> float:
    """Та же формула на массивах, для bootstrap (допускает повторы строк).

    `y`, `p`: массивы `(n, 5)` истинных и предсказанных счётчиков в порядке
    `ts.COUNTS`, `prob`: `occ_prob`. Совпадение с `score_frame` проверяется тестом.
    """
    error = np.abs(y - p).sum(axis=0)
    support = y.sum(axis=0)
    denominator = np.where(support > 0, support, len(y))
    quality = np.maximum(0.0, 1.0 - error / denominator)
    occupied = (y[:, 0] > 0).astype(int)
    if occupied.sum() == 0:
        ap = float(1.0 - prob.mean())
    elif occupied.sum() == len(occupied):
        ap = 1.0
    else:
        ap = float(average_precision_score(occupied, prob))
    return float(0.2 * ap + 0.8 * quality.mean())
