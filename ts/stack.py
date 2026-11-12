"""Стекер: пороги счётчиков и модель `occ_prob` (раздел 11). Задачи A3 и A5, пока заглушка."""
from __future__ import annotations

import pandas as pd


def fit(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> dict:
    """Подогнать пороги по классам (11.1) и логрег занятости (11.2) на train-счётчиках.

    `features`: выход `ts.features.build_features` для строк подгонки.
    `labels`: строки `train_labels.csv`. Возвращает сериализуемую модель.
    """
    raise NotImplementedError('ts.stack.fit: задачи A3 и A5')


def predict(model: dict, features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Предсказать `row_id, people, hands, cups, dishes, items, occ_prob` моделью `fit`."""
    raise NotImplementedError('ts.stack.predict: задачи A3 и A5')
