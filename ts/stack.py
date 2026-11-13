"""Стекер: правила счётчиков и модель `occ_prob` (раздел 11)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import COUNTS, PRED_COLUMNS, SEED
from .temporal import aggregate

DEFAULTS = {
    'source': 'prod',
    # какой столбец агрегата идёт в счётчик: cur (текущий кадр) или med (правило раздела 10)
    'counts': {'people': 'cur', 'hands': 'cur', 'cups': 'cur', 'dishes': 'cur', 'items': None},
    'people_fill': True,   # правило раздела 10 для people: 0 сейчас, одинаково у соседей, есть руки
    'occ_C': 1.0,
    'area_grid': [30000 + 2500 * i for i in range(21)],   # сетка порога площади бокса человека, px²
}


def _cfg(cfg: dict | None) -> dict:
    user = (cfg or {}).get('stack', {})
    out = {**DEFAULTS, **user}
    out['counts'] = {**DEFAULTS['counts'], **user.get('counts', {})}
    return out


def predict_counts(features: pd.DataFrame, cfg: dict | None) -> pd.DataFrame:
    """Счётчики по правилам 11.1 и раздела 10 (без подгонки), индекс `row_id`."""
    p = _cfg(cfg)
    src = p['source']
    out = pd.DataFrame(index=features.index)
    for counter in COUNTS:
        column = p['counts'].get(counter)
        out[counter] = features[f'{src}__{counter}__{column}'].to_numpy(float) if column else 0.0
    if p['people_fill']:
        prev, nxt = features[f'{src}__people__prev'], features[f'{src}__people__next']
        fill = (out['people'] == 0) & (prev > 0) & (prev == nxt) & (features[f'{src}__hands__cur'] > 0)
        out.loc[fill, 'people'] = prev[fill]
    return out


def occ_features(features: pd.DataFrame, counts: pd.DataFrame, cfg: dict | None) -> pd.DataFrame:
    """Признаки занятости (11.2): люди сейчас и у соседей, уверенность, руки, чашки, ИК."""
    src = _cfg(cfg)['source']
    cur = features[f'{src}__people__cur']
    neighbours = features[[f'{src}__people__prev', f'{src}__people__next']]
    x = pd.DataFrame(index=features.index)
    x['people'] = counts['people'].clip(upper=3)
    x['people_ctx'] = neighbours.clip(upper=3).mean(axis=1).fillna(cur.clip(upper=3))
    x['people_ctx_any'] = (neighbours.max(axis=1).fillna(cur) > 0).astype(float)
    x['person_conf'] = features[f'{src}__people__conf_max']
    x['hands'] = features[f'{src}__hands__cur'].clip(upper=2)
    x['cups'] = features[f'{src}__cups__med'].clip(upper=2)
    x['is_ir'] = features['is_ir']
    return x


def fit(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> dict:
    """Подогнать модель занятости (11.2) на train-счётчиках.

    `features`: `ts.features.build_features` (берутся строки из `labels`).
    `labels`: строки `train_labels.csv` дня или дней подгонки. Счётчики в R1
    считаются правилами без подгонки, поэтому модель содержит только логрег.
    """
    p = _cfg(cfg)
    ids = labels['row_id'].to_numpy()
    counts = predict_counts(features.loc[ids], cfg)
    x = occ_features(features.loc[ids], counts, cfg)
    y = (labels['people'].to_numpy() > 0).astype(int)
    if len(np.unique(y)) < 2:
        return {'occ': None, 'prior': float(y.mean()), 'columns': list(x.columns)}
    model = LogisticRegression(C=p['occ_C'], max_iter=2000, random_state=SEED)
    model.fit(x.to_numpy(float), y)
    return {'occ': model, 'columns': list(x.columns), 'coef': dict(zip(x.columns, model.coef_[0].round(4).tolist())),
            'intercept': float(model.intercept_[0])}


def predict(model: dict, features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Предсказать `row_id, people, hands, cups, dishes, items, occ_prob` моделью `fit`.

    Гарантии 11.2: при `people >= 1` значение `occ_prob` не ниже
    `0.5 + 0.01·min(people, 5)` (сжатие в интервал выше порога сохраняет порядок),
    плюс детерминированный микросдвиг для разрыва совпадений.
    """
    counts = predict_counts(features, cfg)
    x = occ_features(features, counts, cfg)[model['columns']]
    if model['occ'] is None:
        prob = np.full(len(x), model['prior'])
    else:
        prob = model['occ'].predict_proba(x.to_numpy(float))[:, 1]
    floor = np.where(counts['people'] >= 1, 0.5 + 0.01 * np.minimum(counts['people'], 5), 0.0)
    prob = floor + (1.0 - floor) * prob
    tie = x.to_numpy(float) @ np.linspace(1.0, 2.0, x.shape[1]) + features['slot'].to_numpy(float) * 1e-3
    rank = pd.Series(tie).rank(method='first').to_numpy() / len(tie)
    out = counts.copy()
    out['occ_prob'] = np.clip(prob * (1 - 2e-6) + 1e-6 * rank, 0.0, 1.0)
    return out.reset_index()[PRED_COLUMNS]


def tune_people_area(assigned: pd.DataFrame, labels: pd.DataFrame, cfg: dict, rows: pd.DataFrame) -> tuple[float, dict]:
    """Подобрать порог площади бокса человека по сумме |ошибки| people на `labels`.

    Сетка `cfg['stack']['area_grid']`. При равенстве берётся середина самого
    длинного плато оптимальных значений (11.1), а не край.
    Возвращает `(порог, {порог: ошибка})`.
    """
    p = _cfg(cfg)
    truth = labels.set_index('row_id')['people']
    part_rows = rows[rows['row_id'].isin(truth.index)]
    part = assigned[assigned['row_id'].isin(truth.index)]
    errors = {}
    for area in p['area_grid']:
        local = {**cfg, 'temporal': {**cfg.get('temporal', {}), 'people_max_area': area}}
        counts = predict_counts(aggregate(part, local, part_rows).join(
            part_rows[part_rows.dt == 0].set_index('row_id')[['is_ir']]), cfg)
        errors[area] = float((counts['people'].reindex(truth.index) - truth).abs().sum())
    best = min(errors.values())
    runs, run = [], []
    for area in p['area_grid']:
        if errors[area] == best:
            run.append(area)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    longest = max(runs, key=len)
    return float(longest[len(longest) // 2]), errors
