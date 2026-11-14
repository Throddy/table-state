"""Стекер: правила счётчиков и модель `occ_prob` (раздел 11)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import COUNTS, PRED_COLUMNS, SEED
from .temporal import aggregate, merge_sources

DEFAULTS = {
    'source': 'all',       # объединённый список объектов всех источников детекций (ts.temporal.merge_sources)
    # какой столбец агрегата идёт в счётчик: cur (текущий кадр) или med (правило раздела 10)
    'counts': {'people': 'cur', 'hands': 'cur', 'cups': 'cur', 'dishes': 'cur', 'items': None},
    # items только при people >= 1 или если вещи есть на обоих соседних кадрах (оставленная вещь)
    'items_need_people': False,
    'people_fill': True,   # правило раздела 10 для people: 0 сейчас, одинаково у соседей, есть руки
    # голосование с признаками строк (11.1): {counter: {column: '<source>__<имя>', mode: median | replace | max}}
    'vote': {},
    'tune_people_area': False,   # фильтр площади бокса person на Public вреден (контракт 1.7), включается только явно
    'tune_conf': True,     # подбирать пороги confidence для источников, кроме prod
    'conf_grid': [round(0.05 * i, 2) for i in range(1, 19)],
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
    # Множитель подносов (post.tray_mult): dishes = тарелки + миски + tray_mult · подносы.
    # Пока тарелок никто не детектирует, множитель 2 заменяет их (поднос обычно несёт посуду); когда
    # появятся детекции plate и bowl, множитель ставится в 1 и остаётся запасным вариантом.
    tray_mult = float((cfg or {}).get('post', {}).get('tray_mult') or 1.0)
    tray_column = f'{src}__tray__cur'
    if tray_mult != 1.0 and p['counts'].get('dishes') and tray_column in features:
        out['dishes'] = out['dishes'] + (tray_mult - 1.0) * features[tray_column].to_numpy(float)
    for counter, rule in (p['vote'] or {}).items():
        column, mode = rule['column'], rule.get('mode', 'median')
        value = features[column].to_numpy(float)
        ok_column = column.split('__')[0] + '__ok'
        ok = (value >= 0) & (features[ok_column].to_numpy(float) == 1 if ok_column in features else True)
        detector = out[counter].to_numpy(float)
        context = features[f'{src}__{counter}__med'].to_numpy(float)
        voted = {'median': np.median(np.stack([detector, value, context]), axis=0),
                 'replace': value, 'max': np.maximum(detector, value)}[mode]
        out[counter] = np.where(ok, voted, detector)
    if p['people_fill']:
        prev, nxt = features[f'{src}__people__prev'], features[f'{src}__people__next']
        fill = (out['people'] == 0) & (prev > 0) & (prev == nxt) & (features[f'{src}__hands__cur'] > 0)
        out.loc[fill, 'people'] = prev[fill]
    if p['items_need_people'] and p['counts'].get('items'):
        prev, nxt = features[f'{src}__items__prev'], features[f'{src}__items__next']
        stays = (prev > 0) & (nxt > 0)     # NaN (соседа нет или источник посчитан только для dt = 0) даёт False
        out.loc[(out['people'] < 1) & ~stays, 'items'] = 0.0
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


def _plateau_middle(grid: list, errors: dict) -> float:
    """Середина самого длинного плато оптимальных значений сетки (11.1)."""
    best = min(errors.values())
    runs, run = [], []
    for value in grid:
        if errors[value] == best:
            run.append(value)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    longest = max(runs, key=len)
    return float(longest[len(longest) // 2])


def _count_error(parts: list[pd.DataFrame], labels: pd.DataFrame, cfg: dict, rows: pd.DataFrame, counter: str) -> float:
    """Сумма |ошибки| счётчика на строках `labels` при текущих порогах `cfg`."""
    truth = labels.set_index('row_id')[counter]
    part_rows = rows[rows['row_id'].isin(truth.index)]
    local = [a[a['row_id'].isin(truth.index)] for a in parts]
    features = aggregate(merge_sources(local, cfg), {**cfg, 'source': 'all'}, part_rows)
    features = features.join(part_rows[part_rows['dt'] == 0].set_index('row_id')[['is_ir']])
    counts = predict_counts(features, {**cfg, 'stack': {**cfg.get('stack', {}), 'vote': {}}})
    return float((counts[counter].reindex(truth.index) - truth).abs().sum())


def tune_people_area(parts: list[pd.DataFrame], labels: pd.DataFrame, cfg: dict, rows: pd.DataFrame) -> tuple[float, dict]:
    """Подобрать порог площади бокса человека по сумме |ошибки| people на `labels`.

    `parts`: привязанные объекты по источникам (`ts.features.assigned_for`).
    Сетка `cfg['stack']['area_grid']`, при равенстве середина плато.
    Возвращает `(порог, {порог: ошибка})`.
    """
    grid = _cfg(cfg)['area_grid']
    errors = {}
    for area in grid:
        local = {**cfg, 'temporal': {**cfg.get('temporal', {}), 'people_max_area': area}}
        errors[area] = _count_error(parts, labels, local, rows, 'people')
    return _plateau_middle(grid, errors), errors


def tune_conf(parts: list[pd.DataFrame], labels: pd.DataFrame, cfg: dict, rows: pd.DataFrame) -> dict:
    """Подобрать пороги confidence по одному на (источник, счётчик) для источников, кроме `prod`.

    Покоординатно, в порядке источников: сетка `conf_grid`, критерий это сумма
    |ошибки| счётчика на `labels` при объединении со всеми остальными
    источниками, при равенстве середина плато. Возвращает
    `{source: {counter: порог}}` (вкладывается в `cfg['temporal']['min_conf']`).
    """
    p = _cfg(cfg)
    used = {c for c, column in p['counts'].items() if column}
    min_conf = json.loads(json.dumps(cfg.get('temporal', {}).get('min_conf', {})))
    for part in parts:
        if not len(part) or part['source'].iloc[0] == 'prod':
            continue
        source = str(part['source'].iloc[0])
        for counter in [c for c in COUNTS if c in used and (part['counter'] == c).any()]:
            errors = {}
            for thr in p['conf_grid']:
                trial = {**min_conf, source: {**min_conf.get(source, {}), counter: thr}}
                local = {**cfg, 'temporal': {**cfg.get('temporal', {}), 'min_conf': trial}}
                errors[thr] = _count_error(parts, labels, local, rows, counter)
            min_conf.setdefault(source, {})[counter] = _plateau_middle(p['conf_grid'], errors)
    return min_conf


def make_fit_predict(cfg: dict, log: list | None = None):
    """Функция `fit_predict(train_rows, predict_rows)` для `ts.cv` по источникам из `cfg['sources']`.

    На строках подгонки: пороги confidence новых источников, порог площади
    бокса человека, логрег занятости. Затем предсказание и `ts.post.apply`.
    Колонки, для которых в `cfg['compose']` указано `baseline`, заменяются
    колонками бейзлайна организаторов (`ts.orgbase`), подогнанного на тех же строках.
    В `log` дописываются подобранные параметры каждого вызова.
    """
    from . import post
    from .features import build_features, source_parts, split_sources
    from .rows import load_rows_ext

    sources = list(cfg['sources'].get('dets', [])) + list(cfg['sources'].get('rowfeats', []))
    dets, _ = split_sources(sources)
    rows = load_rows_ext()
    parts = source_parts(sources, cfg)
    p = _cfg(cfg)
    counters = {'n': 0}

    def fit_predict(train_rows: pd.DataFrame, predict_rows: pd.DataFrame) -> pd.DataFrame:
        local = json.loads(json.dumps(cfg))
        local.setdefault('temporal', {})
        tuned = 0
        if p['tune_conf'] and any(source != 'prod' for source in dets):
            local['temporal']['min_conf'] = tune_conf(parts, train_rows, local, rows)
            tuned += sum(len(v) for k, v in local['temporal']['min_conf'].items() if isinstance(v, dict))
        if p['tune_people_area']:
            local['temporal']['people_max_area'], _ = tune_people_area(parts, train_rows, local, rows)
            tuned += 1
        features = build_features(sources, local)
        model = fit(features, train_rows, local)
        counters['n'] = tuned + (len(model['columns']) + 1 if model['occ'] is not None else 0)
        fit_predict.n_params = counters['n']
        if log is not None:
            log.append({'fit_dates': sorted(train_rows['date_utc'].unique()), 'n_fit_rows': len(train_rows),
                        'people_max_area': local['temporal'].get('people_max_area'),
                        'min_conf': local['temporal'].get('min_conf', {}),
                        'occ_coef': model.get('coef'), 'occ_intercept': model.get('intercept')})
        raw = predict(model, features.loc[predict_rows['row_id']], local)
        limits = (cfg['sources'].get('counters') or {})
        if limits:
            # Источник с ограничением sources.counters меняет только перечисленные колонки: остальные
            # (включая people, от которого зависят правила по рукам, и occ_prob) считаются сборкой без него.
            keep_new = {c for counters_ in limits.values() for c in counters_}
            rest = [source for source in sources if source not in limits]
            base_features = build_features(rest, local)
            base_raw = predict(fit(base_features, train_rows, local), base_features.loc[predict_rows['row_id']], local)
            for column in [c for c in PRED_COLUMNS[1:] if c not in keep_new]:
                raw[column] = base_raw[column].to_numpy(float)
        pred = post.apply(raw, local)
        compose = cfg.get('compose') or {}
        unknown = set(compose.values()) - {'stack', 'baseline', 'baseline_max', 'baseline_avg'}
        if unknown:
            raise ValueError(f'compose: неизвестный источник колонки {sorted(unknown)}, '
                             'допустимы stack, baseline, baseline_max и baseline_avg')
        from_baseline = [c for c, origin in compose.items() if origin in ('baseline', 'baseline_max', 'baseline_avg')]
        if from_baseline:
            from . import orgbase
            base = orgbase.calibrated(train_rows, pred['row_id'].tolist()).set_index('row_id').loc[pred['row_id']]
            for column in from_baseline:
                values = base[column].to_numpy(float)
                # baseline_max: максимум из колонки бейзлайна и своей сборки (новый источник только добавляет объекты)
                own = pred[column].to_numpy(float)
                # baseline_avg: среднее колонки бейзлайна и своей сборки без округления. Ошибка метрики это модуль,
                # он выпуклый: среднее двух прогнозов не хуже среднего их ошибок, а при ошибках разного знака лучше обоих.
                pred[column] = {'baseline': values, 'baseline_max': np.maximum(values, own),
                                'baseline_avg': 0.5 * values + 0.5 * own}[compose[column]]
        return pred

    fit_predict.n_params = None
    return fit_predict
