"""Контекст ±30 секунд (разделы 7.7 и 10)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rows import load_rows_ext

COUNTERS = ['people', 'hands', 'cups', 'dishes', 'items']
STATS = ['cur', 'prev', 'next', 'med', 'max', 'min', 'conf_max', 'conf_sum']

DEFAULTS = {
    'min_conf': {},              # {counter: порог confidence}, по умолчанию без порога
    'people_max_area': None,     # бокс человека крупнее (px²) считается идущим или стоящим в проходе
    'people_rescue': True,       # учитывать людей, привязанных правилом 'rescue' (ts.assign)
}


def _cfg(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {}).get('temporal', {})}


def apply_filters(assigned: pd.DataFrame, cfg: dict | None) -> pd.DataFrame:
    """Оставить объекты, проходящие пороги `cfg['temporal']`."""
    p = _cfg(cfg)
    keep = np.ones(len(assigned), dtype=bool)
    for counter, thr in (p['min_conf'] or {}).items():
        keep &= ~((assigned['counter'] == counter) & (assigned['confidence'] < thr)).to_numpy()
    people = (assigned['counter'] == 'people').to_numpy()
    if p['people_max_area'] is not None:
        keep &= ~(people & (assigned['area'] > p['people_max_area']).to_numpy())
    if not p['people_rescue']:
        keep &= ~(people & (assigned['rule'] == 'rescue').to_numpy())
    return assigned[keep]


def context_median(cur: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Правило раздела 10 для статичных классов.

    Три кадра: медиана. Один сосед: максимум из двух, только если оба больше
    нуля, иначе текущий. Без соседей: текущий. Отсутствующий сосед это NaN.
    """
    has_prev, has_next = ~np.isnan(prev), ~np.isnan(nxt)
    out = cur.astype(float).copy()
    both = has_prev & has_next
    out[both] = np.median(np.stack([cur[both], prev[both], nxt[both]]), axis=0)
    for has, other in [(has_prev & ~has_next, prev), (has_next & ~has_prev, nxt)]:
        use = has & (cur > 0) & (np.nan_to_num(other) > 0)
        out[use] = np.maximum(cur[use], other[use])
    return out


def aggregate(assigned: pd.DataFrame, cfg: dict, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Свернуть привязанные объекты в признаки строки.

    `assigned`: выход `ts.assign.assign` для одного источника. `cfg`: пороги в
    `cfg['temporal']`. `rows`: строки `rows_ext` (по умолчанию весь индекс);
    по ним определяется, какие соседи есть у строки.

    Возвращает по одной строке на `row_id` (индекс) с колонками
    `<source>__<counter>__{cur,prev,next,med,max,min,conf_max,conf_sum}`.
    `prev` и `next` равны NaN, если соседнего кадра нет (нули не подставляются).
    `med` считается по правилу `context_median`; `max`, `min` по доступным
    кадрам; `conf_max`, `conf_sum` по объектам текущего кадра.
    """
    rows = load_rows_ext() if rows is None else rows
    source = str(assigned['source'].iloc[0]) if len(assigned) else (cfg or {}).get('source', 'src')
    kept = apply_filters(assigned, cfg)
    available = rows.pivot_table(index='row_id', columns='dt', values='slot', aggfunc='size')
    available = available.reindex(columns=[-30, 0, 30]).notna()
    counts = kept.groupby(['row_id', 'dt', 'counter']).size()
    current = kept[kept['dt'] == 0].groupby(['row_id', 'counter'])['confidence'].agg(['max', 'sum'])
    out = pd.DataFrame(index=available.index)
    for counter in COUNTERS:
        def at(dt: int) -> np.ndarray:
            try:
                values = counts.xs((dt, counter), level=('dt', 'counter')).reindex(out.index, fill_value=0)
            except KeyError:
                values = pd.Series(0, index=out.index)
            return np.where(available[dt].to_numpy(), values.to_numpy(float), np.nan)
        cur, prev, nxt = at(0), at(-30), at(30)
        stack = np.stack([cur, prev, nxt])
        prefix = f'{source}__{counter}__'
        out[prefix + 'cur'], out[prefix + 'prev'], out[prefix + 'next'] = cur, prev, nxt
        out[prefix + 'med'] = context_median(cur, prev, nxt)
        out[prefix + 'max'] = np.nanmax(stack, axis=0)
        out[prefix + 'min'] = np.nanmin(stack, axis=0)
        try:
            conf = current.xs(counter, level='counter').reindex(out.index, fill_value=0.0)
            out[prefix + 'conf_max'], out[prefix + 'conf_sum'] = conf['max'].to_numpy(), conf['sum'].to_numpy()
        except KeyError:
            out[prefix + 'conf_max'], out[prefix + 'conf_sum'] = 0.0, 0.0
    return out
