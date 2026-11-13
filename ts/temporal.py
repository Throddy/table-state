"""Контекст ±30 секунд (разделы 7.7 и 10)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rows import load_rows_ext

COUNTERS = ['people', 'hands', 'cups', 'dishes', 'items']
STATS = ['cur', 'prev', 'next', 'med', 'max', 'min', 'conf_max', 'conf_sum']

DEFAULTS = {
    # пороги confidence: {counter: порог} для всех источников или {source: {counter: порог}}
    'min_conf': {},
    'people_max_area': None,     # бокс человека крупнее (px²) считается идущим или стоящим в проходе
    'people_rescue': True,       # учитывать людей, привязанных правилом 'rescue' (ts.assign)
    # исключения из фильтра площади (стоящий вплотную гость по правилам считается):
    'people_big_keep_with_hand': False,     # в зоне top этого стола на том же кадре есть рука
    'people_big_keep_persistent': False,    # тот же человек (IoU >= 0.5) есть на обоих соседних кадрах
    # двойной счёт посуды (covered_dishes): тарелка или миска на подносе (IoU не ниже порога или центр внутри
    # бокса подноса) и миска, совпадающая с чашкой, в dishes не идут
    'bowl_overlap_iou': 0.3,
}


def _cfg(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {}).get('temporal', {})}


def _conf_mask(assigned: pd.DataFrame, min_conf: dict) -> np.ndarray:
    """Маска объектов, проходящих пороги confidence (плоские и по источникам)."""
    keep = np.ones(len(assigned), dtype=bool)
    origin = assigned['origin'] if 'origin' in assigned else assigned['source']
    for key, value in (min_conf or {}).items():
        if isinstance(value, dict):
            for counter, thr in value.items():
                keep &= ~((origin == key) & (assigned['counter'] == counter) & (assigned['confidence'] < thr)).to_numpy()
        else:
            keep &= ~((assigned['counter'] == key) & (assigned['confidence'] < value)).to_numpy()
    return keep


def apply_filters(assigned: pd.DataFrame, cfg: dict | None) -> pd.DataFrame:
    """Оставить объекты, проходящие пороги `cfg['temporal']`."""
    p = _cfg(cfg)
    keep = _conf_mask(assigned, p['min_conf'])
    people = (assigned['counter'] == 'people').to_numpy()
    if not p['people_rescue']:
        keep &= ~(people & (assigned['rule'] == 'rescue').to_numpy())
    if p['people_max_area'] is not None:
        big = people & (assigned['area'] > p['people_max_area']).to_numpy()
        exempt = np.zeros(len(assigned), dtype=bool)
        if p['people_big_keep_with_hand'] and big.any():
            hands = assigned[(assigned['counter'] == 'hands').to_numpy() & keep]
            with_hand = set(zip(hands['row_id'], hands['dt']))
            exempt |= np.array([(r, d) in with_hand for r, d in zip(assigned['row_id'], assigned['dt'])], dtype=bool)
        if p['people_big_keep_persistent']:
            exempt |= ((assigned['ctx_n'] == 2) & (assigned['persist'] == 2)).to_numpy()
        keep &= ~(big & ~exempt)
    return assigned[keep]


def covered_dishes(frame: pd.DataFrame, iou_thr: float | None) -> np.ndarray:
    """Маска тарелок и мисок, которые уже посчитаны другим объектом той же строки и кадра.

    - тарелка или миска на подносе: IoU с подносом не ниже `iou_thr` или её центр внутри бокса подноса.
      Поднос идёт в dishes с множителем `post.tray_mult` (2 подтверждено на Public: 1 и 3 хуже на 0.042),
      то есть посуда на нём уже учтена;
    - миска, совпадающая с чашкой (IoU не ниже `iou_thr`): COCO-детектор часто называет чашку миской.
    """
    from .assign import _iou
    mask = np.zeros(len(frame), dtype=bool)
    if not iou_thr or not len(frame) or not frame['class_name'].isin(['plate', 'bowl']).any():
        return mask
    position = {index: i for i, index in enumerate(frame.index)}
    box = ['x1', 'y1', 'x2', 'y2']
    for _, g in frame.groupby(['row_id', 'dt'], sort=False):
        dishes = g[g['class_name'].isin(['plate', 'bowl'])]
        trays, cups = g[g['class_name'] == 'tray'], g[g['class_name'] == 'cup']
        if dishes.empty or (trays.empty and cups.empty):
            continue
        d = dishes[box].to_numpy(float)
        hit = np.zeros(len(dishes), dtype=bool)
        if len(trays):
            t = trays[box].to_numpy(float)
            cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
            inside = ((cx[:, None] >= t[None, :, 0]) & (cx[:, None] <= t[None, :, 2]) &
                      (cy[:, None] >= t[None, :, 1]) & (cy[:, None] <= t[None, :, 3]))
            hit |= (_iou(d, t) >= iou_thr).any(axis=1) | inside.any(axis=1)
        if len(cups):
            is_bowl = (dishes['class_name'] == 'bowl').to_numpy()
            hit |= is_bowl & (_iou(d, cups[box].to_numpy(float)) >= iou_thr).any(axis=1)
        for index in dishes.index[hit]:
            mask[position[index]] = True
    return mask


def merge_sources(parts: list[pd.DataFrame], cfg: dict | None, name: str = 'all') -> pd.DataFrame:
    """Объединить привязанные объекты нескольких источников в один список (9.4).

    К каждому источнику применяются его пороги confidence, затем объекты
    одного счётчика в одной строке дедуплицируются по 9.2.6 (остаётся самый
    уверенный), тарелки и миски на подносе и миски-чашки отбрасываются
    (`covered_dishes`). Результат получает `source = name`, исходное имя источника
    сохраняется в `origin`, число согласных источников в `n_src`.
    """
    from .assign import DEFAULTS as ASSIGN_DEFAULTS, _iou, dedup_objects
    p = _cfg(cfg)
    parts = [a[_conf_mask(a, p['min_conf'])] for a in parts if len(a)]
    if not parts:
        return pd.DataFrame(columns=['row_id', 'dt', 'counter', 'confidence', 'source', 'origin', 'n_src'])
    merged = pd.concat(parts, ignore_index=True)
    merged['origin'] = merged['source']
    small = {**ASSIGN_DEFAULTS['small'], **(cfg or {}).get('assign', {}).get('small', {})}
    kept = merged.loc[dedup_objects(merged, small)].copy()
    kept = kept[~covered_dishes(kept, p['bowl_overlap_iou'])].copy()
    kept['n_src'] = 1
    if len(parts) > 1:
        groups = {key: g for key, g in merged.groupby(['row_id', 'dt', 'counter'], sort=False)}
        counts = []
        for r in kept.itertuples():
            g = groups[(r.row_id, r.dt, r.counter)]
            near = _iou(np.array([[r.x1, r.y1, r.x2, r.y2]]), g[['x1', 'y1', 'x2', 'y2']].to_numpy(float))[0] >= 0.3
            counts.append(int(g.loc[near, 'origin'].nunique()) or 1)
        kept['n_src'] = counts
    kept['source'] = name
    return kept.reset_index(drop=True)


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
    # число подносов на текущем кадре отдельно: для правила post.tray_mult (ts.stack.predict_counts)
    trays = (kept[(kept['dt'] == 0) & (kept['class_name'] == 'tray')].groupby('row_id').size()
             if len(kept) and 'class_name' in kept else pd.Series(dtype=float))
    out[f'{source}__tray__cur'] = trays.reindex(out.index, fill_value=0).to_numpy(float)
    return out
