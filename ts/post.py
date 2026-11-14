"""Постобработка предсказаний по флагам `cfg['post']` (раздел 11.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import COUNTS

DEFAULTS = {
    'hands_need_people': False,   # people = 0 -> hands = 0
    'hands_cap': False,           # hands <= 2 * people
    'round_counts': False,        # округление счётчиков до целого
    'clip': True,                 # счётчики в [0, 50], occ_prob в [0, 1]
    'empty_night': False,         # не реализовано: в тесте есть ночные компании
    # Приор вещей: за занятым столом почти всегда есть хотя бы одна вещь (train: 1.13 на занятую строку).
    'items_prior': False,         # people >= 1 и items = 0 -> items = 1, пока источников items нет
    # Когда источники items есть (stack.counts.items или stack.vote.items заданы), приор действует
    # как нижняя граница только при этом флаге; включать после проверки по правилу 12.3.
    'items_prior_floor': False,
    # Множитель подносов: dishes = тарелки + миски + tray_mult · подносы. Применяется в
    # ts.stack.predict_counts (там есть число подносов), здесь только значение по умолчанию.
    'tray_mult': 1.0,
}


def _cfg(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {}).get('post', {})}


def has_items_source(cfg: dict | None) -> bool:
    """Есть ли в сборке источник вещей (детекции или голосование по признакам строк)."""
    stack = (cfg or {}).get('stack', {})
    return bool(stack.get('counts', {}).get('items')) or bool((stack.get('vote') or {}).get('items'))


def apply(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Применить включённые правила к таблице `row_id, people, hands, cups, dishes, items, occ_prob`.

    Порядок: приор вещей, руки без людей, ограничение рук, округление, обрезка.
    Правил по `slot`, `frame_id`, `row_id` и времени суток здесь нет (11.3).
    """
    p = _cfg(cfg)
    out = pred.copy()
    if p['empty_night']:
        raise NotImplementedError('post.empty_night не реализован и не проверен (раздел 11.3)')
    if p['items_prior'] and (p['items_prior_floor'] or not has_items_source(cfg)):
        lift = (out['people'] >= 1) & (out['items'] < 1)
        out.loc[lift, 'items'] = 1.0
    if p['hands_need_people']:
        out.loc[out['people'] <= 0, 'hands'] = 0.0
    if p['hands_cap']:
        out['hands'] = np.minimum(out['hands'], 2 * out['people'])
    if p['round_counts']:
        out[COUNTS] = out[COUNTS].round()
    if p['clip']:
        out[COUNTS] = out[COUNTS].clip(0, 50)
        out['occ_prob'] = out['occ_prob'].clip(0, 1)
    return out
