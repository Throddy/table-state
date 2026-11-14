"""Привязка объектов к столам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .check import CLASS_TO_COUNTER
from .geometry import zone

ASSIGNED_COLUMNS = ['row_id', 'dt', 'det_id', 'counter', 'weight',
                    'source', 'class_name', 'confidence', 'area', 'rule', 'score',
                    'x1', 'y1', 'x2', 'y2', 'persist', 'ctx_n']
NEGATIVE_CLASSES = [c for c, counter in CLASS_TO_COUNTER.items() if counter is None and c != 'table']
AREA_SHARE_CLASSES = {'bag', 'personal_item'}

DEFAULTS = {
    'small': {
        'hand_max_above': 0.05,      # центр кисти не выше y1 - 0.05h (9.2.4)
        'bag_min_share': 0.60,       # доля площади bag / personal_item внутри зоны top (9.2.5)
        'dedup_iou': 0.5,            # 9.2.6
        'dedup_small_px': 40.0,
        'dedup_center_ratio': 0.5,
        'negative_iou': 0.5,         # подавление негативным классом (8.2)
    },
    'people': {
        'min_table_overlap': 0.15,   # пересечение / площадь стола, как в бейзлайне организаторов (9.3.5)
        'rescue_min': 0.25,          # доля нижней трети бокса человека внутри зоны top
        # выбор стола при доборе, если порог прошли несколько столов:
        # 'horizontal' = стол, к рамке которого центр нижней трети бокса ближе по горизонтали
        #                (внутри рамки расстояние 0), при равенстве большая доля; 'share' = большая доля
        'rescue_pick': 'horizontal',
    },
}


def _cfg(cfg: dict | None) -> dict:
    """Параметры привязки: значения по умолчанию, поверх них `cfg['assign']`."""
    user = (cfg or {}).get('assign', {})
    return {group: {**values, **user.get(group, {})} for group, values in DEFAULTS.items()}


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Площади пересечений боксов `a (n, 4)` и `b (m, 4)`: матрица `(n, m)`."""
    wh = np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    return np.clip(wh, 0, None).prod(axis=-1)


def _area(a: np.ndarray) -> np.ndarray:
    return np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = _inter(a, b)
    return inter / np.maximum(_area(a)[:, None] + _area(b)[None, :] - inter, 1e-9)


def _suppress_negatives(dets: pd.DataFrame, iou_thr: float) -> np.ndarray:
    """Маска детекций счётных классов, не перекрытых более уверенным негативом (8.2)."""
    keep = np.ones(len(dets), dtype=bool)
    negative = dets['class_name'].isin(NEGATIVE_CLASSES).to_numpy()
    if not negative.any() or negative.all():
        return keep & ~negative
    boxes = dets[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
    conf = dets['confidence'].to_numpy(float)
    hit = (_iou(boxes[~negative], boxes[negative]) >= iou_thr) & (conf[negative][None, :] > conf[~negative][:, None])
    keep[np.flatnonzero(~negative)[hit.any(axis=1)]] = False
    return keep & ~negative


def _dedup(boxes: np.ndarray, conf: np.ndarray, p: dict) -> np.ndarray:
    """Индексы объектов после дедупликации 9.2.6 (остаётся самый уверенный)."""
    order = np.argsort(-conf, kind='stable')
    kept: list[int] = []
    for i in order:
        duplicate = False
        for j in kept:
            a, b = boxes[i], boxes[j]
            small = min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
            if small < p['dedup_small_px']:
                dist = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
                duplicate = dist < p['dedup_center_ratio'] * small
            else:
                duplicate = _iou(a[None], b[None])[0, 0] >= p['dedup_iou']
            if duplicate:
                break
        if not duplicate:
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def _assign_small(d: pd.DataFrame, tables: np.ndarray, tops: np.ndarray, p: dict) -> list[tuple[int, int, str, float]]:
    """Малые объекты (9.2): список `(индекс детекции, индекс стола, правило, оценка)`."""
    boxes = d[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
    classes = d['class_name'].to_numpy()
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    inside = ((cx[:, None] >= tops[None, :, 0]) & (cx[:, None] <= tops[None, :, 2]) &
              (cy[:, None] >= tops[None, :, 1]) & (cy[:, None] <= tops[None, :, 3]))
    area = np.maximum(_area(boxes), 1e-9)
    share_table = _inter(boxes, tables) / area[:, None]
    share_top = _inter(boxes, tops) / area[:, None]
    h = tables[:, 3] - tables[:, 1]
    is_hand = classes == 'hand'
    inside &= ~(is_hand[:, None] & (cy[:, None] < tables[None, :, 1] - p['hand_max_above'] * h[None, :]))
    needs_share = np.isin(classes, list(AREA_SHARE_CLASSES))
    inside &= ~(needs_share[:, None] & (share_top < p['bag_min_share']))
    tcx, tcy = (tables[:, 0] + tables[:, 2]) / 2, (tables[:, 1] + tables[:, 3]) / 2
    dist = np.hypot(cx[:, None] - tcx[None, :], cy[:, None] - tcy[None, :])
    out = []
    for i in np.flatnonzero(inside.any(axis=1)):
        cand = np.flatnonzero(inside[i])
        best = min(cand, key=lambda j: (-round(float(share_table[i, j]), 6), float(dist[i, j])))
        out.append((int(i), int(best), 'top', float(share_table[i, best])))
    return out


def _assign_people(d: pd.DataFrame, tables: np.ndarray, tops: np.ndarray, p: dict) -> list[tuple[int, int, str, float]]:
    """Люди (9.3): сначала правило бейзлайна, затем добор сидящих за дальней стороной.

    1. Владелец это стол с максимальным `пересечение / площадь стола`, если оно
       не ниже `min_table_overlap` (правило организаторов, 9.3.5).
    2. Иначе (человек почти не закрывает рамку, типично для сидящих на диване
       за дальней стороной): кандидаты это столы, в зону `top` которых попадает
       не меньше `rescue_min` нижней трети бокса человека. Из нескольких
       кандидатов выбирается по `rescue_pick` (см. `DEFAULTS`).
    """
    boxes = d[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
    overlap = _inter(boxes, tables) / np.maximum(_area(tables), 1e-9)[None, :]
    lower = boxes.copy()
    lower[:, 1] = boxes[:, 3] - (boxes[:, 3] - boxes[:, 1]) / 3
    lower_share = _inter(lower, tops) / np.maximum(_area(lower), 1e-9)[:, None]
    out = []
    for i in range(len(boxes)):
        j = int(overlap[i].argmax())
        if overlap[i, j] >= p['min_table_overlap']:
            out.append((i, j, 'overlap', float(overlap[i, j])))
            continue
        cand = np.flatnonzero(lower_share[i] >= p['rescue_min'])
        if not len(cand):
            continue
        if p['rescue_pick'] == 'horizontal' and len(cand) > 1:
            # Доля пересечения занижена у стола, обрезанного краем кадра (слот 6), поэтому при нескольких
            # кандидатах решает горизонтальное расстояние от центра нижней трети бокса до рамки стола.
            cx = (lower[i, 0] + lower[i, 2]) / 2
            gap = np.maximum(np.maximum(tables[cand, 0] - cx, cx - tables[cand, 2]), 0.0)
            j = int(min(cand, key=lambda k: (round(float(gap[list(cand).index(k)]), 6), -float(lower_share[i, k]))))
        else:
            j = int(cand[lower_share[i, cand].argmax()])
        out.append((i, j, 'rescue', float(lower_share[i, j])))
    return out


def _assign_frame(d: pd.DataFrame | None, t: pd.DataFrame, dt: int, p: dict) -> pd.DataFrame | None:
    """Привязать детекции одного кадра-источника к рамкам столов целевого кадра."""
    if d is None:
        return None
    d = d[_suppress_negatives(d, p['small']['negative_iou'])]
    if d.empty:
        return None
    t = t.sort_values('table_id', kind='stable')
    tables = t[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
    tops = np.array([zone(b, 'top', s) for b, s in zip(tables, t['slot'])])
    row_ids = t['row_id'].to_numpy()
    person = (d['class_name'] == 'person').to_numpy()
    links = []
    for part, fn, group in [(d[person], _assign_people, 'people'), (d[~person], _assign_small, 'small')]:
        if len(part):
            links += [(part.iloc[i], j, rule, score) for i, j, rule, score in fn(part, tables, tops, p[group])]
    if not links:
        return None
    frame = pd.DataFrame([{
        'row_id': row_ids[j], 'dt': dt, 'det_id': r.det_id, 'counter': CLASS_TO_COUNTER[r.class_name],
        'weight': 1.0, 'source': r.source, 'class_name': r.class_name, 'confidence': float(r.confidence),
        'area': float((r.x2 - r.x1) * (r.y2 - r.y1)), 'rule': rule, 'score': score,
        'x1': float(r.x1), 'y1': float(r.y1), 'x2': float(r.x2), 'y2': float(r.y2)} for r, j, rule, score in links])
    return frame.loc[dedup_objects(frame, p['small'])].reset_index(drop=True)


def dedup_objects(frame: pd.DataFrame, p: dict | None = None) -> list:
    """Индексы строк после дедупликации 9.2.6 внутри каждой группы `(row_id, dt, counter)`.

    Для людей критерий один: IoU не ниже `dedup_iou`. Используется и внутри
    источника, и при объединении источников (`ts.temporal.merge_sources`).
    """
    p = p or DEFAULTS['small']
    keep: list = []
    for key, g in frame.groupby(['row_id', 'dt', 'counter'], sort=False):
        if len(g) == 1:
            keep += g.index.tolist()
            continue
        params = {**p, 'dedup_small_px': 0.0} if key[2] == 'people' else p
        idx = _dedup(g[['x1', 'y1', 'x2', 'y2']].to_numpy(float), g['confidence'].to_numpy(float), params)
        keep += g.index[idx].tolist()
    return sorted(keep)


def assign(dets: pd.DataFrame, rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Привязать детекции одного источника к строкам `rows_ext`.

    `dets`: детекции формата 8.1. `rows`: строки `rows_ext` (формат 8.3); для
    строки с `dt != 0` берутся детекции кадра `src_frame_id` и рамка целевого
    кадра. `cfg`: конфиг, параметры в `cfg['assign']` (см. `DEFAULTS`).

    Возвращает длинную таблицу `row_id, dt, det_id, counter, weight` и
    служебные колонки `source, class_name, confidence, area, rule, score`,
    бокс `x1, y1, x2, y2`, а для людей `persist` (на скольких других кадрах
    тройки «целевой + соседи» есть person с IoU >= 0.5) и `ctx_n` (сколько
    других кадров в тройке доступно).
    Каждый объект привязан не больше чем к одному столу в пределах пары
    (целевой кадр, кадр-источник). Фильтры по `confidence`, размеру бокса
    человека и правилу привязки применяет `ts.temporal.aggregate`, чтобы их
    можно было подбирать без повторной привязки. `weight` всегда 1.0.
    """
    p = _cfg(cfg)
    counted = dets[dets['class_name'] != 'table']
    by_frame = {frame_id: g for frame_id, g in counted.groupby('frame_id', sort=False)}
    records = []
    for _, frame_rows in rows.groupby('frame_id', sort=False):
        sources = frame_rows.drop_duplicates('dt').set_index('dt')['src_frame_id'].to_dict()
        people_boxes = {}
        for dt, src in sources.items():
            d = by_frame.get(src)
            part = d[d['class_name'] == 'person'] if d is not None else None
            people_boxes[dt] = part[['x1', 'y1', 'x2', 'y2']].to_numpy(float) if part is not None else np.zeros((0, 4))
        for dt, src in sources.items():
            frame = _assign_frame(by_frame.get(src), frame_rows[frame_rows['dt'] == dt], int(dt), p)
            if frame is None:
                continue
            others = [boxes for other, boxes in people_boxes.items() if other != dt]
            person = (frame['counter'] == 'people').to_numpy()
            persist = np.zeros(len(frame), dtype=int)
            if person.any():
                own = frame.loc[person, ['x1', 'y1', 'x2', 'y2']].to_numpy(float)
                persist[person] = sum((_iou(own, boxes) >= 0.5).any(axis=1).astype(int) for boxes in others if len(boxes))
            frame['persist'] = persist
            frame['ctx_n'] = len(others)
            records.append(frame[ASSIGNED_COLUMNS])
    if not records:
        return pd.DataFrame(columns=ASSIGNED_COLUMNS)
    return pd.concat(records, ignore_index=True)
