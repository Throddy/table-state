"""Оценка open-vocab детекций на 231 train-строке по дням, ИК и без выброса (стенд трека C для C2 и C5).

Запуск:
  python tracks/ov/eval_dets.py ov_gdino__v1 [ov_owl__v1] [--items phone,paper] [--dishes plate,bowl,tray] [--errors]
  python tracks/ov/eval_dets.py ov_gdino__v1 ov_owl__v1 --rowfeats 1   # плюс ART/rowfeats/ov_counts__v1.parquet

С `--rowfeats N` счётчики каждого варианта записываются для всех 4964 строк (формат 8.4): по ним трек A может
заменять колонки items и dishes в сабмите. Тестовые строки считаются с порогом всего train, train-строки с порогом
другого дня.

Это диагностика трека, а не сборка решения: итоговые числа даёт стекер ядра
(`python tools/ablate.py` с источниками в `sources.dets`). Привязка здесь повторяет правила привязки малых объектов в том
виде, как их применяет `ts.assign` (в этой ветке он ещё может быть заглушкой, поэтому реализация временная):
- негативы: счётная детекция отбрасывается, если её перекрывает (IoU >= 0.5) более уверенный негатив источника;
- стол: центр бокса в зоне `top`; если зон две, стол с большей долей бокса в рамке, затем ближайший по центру;
- `bag` и `personal_item`: не меньше 60% площади бокса в зоне `top`;
- дедупликация внутри счётчика: IoU >= 0.5, для боксов с меньшей стороной до 40 px расстояние между центрами
  меньше половины меньшей стороны.
Контекст ±30 с не используется, только целевой кадр (`dt = 0`).

Варианты: один источник, объединение источников (`a|b`) и согласие двух источников (`a&b`: объект есть у обоих).
Для dishes ко всем вариантам добавляются прод-подносы (вариант `+prod`), чтобы сравнивать с посудой прод-детектора.
База сравнения для dishes: `2 × подносы` (`post.tray_mult = 2`, подтверждено на Public); `1 × подносы` для справки.
Пороги: сетка с шагом 0.05, критерий: суммарная абсолютная ошибка , при равенстве середина плато.
`best`: порог подобран на всех 231 строках. `oof`: порог подобран на одном дне, применён к другому .
Все числа качества считает `ts.metric.score_frame` (формула организаторов).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ts import PRED_COLUMNS, io, paths  # noqa: E402
from ts.check import CLASS_TO_COUNTER, check_dets, check_rowfeats  # noqa: E402
from ts.cv import FOLDS, OUTLIER_FRAME  # noqa: E402
from ts.geometry import zone  # noqa: E402
from ts.metric import score_frame  # noqa: E402
from ts.rows import load_rows_ext  # noqa: E402

GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)
NEGATIVES = [k for k, v in CLASS_TO_COUNTER.items() if v is None and k != 'table']
DEDUP_IOU, SMALL_PX, CENTER_RATIO, BAG_MIN_SHARE = 0.5, 40.0, 0.5, 0.60
BOX = ['x1', 'y1', 'x2', 'y2']
TRAY_MULT = 2.0                          # множитель подносов в сборке трека A (`post.tray_mult`)
ROWFEATS_COVERAGE = ('all', 'targets')   # с отладочным покрытием train_only счётчики по всем строкам не пишутся


def partial_coverage(cards: dict) -> list[str]:
    """Имена входных артефактов, покрытие которых не годится для `--rowfeats`."""
    return [name for name, card in cards.items() if card.get('coverage') not in ROWFEATS_COVERAGE]


def inter_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Площади пересечений боксов `a (n, 4)` и `b (m, 4)`."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    wh = np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    return np.clip(wh, 0, None).prod(axis=-1)


def area(a: np.ndarray) -> np.ndarray:
    return np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)


def same_object(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матрица `(n, m)` «один и тот же объект» по правилу 9.2.6."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    inter = inter_area(a, b)
    iou = inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)
    small = np.minimum((a[:, 2:] - a[:, :2]).min(axis=1)[:, None], (b[:, 2:] - b[:, :2]).min(axis=1)[None, :])
    dist = np.linalg.norm((a[:, None, :2] + a[:, None, 2:] - b[None, :, :2] - b[None, :, 2:]) / 2, axis=2)
    return np.where(small < SMALL_PX, dist < CENTER_RATIO * small, iou >= DEDUP_IOU)


def dedup(conf: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Индексы объектов после жадной дедупликации по убыванию уверенности."""
    order = np.argsort(-conf, kind='stable')
    same = same_object(boxes[order], boxes[order])
    keep = []
    for i in range(len(order)):
        if not any(same[i, j] for j in keep):
            keep.append(i)
    return order[keep]


def drop_covered(dets: pd.DataFrame) -> pd.DataFrame:
    """Детекции кадра без негативов и без счётных боксов, перекрытых более уверенным негативом."""
    negative = dets.class_name.isin(NEGATIVES).to_numpy()
    cand, neg = dets[~negative & (dets.class_name != 'table')], dets[negative]
    if len(cand) and len(neg):
        boxes, nboxes = cand[BOX].to_numpy(float), neg[BOX].to_numpy(float)
        inter = inter_area(boxes, nboxes)
        iou = inter / np.maximum(area(boxes)[:, None] + area(nboxes)[None, :] - inter, 1e-9)
        hit = (iou >= DEDUP_IOU) & (neg.confidence.to_numpy(float)[None, :] > cand.confidence.to_numpy(float)[:, None])
        cand = cand[~hit.any(axis=1)]
    return cand


def attach(dets: pd.DataFrame, rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Счётные детекции одного источника -> строки `rows` (`dt = 0`): `row_id` -> детекции стола."""
    by_frame = dict(tuple(dets.groupby('frame_id', sort=False)))
    out = {}
    for frame_id, t in rows.groupby('frame_id', sort=False):
        if frame_id not in by_frame:
            continue
        d = drop_covered(by_frame[frame_id])
        boxes, tables = d[BOX].to_numpy(float), t[BOX].to_numpy(float)
        tops = np.array([zone(b, 'top', s) for b, s in zip(tables, t.slot)])
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        inside = ((centers[:, None, :] >= tops[None, :, :2]) & (centers[:, None, :] <= tops[None, :, 2:])).all(axis=2)
        size = np.maximum(area(boxes), 1e-9)[:, None]
        needs_share = d.class_name.isin(['bag', 'personal_item']).to_numpy()
        inside &= ~(needs_share[:, None] & (inter_area(boxes, tops) / size < BAG_MIN_SHARE))
        share = np.round(inter_area(boxes, tables) / size, 6)
        dist = np.linalg.norm(centers[:, None, :] - (tables[None, :, :2] + tables[None, :, 2:]) / 2, axis=2)
        owner = np.array([min(np.flatnonzero(inside[i]), key=lambda j: (-share[i, j], dist[i, j]))
                          if inside[i].any() else -1 for i in range(len(d))], dtype=int)
        for j, row_id in enumerate(t.row_id):
            out[row_id] = d[owner == j]
    return out


def row_objects(dets: pd.DataFrame | None, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Объекты счётчика на одной строке после дедупликации: уверенность, боксы, классы."""
    if dets is None:
        return np.zeros(0), np.zeros((0, 4)), np.zeros(0, dtype=object)
    cand = dets[dets.class_name.isin(classes)]
    conf, boxes, names = cand.confidence.to_numpy(float), cand[BOX].to_numpy(float), cand.class_name.to_numpy()
    keep = dedup(conf, boxes)
    return conf[keep], boxes[keep], names[keep]


def pick(objects: dict, thr: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Объекты строки выше порогов. `objects`: источник -> (conf, boxes, classes), `thr`: источник -> порог.

    `mode='|'`: объединение источников с дедупликацией; `'&'`: объекты первого источника, найденные и вторым.
    """
    picked = [(c[c >= thr[s]], b[c >= thr[s]]) for s, (c, b, _) in objects.items()]
    if mode == '&':
        (ca, ba), (cb, bb) = picked
        hit = same_object(ba, bb).any(axis=1)
        return ca[hit], ba[hit]
    conf, boxes = np.concatenate([c for c, _ in picked]), np.vstack([b for _, b in picked])
    keep = dedup(conf, boxes)
    return conf[keep], boxes[keep]


def quality(labels: pd.DataFrame, rows: pd.DataFrame, counter: str, pred: np.ndarray, mask: np.ndarray) -> float:
    """1 − WAPE счётчика на подмножестве строк через `ts.metric.score_frame`."""
    frame = pd.DataFrame(0.0, index=range(len(rows)), columns=PRED_COLUMNS).assign(row_id=rows.row_id.to_numpy())
    frame[counter] = pred
    return score_frame(labels[labels.row_id.isin(rows.row_id[mask])], frame)['quality'][counter]


def best_threshold(errors: np.ndarray, combos: list) -> tuple:
    """Комбинация порогов с минимальной ошибкой; при равенстве середина множества лучших."""
    tied = [combos[i] for i in np.flatnonzero(errors == errors.min())]
    centre = np.mean(tied, axis=0)
    return min(tied, key=lambda t: (np.abs(np.array(t) - centre).sum(), t))


def train_rows() -> pd.DataFrame:
    rows = load_rows_ext()
    return rows[(rows.split == 'train') & (rows.dt == 0)].reset_index(drop=True)


def count_row(objects: dict, trays: tuple | None, thr: dict, mode: str) -> int:
    """Число объектов строки: источники по порогам, при `trays` плюс прод-подносы с дедупликацией."""
    conf, boxes = pick(objects, thr, mode)
    if trays is not None:
        conf, boxes = np.concatenate([conf, trays[0]]), np.vstack([boxes, trays[1]])
    return len(dedup(conf, boxes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dets', nargs='+', help='имена артефактов в ART/dets без расширения')
    for counter in ('items', 'dishes'):
        default = [k for k, v in CLASS_TO_COUNTER.items() if v == counter]
        parser.add_argument(f'--{counter}', default=','.join(default), help=f'классы счётчика {counter}')
    parser.add_argument('--errors', action='store_true', help='напечатать строки с ошибкой для вариантов одного источника')
    parser.add_argument('--rowfeats', type=int, metavar='N',
                        help='записать счётчики всех вариантов по 4964 строкам в ART/rowfeats/ov_counts__v<N>.parquet')
    args = parser.parse_args()

    scope = load_rows_ext()   # строки, для которых считаются счётчики: train или все целевые (для --rowfeats)
    scope = scope[scope.dt == 0].reset_index(drop=True)
    if args.rowfeats is None:
        scope = scope[scope.split == 'train'].reset_index(drop=True)
    train = np.flatnonzero(scope.split.to_numpy() == 'train')
    rows = scope.iloc[train].reset_index(drop=True)
    labels = io.load_labels()
    truth = labels.set_index('row_id').loc[rows.row_id]
    masks = {'all': np.ones(len(rows), bool), 'ir1': rows.is_ir.to_numpy() == 1, 'ir0': rows.is_ir.to_numpy() == 0,
             'robust': rows.frame_id.to_numpy() != OUTLIER_FRAME}
    day = {name: rows.date_utc.to_numpy() == val for name, (_, val) in FOLDS.items()}   # строки проверки фолда

    sources, cards = {}, {}
    for name in args.dets:
        path = paths.ART / 'dets' / f'{name}.parquet'
        check_dets(path)
        cards[name] = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        sources[name] = attach(pd.read_parquet(path), scope)
    if args.rowfeats is not None:
        partial = partial_coverage(cards)
        if partial:
            raise SystemExit(f'--rowfeats требует coverage из {ROWFEATS_COVERAGE} (детекции на всех '
                             f'{scope.frame_id.nunique()} целевых кадрах): {partial}')
    prod = attach(io.load_prod_dets().query("class_name == 'tray'"), scope)
    features, report = {}, {}

    print(f"{'счётчик':<8}{'вариант':<34}{'порог best':<14}{'best: all':>10}{'25.10':>7}{'26.10':>7}"
          f"{'oof: all':>10}{'val26':>7}{'val25':>7}{'robust':>8}{'ir1':>7}{'ir0':>7}  пороги oof (fit25, fit26)")
    for counter in ('items', 'dishes'):
        classes = getattr(args, counter).split(',')
        y = truth[counter].to_numpy(float)
        objects = {name: [row_objects(by_row.get(row_id), classes) for row_id in scope.row_id]
                   for name, by_row in sources.items()}
        trays = [row_objects(prod.get(row_id), ['tray']) for row_id in scope.row_id]
        variants = [((name,), '|') for name in sources]
        variants += [(pair, mode) for pair in itertools.combinations(sources, 2) for mode in ('|', '&')]
        if counter == 'dishes':
            for mult in (TRAY_MULT, 1.0):   # база сравнения: 2 × подносы; 1 × подносы для справки
                base = mult * np.array([len(trays[j][0]) for j in train], float)
                title = f'prod ({mult:g} × подносы)' + (', база' if mult == TRAY_MULT else '')
                print(f"{counter:<8}{title:<34}{'':<14}{quality(labels, rows, counter, base, masks['all']):>10.3f}"
                      f"{quality(labels, rows, counter, base, day['f26']):>7.3f}"
                      f"{quality(labels, rows, counter, base, day['f25']):>7.3f}")
        for names, mode in variants:
            for with_prod in ([False, True] if counter == 'dishes' else [False]):
                def count(j: int, combo: tuple) -> int:
                    return count_row({s: objects[s][j] for s in names}, trays[j] if with_prod else None,
                                     dict(zip(names, combo)), mode)

                combos = list(itertools.product(GRID, repeat=len(names)))
                preds = np.array([[count(j, combo) for j in train] for combo in combos], float)
                errors = np.abs(preds - y[None])
                best = best_threshold(errors.sum(axis=1), combos)
                fit = {name: best_threshold(errors[:, ~mask].sum(axis=1), combos) for name, mask in day.items()}
                p_best = preds[combos.index(best)]
                p_oof = np.where(day['f25'], preds[combos.index(fit['f25'])], preds[combos.index(fit['f26'])])
                title = mode.join(n.split('__')[0] for n in names) + ('+prod' if with_prod else '')
                scores = {'best_all': quality(labels, rows, counter, p_best, masks['all']),
                          'oof_all': quality(labels, rows, counter, p_oof, masks['all']),
                          'oof_val_2026-10-26': quality(labels, rows, counter, p_oof, day['f25']),
                          'oof_val_2026-10-25': quality(labels, rows, counter, p_oof, day['f26']),
                          'oof_robust': quality(labels, rows, counter, p_oof, masks['robust']),
                          'oof_ir1': quality(labels, rows, counter, p_oof, masks['ir1']),
                          'oof_ir0': quality(labels, rows, counter, p_oof, masks['ir0'])}
                print(f"{counter:<8}{title:<34}{'/'.join(f'{t:.2f}' for t in best):<14}{scores['best_all']:>10.3f}"
                      f"{quality(labels, rows, counter, p_best, day['f26']):>7.3f}"
                      f"{quality(labels, rows, counter, p_best, day['f25']):>7.3f}{scores['oof_all']:>10.3f}"
                      f"{scores['oof_val_2026-10-26']:>7.3f}{scores['oof_val_2026-10-25']:>7.3f}"
                      f"{scores['oof_robust']:>8.3f}{scores['oof_ir1']:>7.3f}{scores['oof_ir0']:>7.3f}  "
                      + ', '.join('/'.join(f'{t:.2f}' for t in fit[f]) for f in ('f25', 'f26'))
                      + f"   сумма best {p_best.sum():g} из {y.sum():g}, ошибка {np.abs(p_best - y).sum():g}")
                if args.errors and len(names) == 1 and not with_prod:
                    for n in np.flatnonzero(p_best != y):
                        conf, _, found = objects[names[0]][train[n]]
                        text = ', '.join(f'{k} {c:.2f}' for c, k in zip(conf, found) if c >= best[0])
                        print(f"    {rows.row_id[n]} {rows.date_utc[n][5:]} slot {rows.slot[n]} ir {rows.is_ir[n]} "
                              f"люди {int(truth.people.iloc[n])}: истина {y[n]:g}, найдено {p_best[n]:g} [{text}]")
                if args.rowfeats is not None:   # тест: порог со всего train; train-строки: порог другого дня
                    column = f"ov_counts__{counter}_" + title.replace('ov_', '').replace('|', '_or_').replace(
                        '&', '_and_').replace('+', '_')
                    values = np.array([count(j, best) for j in range(len(scope))], float)
                    values[train] = p_oof
                    features[column] = values
                    report[column] = {'thr': dict(zip(names, map(float, best))),
                                      'thr_oof': {f: dict(zip(names, map(float, fit[f]))) for f in fit},
                                      **{k: round(float(v), 4) for k, v in scores.items()}}

    if args.rowfeats is not None:
        from run_ov import code_sha, utc_now
        name = f'ov_counts__v{args.rowfeats}'
        out_path = paths.art('rowfeats', f'{name}.parquet')
        if out_path.is_file():
            print(f'{name}: уже есть ({out_path}), не перезаписан. Артефакты неизменяемы: укажите новый номер.')
            return
        frame = pd.DataFrame({'row_id': scope.row_id, 'dt': 0, **features, 'ov_counts__ok': 1.0})
        frame.to_parquet(out_path, index=False)
        card = {
            'name': name, 'kind': 'rowfeats', 'owner': 'C', 'created_utc': utc_now(), 'git_sha': code_sha(),
            'command': 'python tracks/ov/eval_dets.py ' + ' '.join(sys.argv[1:]),
            'models': sorted({m for c in cards.values() for m in c['models']}),
            'inputs': [f'dets/{n}.parquet' for n in args.dets] + ['detections.csv', 'train_labels.csv'],
            'manual_labels_used': False,
            'uses_train_counters': True,
            'train_values': 'oof_by_date',
            'classes': {c: getattr(args, c).split(',') for c in ('items', 'dishes')},
            'columns': report,
            'notes': 'счётчики items и dishes по целевому кадру (dt = 0) для всех 4964 строк, по колонке на вариант; '
                     'привязка временная (tracks/ov/eval_dets.py), контекст не используется; пороги подобраны по '
                     'счётчикам train: для тестовых строк на всём train, для train-строк на другом дне (out-of-fold)',
        }
        out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        check_rowfeats(out_path)
        print(f'{name}: {len(frame)} строк, колонок {len(features)} -> {out_path}')


if __name__ == '__main__':
    main()
