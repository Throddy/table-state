"""Сверка фактов раздела 4 контракта и опорных скоров таблицы 4.5.

Запуск: python tools/sanity.py
При любом расхождении печатает список несовпадений и завершается с кодом 1.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _bootstrap import organizer_baseline

from ts import COUNTS, io, paths
from ts.metric import score_frame
from ts.rows import ROWS_EXT_NAME

FAILS: list[str] = []


def expect(name: str, got, want) -> None:
    """Сравнить факт с контрактом и запомнить расхождение."""
    if got != want:
        FAILS.append(f'{name}: получено {got}, в контракте {want}')


def counts(series: pd.Series) -> dict:
    return {k: int(v) for k, v in series.value_counts().sort_index().items()}


def check_facts() -> None:
    """Факты разделов 4.1–4.6."""
    frames, targets, tables = io.load_frames(), io.load_targets(), io.load_tables()
    dets, labels, aux = io.load_prod_dets(), io.load_labels(), io.load_aux_boxes()
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')
    aux_frames = pd.read_csv(paths.DATA / 'aux_frames.csv')

    # 4.1 файлы
    manifest = (paths.DATA / 'MANIFEST.sha256').read_text().splitlines()
    names = [line.split()[-1] for line in manifest if line.strip()]
    expect('MANIFEST: файлов', len(names), 1780)
    expect('MANIFEST: im_*.jpg', sum(n.startswith('im_') and n.endswith('.jpg') for n in names), 1718)
    expect('MANIFEST: aux_*.jpg', sum(n.startswith('aux_') and n.endswith('.jpg') for n in names), 45)
    for title, df, n in [('frames.csv', frames, 1718), ('targets.csv', targets, 839), ('tables.csv', tables, 4964),
                         ('detections.csv', dets, 15946), ('train_labels.csv', labels, 231),
                         ('sample_submission.csv', sample, 4733), ('aux_frames.csv', aux_frames, 45),
                         ('aux_boxes.csv', aux, 610)]:
        expect(f'{title}: строк', len(df), n)
    expect('frames: размер кадра', (set(frames.width), set(frames.height)), ({1280}, {720}))

    # 4.2 кадры и сплиты
    expect('кадров по дням', counts(frames.date_utc),
           {'2026-10-25': 63, '2026-10-26': 54, '2026-10-27': 905, '2026-10-28': 696})
    t = targets.merge(frames[['frame_id', 'date_utc']], on='frame_id')
    expect('целевых по сплитам', counts(t.split), {'test': 800, 'train': 39})
    expect('целевых по дням', counts(t.date_utc),
           {'2026-10-25': 21, '2026-10-26': 18, '2026-10-27': 471, '2026-10-28': 329})
    expect('train только 25–26.10', set(t.loc[t.split == 'train', 'date_utc']), {'2026-10-25', '2026-10-26'})
    n_ctx = t.context.map(len)
    expect('контекст train', counts(n_ctx[t.split == 'train']), {2: 39})
    expect('контекст test', counts(n_ctx[t.split == 'test']), {1: 16, 2: 784})
    context_ids = {n for ctx in targets.context for n in ctx}
    expect('контекстных кадров, которые сами целевые', len(context_ids & set(targets.frame_id)), 505)
    expect('каждый кадр целевой или сосед', set(frames.frame_id), context_ids | set(targets.frame_id))

    # 4.3 столы
    # Соответствие один к одному; в части кадров tables.csv округлён, допуск 0.5 px.
    box = ['x1', 'y1', 'x2', 'y2']
    prod = {f: g[box].to_numpy() for f, g in dets[dets.class_name == 'table'].groupby('frame_id')}
    worst, one_to_one = 0.0, True
    for frame_id, g in tables.groupby('frame_id'):
        other = prod.get(frame_id, np.zeros((0, 4)))
        if len(other) != len(g):
            one_to_one = False
            continue
        dist = np.abs(g[box].to_numpy()[:, None, :] - other[None, :, :]).max(axis=-1)
        one_to_one &= len(set(dist.argmin(axis=1))) == len(g)
        worst = max(worst, float(dist.min(axis=1).max()))
    expect('рамки tables.csv соответствуют детекциям table один к одному', one_to_one, True)
    expect('макс. отличие рамки от детекции table не больше 0.5 px', worst <= 0.5 + 1e-9, True)
    expect('столов на кадре', counts(tables.groupby('frame_id').size()), {3: 2, 4: 14, 5: 38, 6: 783, 7: 2})
    match = (tables.table_id.str.replace('table_', '').astype(int) == tables.slot).mean()
    expect('table_id совпадает со слотом (~97%)', 0.96 <= match <= 0.98, True)
    expect('кадров с двумя столами в одном слоте',
           int((tables.groupby(['frame_id', 'slot']).size() > 1).groupby('frame_id').any().sum()), 2)
    centers = tables.groupby('slot').cx.median()
    for slot, cx in {1: 194, 2: 348, 3: 560, 4: 798, 5: 1068, 6: 1228}.items():
        expect(f'медиана cx слота {slot} (±3 px)', abs(centers.get(slot, -1e9) - cx) <= 3, True)

    # 4.4 train-счётчики
    expect('суммы счётчиков', {c: int(labels[c].sum()) for c in COUNTS},
           {'people': 29, 'hands': 29, 'cups': 21, 'dishes': 19, 'items': 28})
    for c, dist in {'people': {0: 208, 1: 20, 2: 2, 5: 1}, 'hands': {0: 214, 1: 6, 2: 10, 3: 1},
                    'cups': {0: 213, 1: 17, 4: 1}, 'dishes': {0: 220, 1: 8, 2: 2, 7: 1},
                    'items': {0: 214, 1: 8, 2: 7, 3: 2}}.items():
        expect(f'распределение {c}', counts(labels[c]), dist)
    by_day = labels.groupby('date_utc').agg(rows=('row_id', 'size'), occ=('occupied', 'sum'),
                                            people=('people', 'sum'), hands=('hands', 'sum'), cups=('cups', 'sum'))
    expect('25.10: строк/занято/people/hands/cups', by_day.loc['2026-10-25'].astype(int).tolist(), [126, 7, 9, 8, 7])
    expect('26.10: строк/занято/people/hands/cups', by_day.loc['2026-10-26'].astype(int).tolist(), [105, 16, 20, 21, 14])
    # 4.4 (контракт 1.4): суммы по table_id и по slot различаются, потому что на
    # кадре-выбросе 5 рамок и его table_1 физически стоит в слоте 2.
    by_table = labels.groupby('table_id')[COUNTS].sum().astype(int)
    for n, want in {1: [15, 14, 11, 10, 9], 2: [1, 2, 1, 1, 1], 3: [6, 8, 7, 7, 8],
                    4: [0, 0, 0, 0, 0], 5: [2, 2, 0, 0, 5], 6: [5, 3, 2, 1, 5]}.items():
        expect(f'суммы по table_{n}', by_table.loc[f'table_{n}'].tolist(), want)
    by_slot = labels.merge(tables[['row_id', 'slot']], on='row_id').groupby('slot')[COUNTS].sum().astype(int)
    for slot, want in {1: [10, 12, 7, 3, 8], 2: [6, 4, 5, 8, 2], 3: [6, 8, 7, 7, 8],
                       4: [0, 0, 0, 0, 0], 5: [2, 2, 0, 0, 5], 6: [5, 3, 2, 1, 5]}.items():
        expect(f'суммы по slot {slot} (ts.slots)', by_slot.loc[slot].tolist() if slot in by_slot.index else None, want)
    expect('строк по слотам', counts(tables.slot), {1: 814, 2: 839, 3: 839, 4: 839, 5: 829, 6: 804})
    outlier = labels[labels.row_id == 'q_1d3cfd4459d2c4fef90f']
    expect('строка-выброс', outlier[['frame_id'] + COUNTS].to_numpy().tolist(),
           [['im_cf13426b6f011b9cb35c', 5, 2, 4, 7, 1]])
    empty = labels[labels.people == 0]
    expect('строк с объектами при people = 0', int((empty[COUNTS].sum(axis=1) > 0).sum()), 2)
    expect('hands > 0 при people = 0', int((empty.hands > 0).sum()), 0)

    # 4.5 прод-детектор, 4.6 aux
    expect('классы прод-детектора', counts(dets.class_name),
           {'cup': 1209, 'hand': 1724, 'napkin': 3, 'person': 2451, 'table': 10135, 'tray': 424})
    expect('минимальный confidence', round(float(dets.confidence.min()), 2), 0.40)
    raw_aux = pd.read_csv(paths.DATA / 'aux_boxes.csv')
    expect('классы aux', counts(raw_aux.class_name),
           {'cup': 38, 'empty_plate': 15, 'full_plate': 12, 'headphones': 4, 'nude_hands': 72, 'person': 101,
            'personal_items': 53, 'phone': 36, 'table': 262, 'tray': 17})

    # 8.3 индекс строк, если уже построен
    rows_path = paths.ART / ROWS_EXT_NAME
    if rows_path.is_file():
        rows = pd.read_parquet(rows_path)
        expect('rows_ext: строк с dt = 0', int((rows.dt == 0).sum()), 4964)
    else:
        print(f'(нет {rows_path}, проверка rows_ext пропущена: python -m ts.rows)')


def reference_scores() -> list[tuple[str, str, dict, float, float]]:
    """Таблица 4.5: бейзлайн организаторов без изменений, оценка через ts.metric."""
    baseline = organizer_baseline()
    features = baseline.make_features(paths.DATA)
    labels = io.load_labels()
    val = labels[labels.date_utc == '2026-10-26']
    fit = labels[labels.date_utc == '2026-10-25']
    detector = baseline.detector_prediction(features.loc[labels.row_id]).reset_index()
    calibrated, _ = baseline.calibrated_prediction(features, fit, val.row_id.tolist())
    zeros = pd.DataFrame({'row_id': labels.row_id, **{c: 0.0 for c in COUNTS}, 'occ_prob': 0.0})
    return [
        ('detector', 'весь train (231)', score_frame(labels, detector), 0.489, 0.746),
        ('detector', 'валидация 26.10 (105)', score_frame(val, detector), 0.490, 0.785),
        ('calibrated', 'валидация 26.10 (105)', score_frame(val, calibrated.reset_index()), 0.551, 0.989),
        ('нули', 'весь train (231)', score_frame(labels, zeros), 0.020, 0.100),
    ]


def main() -> int:
    check_facts()
    print(f'Факты раздела 4: {"OK" if not FAILS else "есть расхождения"}')
    results = reference_scores()
    print('\nОпорные скоры (раздел 4.5)')
    print(f"{'вариант':<12}{'где':<24}{'score':>7}{'AP':>7}   " + ' '.join(f'{c:>7}' for c in COUNTS))
    for name, where, r, want_score, want_ap in results:
        print(f"{name:<12}{where:<24}{r['score']:>7.3f}{r['occupancy_ap']:>7.3f}   "
              + ' '.join(f"{r['quality'][c]:>7.3f}" for c in COUNTS))
        expect(f'{name} / {where}: score', f"{r['score']:.3f}", f'{want_score:.3f}')
        expect(f'{name} / {where}: AP', f"{r['occupancy_ap']:.3f}", f'{want_ap:.3f}')
    detector_train = results[0][2]
    for c, want in {'people': 0.655, 'hands': 0.586, 'cups': 0.619, 'dishes': 0.263, 'items': 0.000}.items():
        expect(f'detector / весь train: {c}', f"{detector_train['quality'][c]:.3f}", f'{want:.3f}')
    if FAILS:
        print(f'\nSANITY FAILED, расхождений: {len(FAILS)}')
        for line in FAILS:
            print('  - ' + line)
        print('Сначала ищите ошибку в своём коде и данных (раздел 16.8 контракта).')
        return 1
    print('\nSANITY OK: факты раздела 4 и опорные скоры совпадают с контрактом.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
