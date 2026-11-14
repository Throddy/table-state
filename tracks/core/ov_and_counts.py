"""Счётчик чашек трека C при согласии двух моделей: признаки строк `ovand__v1` (формат 8.4).

Запуск: python tracks/core/ov_and_counts.py
Вход: ART/dets/ov_gdino__v1.parquet и ov_owl__v1.parquet (класс cup), привязка ts.assign (зона top).
Чашка засчитывается, если её нашли обе модели (совпадение по правилу 9.2.6) с confidence не ниже порогов.
Пара порогов подбирается по train: минимум суммы |ошибки| для max(чашки прод-детектора, согласованные чашки),
при равенстве более строгая пара. Тестовые строки: пороги со всего train; train-строки: пороги другого дня.
Выход: ART/rowfeats/ovand__v1.parquet (`ovand__cups`, `ovand__ok`) и карточка.

Ограничение: значения train-строк получены с порогами другого дня (out-of-fold, раздел 8.5). Колонку можно
использовать только фиксированным правилом (как `compose: baseline_max`). Обучать на ней модель внутри фолда
нельзя: значения строк дня подгонки зависят от счётчиков дня проверки.
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from _bootstrap import git_sha  # noqa: E402

from ts import io, orgbase, paths  # noqa: E402
from ts.assign import DEFAULTS, _dedup  # noqa: E402
from ts.check import check_rowfeats  # noqa: E402
from ts.cv import FOLDS  # noqa: E402
from ts.features import assigned_for  # noqa: E402

NAME = 'ovand__v1'
SOURCES = ['ov_gdino__v1', 'ov_owl__v1']
GRID = [round(0.05 * i, 2) for i in range(2, 13)]   # 0.10 … 0.60
BOX = ['x1', 'y1', 'x2', 'y2']


def matched_pairs() -> pd.DataFrame:
    """Пары «чашка gdino, чашка owl» одной строки, совпадающие по правилу дедупликации 9.2.6."""
    g, o = (assigned_for(s, {}) for s in SOURCES)
    g, o = (a[(a['dt'] == 0) & (a['class_name'] == 'cup')] for a in (g, o))
    by_row = {row_id: part for row_id, part in o.groupby('row_id')}
    pairs = []
    for row_id, part in g.groupby('row_id'):
        other = by_row.get(row_id)
        if other is None:
            continue
        free = list(range(len(other)))
        for a in part.sort_values('confidence', ascending=False).itertuples():
            for j in sorted(free, key=lambda k: -other['confidence'].iloc[k]):
                boxes = np.array([[a.x1, a.y1, a.x2, a.y2], other[BOX].iloc[j].to_numpy(float)])
                if len(_dedup(boxes, np.array([1.0, 0.5]), DEFAULTS['small'])) == 1:   # считаются одним объектом
                    pairs.append((row_id, float(a.confidence), float(other['confidence'].iloc[j])))
                    free.remove(j)
                    break
    return pd.DataFrame(pairs, columns=['row_id', 'conf_gdino', 'conf_owl'])


def count(pairs: pd.DataFrame, thr: tuple[float, float], index: pd.Index) -> pd.Series:
    kept = pairs[(pairs['conf_gdino'] >= thr[0]) & (pairs['conf_owl'] >= thr[1])]
    return kept.groupby('row_id').size().reindex(index, fill_value=0).astype(float)


def tune(pairs: pd.DataFrame, labels: pd.DataFrame, prod_cups: pd.Series) -> tuple[tuple[float, float], float]:
    truth = labels.set_index('row_id')['cups']
    base = prod_cups.reindex(truth.index)
    best = None
    for thr in itertools.product(GRID, GRID):
        error = float((np.maximum(base, count(pairs, thr, truth.index)) - truth).abs().sum())
        key = (error, -thr[0] - thr[1])
        if best is None or key < best[0]:
            best = (key, thr)
    return best[1], best[0][0]


def main() -> None:
    pairs = matched_pairs()
    labels = io.load_labels()
    tables = io.load_tables()
    prod_cups = orgbase.features()['det_cups']
    out = pd.Series(0.0, index=pd.Index(tables['row_id'], name='row_id'))
    report = {}
    thr_all, err_all = tune(pairs, labels, prod_cups)
    test_ids = tables.loc[tables['split'] == 'test', 'row_id']
    out.loc[test_ids] = count(pairs, thr_all, pd.Index(test_ids)).to_numpy()
    report['all_train'] = {'thr_gdino_owl': thr_all, 'abs_error': err_all,
                           'abs_error_prod_only': float((prod_cups.reindex(labels.row_id).to_numpy() - labels.cups).abs().sum())}
    for fold, (fit_date, val_date) in FOLDS.items():
        thr, err = tune(pairs, labels[labels.date_utc == fit_date], prod_cups)
        ids = labels.loc[labels.date_utc == val_date, 'row_id']
        out.loc[ids] = count(pairs, thr, pd.Index(ids)).to_numpy()
        report[fold] = {'fit_date': fit_date, 'thr_gdino_owl': thr, 'abs_error_fit': err}
    frame = pd.DataFrame({'row_id': out.index, 'dt': 0, 'ovand__cups': out.to_numpy(), 'ovand__ok': 1})
    # Файл и карточка сначала пишутся во временный каталог и проверяются: непроверенный артефакт
    # не должен появиться в ART/rowfeats, где его подхватит сборка.
    final = paths.art('rowfeats', f'{NAME}.parquet')
    staging = Path(tempfile.mkdtemp(prefix='ovand_', dir=paths.ART))
    path = staging / final.name
    frame.to_parquet(path, index=False)
    sha, _ = git_sha()
    cards = [json.loads((paths.ART / 'dets' / f'{s}.json').read_text(encoding='utf-8')) for s in SOURCES]
    card = {'name': NAME, 'kind': 'rowfeats', 'owner': 'A',
            'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'git_sha': sha,
            'command': 'python tracks/core/ov_and_counts.py',
            'models': sorted({m for c in cards for m in c.get('models', [])}),
            'inputs': [f'dets/{s}.parquet' for s in SOURCES] + ['detections.csv', 'train_labels.csv'],
            'manual_labels_used': False, 'uses_train_counters': True, 'train_values': 'oof_by_date',
            'pairs': int(len(pairs)), 'thresholds': report,
            'notes': 'чашки, найденные обеими моделями трека C (dt = 0); пороги по счётчикам train'}
    path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        check_rowfeats(path)
        os.replace(path, final)
        os.replace(path.with_suffix('.json'), final.with_suffix('.json'))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    path = final
    print(json.dumps(report, ensure_ascii=False))
    print(f'{NAME}: пар {len(pairs)}, чашек в тесте {int(out.loc[test_ids].sum())} в {int((out.loc[test_ids] > 0).sum())} строках -> {path}')


if __name__ == '__main__':
    main()
