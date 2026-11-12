"""Опорное предсказание `base_calibrated` (задача A2): бейзлайн организаторов, режим calibrated, без изменений.

Запуск: python tracks/core/base_calibrated.py
Пишет ART/preds/base_calibrated.csv (4964 строки) и карточку.
- тест (4733 строки): `baseline.run(mode='calibrated')`, подгонка на всём train;
- train (231 строка): out-of-fold по дням (раздел 12.1) тем же `calibrated_prediction`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from _bootstrap import git_sha, organizer_baseline  # noqa: E402

from ts import PRED_COLUMNS, io, paths  # noqa: E402
from ts.check import check_pred  # noqa: E402
from ts.cv import FOLDS  # noqa: E402

NAME = 'base_calibrated'


def main() -> None:
    baseline = organizer_baseline()
    features = baseline.make_features(paths.DATA)
    labels = io.load_labels()
    with tempfile.TemporaryDirectory() as tmp:
        test_pred, info = baseline.run(paths.DATA, out=Path(tmp) / 'submission.csv', mode='calibrated',
                                       features=features)
    parts = [test_pred]
    for fit_date, val_date in FOLDS.values():
        ids = labels.loc[labels.date_utc == val_date, 'row_id'].tolist()
        oof, _ = baseline.calibrated_prediction(features, labels[labels.date_utc == fit_date], ids)
        parts.append(oof.reset_index()[PRED_COLUMNS])
    pred = pd.concat(parts, ignore_index=True)[PRED_COLUMNS]

    out = paths.art('preds', f'{NAME}.csv')
    pred.to_csv(out, index=False)
    sha, _ = git_sha()
    card = {
        'name': NAME, 'kind': 'preds', 'owner': 'A',
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'command': 'python tracks/core/base_calibrated.py',
        'models': [],
        'inputs': ['detections.csv', 'tables.csv', 'targets.csv', 'frames.csv', 'train_labels.csv'],
        'sources': ['prod'],
        'train_pred': 'oof_by_date',
        'manual_labels_used': False,
        'uses_train_counters': True,
        'scales_full_train': info['scales'],
        'notes': 'data/baseline.py организаторов, режим calibrated, без изменений; сид 419 внутри бейзлайна',
    }
    out.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(out)
    print(f'{NAME}: {len(pred)} строк -> {out}')


if __name__ == '__main__':
    main()
