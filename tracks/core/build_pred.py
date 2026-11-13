"""Предсказание R1 (задачи A3 и A4): только прод-детекции, привязка v2, контекст, мягкий occ_prob.

Запуск: python tracks/core/r1.py [--name R1]
Пишет ART/preds/<name>.csv (4964 строки) и карточку.
- train-строки: out-of-fold по дням (порог площади бокса человека и логрег подгоняются на другом дне);
- тест: подгонка на всём train.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from _bootstrap import ROOT, git_sha  # noqa: E402

from ts import PRED_COLUMNS, io, paths, stack  # noqa: E402
from ts.check import check_pred  # noqa: E402
from ts.cv import oof_predictions  # noqa: E402
from ts.features import assigned_for, build_features  # noqa: E402
from ts.rows import load_rows_ext  # noqa: E402


def make_fit_predict(cfg: dict, log: list):
    """Функция `fit_predict(train_rows, predict_rows)` для `ts.cv` на прод-детекциях."""
    rows = load_rows_ext()
    assigned = assigned_for('prod', cfg)

    def fit_predict(train_rows: pd.DataFrame, predict_rows: pd.DataFrame) -> pd.DataFrame:
        local = json.loads(json.dumps(cfg))
        if cfg.get('stack', {}).get('tune_people_area', True):
            area, errors = stack.tune_people_area(assigned, train_rows, local, rows)
            local.setdefault('temporal', {})['people_max_area'] = area
        features = build_features(['prod'], local)
        model = stack.fit(features, train_rows, local)
        log.append({'fit_dates': sorted(train_rows.date_utc.unique()), 'n_fit_rows': len(train_rows),
                    'people_max_area': local.get('temporal', {}).get('people_max_area'),
                    'occ_coef': model.get('coef'), 'occ_intercept': model.get('intercept')})
        return stack.predict(model, features.loc[predict_rows.row_id], local)

    fit_predict.n_params = 1 + 8   # порог площади + коэффициенты логрега (7 признаков и свободный член)
    return fit_predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='R1')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))

    log: list = []
    fit_predict = make_fit_predict(cfg, log)
    oof = oof_predictions(fit_predict)
    labels = io.load_labels()
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')
    test = fit_predict(labels, sample[['row_id']])
    pred = pd.concat([test, oof], ignore_index=True)[PRED_COLUMNS]

    out = paths.art('preds', f'{args.name}.csv')
    pred.to_csv(out, index=False)
    sha, _ = git_sha()
    card = {
        'name': args.name, 'kind': 'preds', 'owner': 'A',
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'command': f'python tracks/core/r1.py --name {args.name} --config {args.config.relative_to(ROOT)}',
        'models': [], 'inputs': ['rows_ext.parquet', 'detections.csv', 'train_labels.csv'],
        'sources': ['prod'],
        'train_pred': 'oof_by_date',
        'manual_labels_used': False,
        'uses_train_counters': True,
        'n_params': fit_predict.n_params,
        'fits': log,
        'notes': 'привязка ts.assign v2, контекст раздела 10, логрег занятости; items = 0',
    }
    out.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(out)
    for entry in log:
        print(entry)
    print(f'{args.name}: {len(pred)} строк -> {out}')


if __name__ == '__main__':
    main()
