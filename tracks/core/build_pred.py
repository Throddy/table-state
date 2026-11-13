"""Сборка предсказания по конфигу: источники из `sources`, привязка, контекст, стекер, постобработка.

Запуск: python tracks/core/build_pred.py --name R1b [--config configs/default.yaml]
Пишет ART/preds/<name>.csv (4964 строки) и карточку.
- train-строки: out-of-fold по дням (все подгоняемые параметры подбираются на другом дне);
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


def build(name: str, cfg: dict, notes: str = '', command: str = '') -> Path:
    """Собрать `ART/preds/<name>.csv` (4964 строки: тест + out-of-fold train) и карточку по конфигу `cfg`."""
    log: list = []
    fit_predict = stack.make_fit_predict(cfg, log)
    oof = oof_predictions(fit_predict)
    labels = io.load_labels()
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')
    test = fit_predict(labels, sample[['row_id']])
    pred = pd.concat([test, oof], ignore_index=True)[PRED_COLUMNS]

    out = paths.art('preds', f'{name}.csv')
    pred.to_csv(out, index=False)
    sha, _ = git_sha()
    sources = list(cfg['sources'].get('dets', [])) + list(cfg['sources'].get('rowfeats', []))
    card = {
        'name': name, 'kind': 'preds', 'owner': 'A',
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'command': command,
        'models': [], 'inputs': ['rows_ext.parquet', 'train_labels.csv'] + sources,
        'sources': sources,
        'train_pred': 'oof_by_date',
        'manual_labels_used': False,
        'uses_train_counters': True,
        'n_params': fit_predict.n_params,
        'post': cfg.get('post', {}),
        'stack_counts': cfg.get('stack', {}).get('counts', {}),
        'fits': log,
        'notes': notes,
    }
    out.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(out)
    for entry in log:
        print({k: entry[k] for k in ['fit_dates', 'people_max_area', 'min_conf']})
    print(f'{name}: {len(pred)} строк -> {out}')
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    parser.add_argument('--notes', default='')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    config_path = args.config.resolve()
    shown = config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path.name
    build(args.name, cfg, args.notes, f'python tracks/core/build_pred.py --name {args.name} --config {shown}')


if __name__ == '__main__':
    main()
