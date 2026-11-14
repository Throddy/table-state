"""Шаг 1 задачи C3: оценка VLM-счётчика на 231 train-строке и решение Go.

Запуск: python tracks/det/vlm/eval_vlm.py [--rowfeats vlm_qwen__v1__train] [--base COMBO]
1. VLM в одиночку против нулей: сумма |ошибки| по классу на каждом дне должна быть меньше суммы объектов.
2. Связка по 11.1 поверх основы: счётчик = медиана из (значение основы, VLM, контекстная медиана детекций);
   строка без ответа VLM (__ok = 0) остаётся как в основе. Колонка подставляется в основу по одной.
Go: хотя бы по двум классам из dishes, items, people VLM лучше нулей на обоих днях и связка не хуже основы.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from ts import COUNTS, PRED_COLUMNS, io, paths  # noqa: E402
from ts.cv import evaluate  # noqa: E402
from ts.features import build_features  # noqa: E402

SOURCE = 'vlm_qwen'
GO_CLASSES = ['dishes', 'items', 'people']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rowfeats', default=f'{SOURCE}__v1__train')
    parser.add_argument('--base', default='COMBO')
    args = parser.parse_args()
    labels = io.load_labels().set_index('row_id')
    vlm = pd.read_parquet(paths.ART / 'rowfeats' / f'{args.rowfeats}.parquet').set_index('row_id').loc[labels.index]
    ok = vlm[f'{SOURCE}__ok'] == 1
    base = io.read_csv_exact(paths.ART / 'preds' / f'{args.base}.csv').set_index('row_id')
    cfg = yaml.safe_load((common.ROOT / 'configs' / 'submissions' / f'{args.base}.yaml').read_text(encoding='utf-8'))
    feats = build_features(cfg['sources']['dets'] + cfg['sources'].get('rowfeats', []), cfg).loc[labels.index]
    days = sorted(labels['date_utc'].unique())
    print(f'{args.rowfeats}: строк {len(vlm)}, без ответа {int((~ok).sum())} ({(~ok).mean():.1%})')

    alone, beats = [], {}
    for c in COUNTS:
        pred = vlm[f'{SOURCE}__{c}'].where(ok, 0.0)
        row = {'класс': c}
        wins = []
        for day in days:
            m = labels['date_utc'] == day
            error, objects = float((pred[m] - labels.loc[m, c]).abs().sum()), float(labels.loc[m, c].sum())
            base_error = float((base.loc[labels.index[m], c] - labels.loc[m, c]).abs().sum())
            row[day[5:]] = f'{error:g}/{objects:g}'
            row[f'основа {day[5:]}'] = f'{base_error:g}/{objects:g}'
            wins.append(error < objects)
        beats[c] = all(wins)
        row['лучше нулей на обоих днях'] = 'да' if beats[c] else 'нет'
        row['сумма VLM / истина'] = f'{pred.sum():g}/{labels[c].sum():g}'
        alone.append(row)
    print('\nVLM в одиночку: ошибка / объектов по дням (основа рядом для сравнения)')
    print(pd.DataFrame(alone).to_string(index=False))

    reference = evaluate(base.reset_index()[PRED_COLUMNS], bootstrap=0)
    rows, not_worse = [], {}
    for c in COUNTS:
        context = feats[f'all__{c}__med'].to_numpy(float)
        stacked = np.median(np.stack([base.loc[labels.index, c].to_numpy(float), vlm[f'{SOURCE}__{c}'].to_numpy(float), context]), axis=0)
        new = base.copy()
        new.loc[labels.index, c] = np.where(ok.to_numpy(), stacked, base.loc[labels.index, c].to_numpy(float))
        report = evaluate(new.reset_index()[PRED_COLUMNS], bootstrap=0)
        delta = report['oof_all']['score'] - reference['oof_all']['score']
        not_worse[c] = (report['oof_all']['score'] >= reference['oof_all']['score'] - 1e-9
                        and report['f25']['score'] >= reference['f25']['score'] - 1e-9
                        and report['f26']['score'] >= reference['f26']['score'] - 1e-9)
        rows.append({'класс': c, 'oof_all': round(report['oof_all']['score'], 4), 'f25': round(report['f25']['score'], 4),
                     'f26': round(report['f26']['score'], 4), 'колонка': round(report['oof_all']['quality'][c], 3),
                     'основа': round(reference['oof_all']['quality'][c], 3), 'к основе': round(delta, 4),
                     'строк train изменено': int((new.loc[labels.index, c] != base.loc[labels.index, c]).sum()),
                     'не хуже основы': 'да' if not_worse[c] else 'нет'})
    print(f"\nСвязка по 11.1 поверх {args.base} (основа: oof_all {reference['oof_all']['score']:.4f}, "
          f"f25 {reference['f25']['score']:.4f}, f26 {reference['f26']['score']:.4f})")
    print(pd.DataFrame(rows).to_string(index=False))
    passed = [c for c in GO_CLASSES if beats[c] and not_worse[c]]
    print(f"\nКлассы из {GO_CLASSES}, где VLM лучше нулей на обоих днях и связка не хуже основы: {passed}")
    print('РЕШЕНИЕ:', 'GO' if len(passed) >= 2 else 'СТОП')


if __name__ == '__main__':
    main()
