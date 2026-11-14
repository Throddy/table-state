"""Файлы поверх основы с колонками из VLM-счётчика (задача C3, шаг 2). На Kaggle ничего не отправляет.

Запуск: python tracks/det/vlm/apply_vlm.py [--rowfeats vlm_qwen__v1] [--base COMBO] [--no-gate]
Собирает три файла ART/preds/<имя>.csv (4964 строки) и регистрирует их через tools/submit_gate.py:
  VLM_people        people = ответ VLM (замена);
  VLM_items         items = max(основа, ответ VLM);
  VLM_people_items  обе колонки сразу.
Строка без ответа VLM (__ok = 0) остаётся как в основе. hands, cups, dishes, occ_prob не меняются.
VLM работает без подгонки по счётчикам, поэтому его train-строки подставляются как есть.
Для каждого файла печатается число изменённых строк теста и число строк, где после замены people
остаются hands > 0 при people = 0 (руки при этом не трогаются).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from ts import PRED_COLUMNS, io, paths  # noqa: E402
from ts.check import check_pred  # noqa: E402
from ts.cv import evaluate  # noqa: E402

SOURCE = 'vlm_qwen'
VARIANTS = {'VLM_people': ['people'], 'VLM_items': ['items'], 'VLM_people_items': ['people', 'items']}
NOTES = {'VLM_people': 'COMBO с people из VLM-счётчика (замена): убирает ли VLM лишних людей и находит ли пропущенных?',
         'VLM_items': 'COMBO с items = max(COMBO, VLM): добавляет ли VLM вещи, которых не видят модели трека C?',
         'VLM_people_items': 'COMBO с people из VLM (замена) и items = max(COMBO, VLM) одновременно'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rowfeats', default=f'{SOURCE}__v1')
    parser.add_argument('--base', default='COMBO')
    parser.add_argument('--no-gate', action='store_true', help='только расчёт, без регистрации файлов')
    args = parser.parse_args()
    base_path = paths.ART / 'preds' / f'{args.base}.csv'
    base = io.read_csv_exact(base_path).set_index('row_id')
    vlm = pd.read_parquet(paths.ART / 'rowfeats' / f'{args.rowfeats}.parquet')
    vlm = vlm[vlm['dt'] == 0].set_index('row_id')
    ids = base.index.intersection(vlm.index[vlm[f'{SOURCE}__ok'] == 1])
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')['row_id']
    labels = io.load_labels()
    print(f'{args.rowfeats}: строк {len(vlm)}, с ответом {len(ids)}; основа {args.base}: {len(base)} строк')
    reference = evaluate(base.reset_index()[PRED_COLUMNS], bootstrap=0)
    rows = [{'файл': args.base, 'oof_all': round(reference['oof_all']['score'], 4), 'f25': round(reference['f25']['score'], 4),
             'f26': round(reference['f26']['score'], 4), 'people': round(reference['oof_all']['quality']['people'], 3),
             'items': round(reference['oof_all']['quality']['items'], 3)}]
    base_card = json.loads(base_path.with_suffix('.json').read_text(encoding='utf-8'))
    for name, columns in VARIANTS.items():
        new = base.copy()
        if 'people' in columns:
            new.loc[ids, 'people'] = vlm.loc[ids, f'{SOURCE}__people'].to_numpy(float)
        if 'items' in columns:
            new.loc[ids, 'items'] = np.maximum(base.loc[ids, 'items'].to_numpy(float), vlm.loc[ids, f'{SOURCE}__items'].to_numpy(float))
        path = paths.art('preds', f'{name}.csv')
        new.reset_index()[PRED_COLUMNS].to_csv(path, index=False)
        card = dict(base_card)
        card.update({'name': name, 'owner': common.OWNER, 'created_utc': common.utc_now(), 'git_sha': common.git_sha(),
                     'command': common.command_line(), 'sources': list(base_card.get('sources', [])) + [args.rowfeats],
                     'vlm': {'rowfeats': args.rowfeats, 'columns': {c: ('replace' if c == 'people' else 'max') for c in columns},
                             'rows_with_answer': int(len(ids))},
                     'notes': NOTES[name] + '; остальные колонки как в основе'})
        path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        check_pred(path)
        report = evaluate(new.reset_index()[PRED_COLUMNS], bootstrap=0)
        test_new, test_base = new.loc[sample], base.loc[sample]
        row = {'файл': name, 'oof_all': round(report['oof_all']['score'], 4), 'f25': round(report['f25']['score'], 4),
               'f26': round(report['f26']['score'], 4), 'people': round(report['oof_all']['quality']['people'], 3),
               'items': round(report['oof_all']['quality']['items'], 3)}
        for c in columns:
            d = test_new[c] - test_base[c]
            row[f'тест {c}: изменено'] = f'{int((d != 0).sum())} (вверх {int((d > 0).sum())}, вниз {int((d < 0).sum())}), сумма {test_base[c].sum():g} -> {test_new[c].sum():g}'
        row['тест: hands > 0 при people = 0'] = f'{int(((test_new.hands > 0) & (test_new.people == 0)).sum())} (в основе {int(((test_base.hands > 0) & (test_base.people == 0)).sum())})'
        row['train изменено'] = int((new.loc[labels.row_id, columns] != base.loc[labels.row_id, columns]).any(axis=1).sum())
        rows.append(row)
        if not args.no_gate:
            done = subprocess.run([sys.executable, str(common.ROOT / 'tools' / 'submit_gate.py'), '--pred', str(path), '--note', NOTES[name],
                                   '--bootstrap', '0'], cwd=common.ROOT, capture_output=True, text=True)
            gate = [line for line in done.stdout.splitlines() if 'GATE' in line]
            print(gate[0] if gate else f'{name}: gate не прошёл: {done.stdout[-300:]} {done.stderr[-300:]}')
    table = pd.DataFrame(rows).fillna('')
    pd.set_option('display.width', 260, 'display.max_colwidth', 70)
    print('\n' + table.to_string(index=False))


if __name__ == '__main__':
    main()
