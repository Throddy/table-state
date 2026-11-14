"""Среднее одной колонки двух файлов предсказаний (ансамбль двух версий детектора).

Запуск: python tracks/det/avg_cols.py --base COMBO --other AB_ft2_hands_pp --col hands --name AB_ft2_hands_avg
Берёт ART/preds/<base>.csv, заменяет колонку на 0.5·base + 0.5·other, остальные колонки остаются из base.
Значения не округляются: дробный счётчик допустим как результат усреднения ансамбля (раздел 1 контракта).
Пишет ART/preds/<name>.csv с карточкой. Регистрация: tools/submit_gate.py.
"""
from __future__ import annotations

import argparse
import json

import common

from ts import COUNTS, PRED_COLUMNS, io, paths
from ts.check import check_pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--other', required=True)
    parser.add_argument('--col', required=True, choices=COUNTS)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    base_path = paths.ART / 'preds' / f'{args.base}.csv'
    base = io.read_csv_exact(base_path).set_index('row_id')
    other = io.read_csv_exact(paths.ART / 'preds' / f'{args.other}.csv').set_index('row_id').loc[base.index]
    out = base.copy()
    out[args.col] = 0.5 * base[args.col] + 0.5 * other[args.col]
    changed = int((out[args.col] != base[args.col]).sum())
    path = paths.art('preds', f'{args.name}.csv')
    out.reset_index()[PRED_COLUMNS].to_csv(path, index=False)
    card = json.loads(base_path.with_suffix('.json').read_text(encoding='utf-8'))
    card.update({'name': args.name, 'owner': common.OWNER, 'created_utc': common.utc_now(), 'git_sha': common.git_sha(),
                 'command': common.command_line(), 'average': {'col': args.col, 'files': [args.base, args.other]},
                 'rows_changed': changed,
                 'notes': f'{args.base} с колонкой {args.col} = среднее {args.base} и {args.other}, без округления'})
    path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(path)
    print(f'{args.name}: {args.col} изменена в {changed} строках из {len(out)} -> {path}')


if __name__ == '__main__':
    main()
