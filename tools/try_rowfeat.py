"""Проверка счётчика из признаков строк: колонка rowfeats -> колонка предсказания -> замена в базовом файле.

Запуск:
  python tools/try_rowfeat.py --rowfeats ov_counts__v1 --column ov_counts__items_gdino_and_owl \
      --col items --base NB1 --name OV_items_rf [--mode replace|median|max] [--need-people]

Колонка rowfeats подставляется в счётчик `--col` (stack.vote), остальные колонки остаются как в основе.
Затем tools/swap_cols.py и submit_gate. На Kaggle не отправляет.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT / 'tracks' / 'core'))
from build_pred import build  # noqa: E402
from swap_cols import swap  # noqa: E402
from try_source import report_items, report_test_changes  # noqa: E402

from ts import paths  # noqa: E402
from ts.check import check_rowfeats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rowfeats', required=True, help='имя файла ART/rowfeats/<source>__v<N> без расширения')
    parser.add_argument('--column', required=True, help='колонка признаков, например ov_counts__items_gdino_and_owl')
    parser.add_argument('--col', required=True, help='счётчик предсказания: items, dishes, ...')
    parser.add_argument('--mode', default='replace', choices=['replace', 'median', 'max'])
    parser.add_argument('--need-people', action='store_true', help='items только при people >= 1 (stack.items_need_people)')
    parser.add_argument('--base', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--note')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    args = parser.parse_args()

    check_rowfeats(paths.ART / 'rowfeats' / f'{args.rowfeats}.parquet')
    print(f'ts.check: {args.rowfeats} OK')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cfg['sources']['dets'] = ['prod']
    cfg['sources'].pop('counters', None)
    cfg['sources']['rowfeats'] = [args.rowfeats]
    cfg['stack']['counts'][args.col] = cfg['stack']['counts'].get(args.col) or 'cur'
    cfg['stack']['vote'] = {args.col: {'column': args.column, 'mode': args.mode}}
    cfg['stack']['items_need_people'] = bool(args.need_people)
    cfg['compose'] = {**(cfg.get('compose') or {}), args.col: 'stack', 'hands': 'baseline'}
    cfg['post'].update({'hands_need_people': False, 'hands_cap': False})
    if args.col == 'dishes' and args.mode == 'replace':
        cfg['post']['tray_mult'] = 1.0      # колонка rowfeats уже содержит свой счёт посуды; при max и median сравнивается с 2 × подносы
    full = f'{args.name}__src'
    build(full, cfg, notes=f'сборка для проверки колонки {args.column} из {args.rowfeats} как {args.col} ({args.mode})',
          command='python tools/try_rowfeat.py ' + ' '.join(sys.argv[1:]))
    report_test_changes(args.base, full, [args.col])
    if args.col == 'items':
        report_items(args.base, full)
    print()
    swap(args.base, full, [args.col], args.name, args.note)


if __name__ == '__main__':
    main()
