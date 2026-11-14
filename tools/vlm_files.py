"""Три файла с колонками VLM поверх COMBO: сборка по конфигам, gate, сводка изменений. На Kaggle не отправляет.

Запуск: python tools/vlm_files.py [--rowfeats vlm_qwen__v1] [--base COMBO] [--only VLM_people VLM_items]
Нужен ART/rowfeats/<rowfeats>.parquet с карточкой (вывод tracks/det/vlm/run_vlm.py --merge).
Конфиги configs/VLM_people.yaml, VLM_items.yaml, VLM_both.yaml: это COMBO плюс блок overrides.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd
import yaml

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT / 'tracks' / 'core'))
from build_pred import build  # noqa: E402
from swap_cols import pred_path  # noqa: E402

from ts import PRED_COLUMNS, io, paths  # noqa: E402
from ts.check import check_rowfeats  # noqa: E402
from ts.cv import evaluate  # noqa: E402

FILES = {'VLM_people': (['people'], 'COMBO с people из VLM-счётчика (замена): убирает ли VLM лишних людей и находит ли пропущенных?'),
         'VLM_items': (['items'], 'COMBO с items = max(COMBO, VLM): добавляет ли VLM вещи, которых не видят модели трека C?'),
         'VLM_both': (['people', 'items'], 'COMBO с people из VLM (замена) и items = max(COMBO, VLM) одновременно')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rowfeats', default='vlm_qwen__v1')
    parser.add_argument('--base', default='COMBO')
    parser.add_argument('--only', nargs='+', choices=list(FILES), default=list(FILES))
    parser.add_argument('--no-gate', action='store_true')
    args = parser.parse_args()
    path = paths.ART / 'rowfeats' / f'{args.rowfeats}.parquet'
    check_rowfeats(path)
    vlm = pd.read_parquet(path)
    source = args.rowfeats.split('__v')[0]
    print(f'ts.check: {args.rowfeats} OK; строк {len(vlm)}, без ответа {(vlm[source + "__ok"] == 0).sum()}')
    base = io.read_csv_exact(pred_path(args.base)).set_index('row_id')
    tables = io.load_tables().set_index('row_id')
    test = tables.index[tables['split'] == 'test']
    date = tables.loc[test, 'date_utc'].str[5:]
    report = {args.base: evaluate(base.reset_index()[PRED_COLUMNS], bootstrap=0)}
    for name in args.only:
        columns, note = FILES[name]
        cfg = yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text(encoding='utf-8'))
        cfg['sources']['rowfeats'] = [args.rowfeats if r.startswith(source + '__v') else r for r in cfg['sources']['rowfeats']]
        out = build(name, cfg, notes=note, command='python tools/vlm_files.py ' + ' '.join(sys.argv[1:]))
        effective = out.with_suffix('.config.yaml')      # конфиг, по которому файл реально собран, его и замораживает gate
        effective.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        new = io.read_csv_exact(out).set_index('row_id').loc[base.index]
        report[name] = evaluate(new.reset_index()[PRED_COLUMNS], bootstrap=0)
        print(f'\n{name}: {note}')
        for column in PRED_COLUMNS[1:]:
            diff = (new.loc[test, column] - base.loc[test, column])
            changed = diff.abs() > 1e-9
            if column in columns:
                by = changed.groupby(date).sum()
                print(f"  {column:<9}: изменено {int(changed.sum())} строк теста ({', '.join(f'{d}: {int(n)}' for d, n in by.items())}), "
                      f"вверх {int((diff > 1e-9).sum())}, вниз {int((diff < -1e-9).sum())}, сумма {base.loc[test, column].sum():g} -> {new.loc[test, column].sum():g}")
            elif changed.any():
                raise SystemExit(f'{name}: колонка {column} изменилась в {int(changed.sum())} строках, хотя не должна')
        print('  остальные колонки совпадают с основой (hands и items посчитаны по people основы)')
        if not args.no_gate:
            done = subprocess.run([sys.executable, str(ROOT / 'tools' / 'submit_gate.py'), '--pred', str(out),
                                   '--config', str(effective), '--note', note, '--brief'],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                out.unlink(missing_ok=True)
                raise SystemExit(f'{name}: gate не прошёл, файл удалён: {done.stdout[-400:]} {done.stderr[-400:]}')
            print('  ' + '\n  '.join(line for line in done.stdout.splitlines() if 'GATE' in line or 'kaggle competitions' in line))
    print(f"\n{'файл':<12}{'oof_all':>8}{'f25':>8}{'f26':>8}{'robust':>8}{'people':>8}{'items':>8}   ошибка people / items")
    for name, r in report.items():
        a = r['oof_all']
        print(f"{name:<12}{a['score']:>8.4f}{r['f25']['score']:>8.4f}{r['f26']['score']:>8.4f}{r['robust']['oof_all']['score']:>8.4f}"
              f"{a['quality']['people']:>8.3f}{a['quality']['items']:>8.3f}   {a['abs_error']['people']:g}/29  {a['abs_error']['items']:g}/28")


if __name__ == '__main__':
    main()
