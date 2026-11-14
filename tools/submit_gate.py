"""Проверка и регистрация сабмита (раздел 14). На Kaggle ничего не отправляет.

Запуск:
  python tools/submit_gate.py --pred artifacts/preds/<name>.csv --note "<вопрос, на который отвечает сабмит>"

Шаги: ts.check_pred -> 4733 тестовые строки в порядке шаблона -> валидатор
организаторов -> не меньше 200 уникальных occ_prob -> submissions/<name>.csv,
submissions/<name>.json и копия конфига в configs/submissions/<name>.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT, git_sha

from ts import PRED_COLUMNS, io, paths
from ts.check import check_pred
from ts.cv import evaluate, format_report
from ts.vendor import score as vscore

MIN_UNIQUE_OCC = 200


def rel(path: Path) -> str:
    """Путь относительно корня репозитория; вне репозитория остаётся только имя файла (без домашних каталогов)."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else resolved.name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred', type=Path, required=True, help='artifacts/preds/<name>.csv, все 4964 строки')
    parser.add_argument('--note', required=True, help='один вопрос, на который отвечает сабмит')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    parser.add_argument('--name', help='имя сабмита, по умолчанию имя файла предсказаний')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--brief', action='store_true', help='печатать только oof_all, f25, f26 и robust')
    args = parser.parse_args()
    name = args.name or args.pred.stem

    check_pred(args.pred)
    pred = io.read_csv_exact(args.pred)
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')
    submission = pred.set_index('row_id').loc[sample['row_id']].reset_index()[PRED_COLUMNS]
    if len(submission) != len(sample) or not (submission['row_id'] == sample['row_id']).all():
        raise SystemExit('Строки сабмита не совпали с шаблоном sample_submission.csv')
    vscore.validate_submission(submission, sample['row_id'])
    unique = int(submission['occ_prob'].nunique())
    if unique < MIN_UNIQUE_OCC:
        raise SystemExit(f'occ_prob: {unique} уникальных значений, нужно не меньше {MIN_UNIQUE_OCC} '
                         '(AP зависит от порядка, раздел 1 контракта)')

    report = evaluate(pred, bootstrap=args.bootstrap)
    pred_card = json.loads(args.pred.with_suffix('.json').read_text(encoding='utf-8'))
    sha, dirty = git_sha()

    out_dir = ROOT / 'submissions'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f'{name}.csv'
    submission.to_csv(out_csv, index=False)
    config_copy = ROOT / 'configs' / 'submissions' / f'{name}.yaml'
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, config_copy)
    card = {
        'name': name,
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'git_dirty': dirty,
        'note': args.note,
        'command': f'python tools/submit_gate.py --pred {rel(args.pred)} --config {rel(args.config)} --note "..."',
        'pred': rel(args.pred),
        'pred_sha256': sha256(args.pred),
        'submission_sha256': sha256(out_csv),
        'rows': len(submission),
        'occ_prob_unique': unique,
        'config': str(config_copy.relative_to(ROOT)),
        'sources': pred_card.get('sources', []),
        'pred_card': pred_card,
        'local_metrics': report,
        'public_lb': None,
    }
    (out_dir / f'{name}.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    text = format_report(report)
    if args.brief:
        text = '\n'.join(line for line in text.splitlines() if not line.startswith(('ir', 'robust/f', 'подгоняемых')))
    print(text)
    print(f'\nGATE OK: {out_csv.relative_to(ROOT)} ({len(submission)} строк, occ_prob уникальных: {unique}), '
          f'карточка {name}.json, конфиг {config_copy.relative_to(ROOT)}')
    if dirty:
        print('ВНИМАНИЕ: в репозитории есть незакоммиченные изменения, git_sha в карточке неточен.')
    print('Отправка (делает A вручную, после неё дописать public_lb в карточку и строку в журнал):')
    print(f'  kaggle competitions submit -c rascar-hack-case-2 -f submissions/{name}.csv -m "{name} {sha}"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
