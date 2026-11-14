"""Сборка сабмита из колонок нескольких файлов с проверкой воспроизводимости по конфигу.

Запуск:
  python tools/combo.py --base NB1 --take hands=AB_ft_hands_pp items=OV_items_and_pp dishes=OV_dishes_owl_max \
      --config configs/COMBO.yaml --name COMBO

1. Берёт ART/preds/<base>.csv и подставляет колонки из указанных файлов ART/preds/<файл>.csv.
2. Проверяет, что каждая подставленная колонка совпадает со своим источником, а остальные с базой.
3. Собирает предсказание по `--config` (тот же код, что у run.py) и сверяет с результатом замены:
   счётчики обязаны совпасть точно, occ_prob до 1e-9 (бейзлайн организаторов не детерминирован до бита).
4. Пишет ART/preds/<name>.csv с карточкой, проводит через submit_gate; в configs/submissions/<name>.yaml
   попадает именно `--config`, поэтому `python run.py --config configs/submissions/<name>.yaml` даёт тот же файл.
На Kaggle не отправляет.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from _bootstrap import ROOT, git_sha

sys.path.insert(0, str(ROOT / 'tracks' / 'core'))
from build_pred import build  # noqa: E402
from swap_cols import pred_path  # noqa: E402

from ts import COUNTS, PRED_COLUMNS, io, paths  # noqa: E402
from ts.check import check_pred  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--take', nargs='*', default=[], metavar='колонка=файл', help='например hands=AB_ft_hands_pp')
    parser.add_argument('--config', type=Path, required=True, help='конфиг, сборка по которому обязана дать тот же файл')
    parser.add_argument('--name', required=True)
    parser.add_argument('--note')
    args = parser.parse_args()
    take = dict(item.split('=', 1) for item in args.take)
    wrong = [c for c in take if c not in PRED_COLUMNS[1:]]
    if wrong:
        raise SystemExit(f'Неизвестные колонки {wrong}')

    base = io.read_csv_exact(pred_path(args.base)).set_index('row_id')
    out = base.copy()
    sources = {}
    for column, name in take.items():
        check_pred(pred_path(name))
        sources[column] = io.read_csv_exact(pred_path(name)).set_index('row_id').loc[out.index]
        out[column] = sources[column][column]
    for column in PRED_COLUMNS[1:]:
        reference = sources[column][column] if column in sources else base[column]
        if not (out[column] == reference).all():
            raise SystemExit(f'Колонка {column} не совпала со своим источником')
    print('замена колонок: ' + ', '.join(f'{c} из {take.get(c, args.base)}' for c in PRED_COLUMNS[1:]))

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    built_path = build(f'{args.name}__cfg', cfg, notes=f'сборка по конфигу {args.config.name} для сверки с {args.name}',
                       command='python tools/combo.py ' + ' '.join(sys.argv[1:]))
    built = io.read_csv_exact(built_path).set_index('row_id').loc[out.index]
    tables = io.load_tables().set_index('row_id')
    test = tables.index[tables['split'] == 'test']
    problems = []
    for column in PRED_COLUMNS[1:]:
        for label, ids in [('test', test), ('train', out.index.difference(test))]:
            diff = (built.loc[ids, column] - out.loc[ids, column]).abs()
            limit = 1e-9 if column == 'occ_prob' else 0.0
            n = int((diff > limit).sum())
            print(f'  {column:<9} {label:<5}: расхождений {n:>4}, макс. {diff.max():.3g}')
            if n and label == 'test':
                problems.append(column)
    if problems:
        raise SystemExit(f'Сборка по конфигу {args.config} не совпала с заменой колонок в тесте: {problems}')

    # Итоговый файл это сборка по конфигу: её и воспроизводит run.py. Тестовые счётчики совпадают с заменой точно.
    final = built.reset_index()[PRED_COLUMNS]
    out_path = paths.art('preds', f'{args.name}.csv')
    final.to_csv(out_path, index=False)
    sha, _ = git_sha()
    cards = {n: json.loads(pred_path(n).with_suffix('.json').read_text(encoding='utf-8')) for n in {args.base, *take.values()}}
    card = {
        'name': args.name, 'kind': 'preds', 'owner': 'A',
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'command': 'python tools/combo.py ' + ' '.join(sys.argv[1:]),
        'combo': {'base': args.base, 'take': take,
                  'public_lb_of_parts': {n: c.get('public_lb') for n, c in cards.items()}},
        'reproduce': f'python run.py --data $TS_DATA --artifacts $TS_ARTIFACTS --config configs/submissions/{args.name}.yaml --out submission.csv',
        'models': sorted({m for c in cards.values() for m in c.get('models', [])}),
        'inputs': list(cfg['sources'].get('dets', [])) + list(cfg['sources'].get('rowfeats', [])),
        'sources': list(cfg['sources'].get('dets', [])) + list(cfg['sources'].get('rowfeats', [])),
        'train_pred': 'oof_by_date', 'manual_labels_used': False, 'uses_train_counters': True,
        'notes': f'{args.base} с колонками ' + ', '.join(f'{c} из {n}' for c, n in take.items())
                 + '; файл собран по конфигу и сверен с заменой колонок',
    }
    out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(out_path)
    note = args.note or f'Сборка колонок: {card["notes"]}'
    subprocess.run([sys.executable, str(ROOT / 'tools' / 'submit_gate.py'), '--pred', str(out_path),
                    '--config', str(args.config), '--note', note, '--brief'], check=True)


if __name__ == '__main__':
    main()
