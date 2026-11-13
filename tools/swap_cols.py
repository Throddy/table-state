"""Абляция заменой колонок: базовое предсказание с отдельными колонками из другого файла.

Метрика аддитивна по колонкам, поэтому вклад одной колонки проверяется сабмитом,
в котором заменена только она.

Запуск:
  python tools/swap_cols.py --base base_calibrated --donor R1c --cols occ_prob --name AB_occ
  python tools/swap_cols.py --base base_calibrated --donor R1c --cols hands cups dishes --name AB_small

`--base` и `--donor`: имена файлов ART/preds/<имя>.csv (4964 строки, с карточкой) или пути к ним.
Пишет ART/preds/<name>.csv с карточкой и рецептом, затем запускает tools/submit_gate.py
(submissions/<name>.csv, <name>.json, configs/submissions/<name>.yaml). На Kaggle не отправляет.
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

from ts import PRED_COLUMNS, io, paths
from ts.check import check_pred


def pred_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    return path if path.suffix == '.csv' else paths.ART / 'preds' / f'{name_or_path}.csv'


def swap(base: str, donor: str, cols: list[str], name: str, note: str | None = None, gate: bool = True) -> Path:
    """Собрать `ART/preds/<name>.csv`: файл `base` с колонками `cols` из `donor`, и провести через gate."""
    wrong = [c for c in cols if c not in PRED_COLUMNS[1:]]
    if wrong or not cols:
        raise SystemExit(f'Неизвестные колонки {wrong}, допустимы: {PRED_COLUMNS[1:]}')
    base_path, donor_path = pred_path(base), pred_path(donor)
    for path in (base_path, donor_path):
        check_pred(path)
    out = io.read_csv_exact(base_path).set_index('row_id')
    source = io.read_csv_exact(donor_path).set_index('row_id').loc[out.index]
    changed = {c: int((out[c] != source[c]).sum()) for c in cols}
    out[cols] = source[cols]
    out_path = paths.art('preds', f'{name}.csv')
    out.reset_index()[PRED_COLUMNS].to_csv(out_path, index=False)

    cards = [json.loads(p.with_suffix('.json').read_text(encoding='utf-8')) for p in (base_path, donor_path)]
    sha, _ = git_sha()
    recipe = {'base': base_path.stem, 'donor': donor_path.stem, 'cols': cols}
    card = {
        'name': name, 'kind': 'preds', 'owner': 'A',
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_sha': sha,
        'command': 'python tools/swap_cols.py ' + ' '.join(sys.argv[1:]),
        'swap': {**recipe, 'rows_changed': changed, 'base_git_sha': cards[0].get('git_sha'),
                 'donor_git_sha': cards[1].get('git_sha')},
        'models': sorted(set(cards[0].get('models', [])) | set(cards[1].get('models', []))),
        'inputs': [f'preds/{base_path.name}', f'preds/{donor_path.name}'],
        'sources': sorted(set(cards[0].get('sources', [])) | set(cards[1].get('sources', []))),
        'train_pred': 'oof_by_date' if 'oof_by_date' in (cards[0]['train_pred'], cards[1]['train_pred']) else 'zero_shot',
        'manual_labels_used': False,
        'uses_train_counters': bool(cards[0]['uses_train_counters'] or cards[1]['uses_train_counters']),
        'notes': f'{base_path.stem} с колонками {", ".join(cols)} из {donor_path.stem}',
    }
    out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    recipe_path = out_path.with_suffix('.swap.yaml')
    recipe_path.write_text('# Рецепт сабмита: базовый файл и колонки, заменённые из другого предсказания.\n'
                           + yaml.safe_dump({'swap': recipe}, allow_unicode=True, sort_keys=False), encoding='utf-8')
    check_pred(out_path)
    print(f'{name}: {card["notes"]}; изменено строк (из 4964): {changed}')
    if gate:
        note = note or f'Вклад колонок {", ".join(cols)} из {donor_path.stem} при замене в {base_path.stem}'
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'submit_gate.py'), '--pred', str(out_path),
                        '--config', str(recipe_path), '--note', note, '--brief'], check=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--donor', required=True)
    parser.add_argument('--cols', nargs='+', required=True, help='колонки для замены: ' + ', '.join(PRED_COLUMNS[1:]))
    parser.add_argument('--name', required=True)
    parser.add_argument('--note', help='вопрос, на который отвечает сабмит')
    parser.add_argument('--no-gate', action='store_true', help='только собрать ART/preds/<name>.csv')
    args = parser.parse_args()
    swap(args.base, args.donor, args.cols, args.name, args.note, gate=not args.no_gate)


if __name__ == '__main__':
    main()
