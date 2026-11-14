"""Сборка обоих отмеченных финалов и сверка с сохранёнными сабмитами.

Запуск: python tools/verify_final.py [--data data] [--artifacts artifacts] [--out <каталог>]

Для каждого финала вызывает run.py с замороженным конфигом из configs/submissions/ и сравнивает
результат с submissions/<имя>.csv. Пять счётчиков должны совпасть побайтно. occ_prob допускает
расхождение до 1e-12 при одинаковом порядке строк: бейзлайн организаторов (ExtraTrees, n_jobs=2)
складывает деревья в двух потоках, и младшие биты меняются от запуска к запуску. Скор от этого
не меняется, потому что AP зависит только от порядка.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINALS = ['VLM_hands_avg', 'VLM_items']
COUNTS = ['people', 'hands', 'cups', 'dishes', 'items']


def compare(new: pd.DataFrame, saved: pd.DataFrame) -> list[str]:
    """Список расхождений; пустой, если файл совпадает с сохранённым."""
    problems = []
    if len(new) != len(saved) or not (new['row_id'] == saved['row_id']).all():
        return ['не совпали строки']
    for column in COUNTS:
        changed = int((new[column] != saved[column]).sum())
        if changed:
            problems.append(f'{column}: отличается в {changed} строках')
    diff = (new['occ_prob'] - saved['occ_prob']).abs().max()
    same_order = (new['occ_prob'].rank(method='min') == saved['occ_prob'].rank(method='min')).all()
    if diff > 1e-12 or not same_order:
        problems.append(f'occ_prob: максимум расхождения {diff:.3g}, порядок {"тот же" if same_order else "другой"}')
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--out', type=Path, help='куда сохранить собранные файлы (по умолчанию временный каталог)')
    args = parser.parse_args()
    out_dir = args.out or Path(tempfile.mkdtemp(prefix='verify_final_'))
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in FINALS:
        out = out_dir / f'{name}.csv'
        started = time.time()
        subprocess.run([sys.executable, str(ROOT / 'run.py'), '--data', str(args.data), '--artifacts', str(args.artifacts),
                        '--config', str(ROOT / 'configs' / 'submissions' / f'{name}.yaml'), '--out', str(out), '--from-cache'],
                       check=True, stdout=subprocess.DEVNULL)
        new, saved = pd.read_csv(out), pd.read_csv(ROOT / 'submissions' / f'{name}.csv')
        problems = compare(new, saved)
        diff = (new['occ_prob'] - saved['occ_prob']).abs().max()
        status = 'OK' if not problems else 'РАСХОЖДЕНИЕ: ' + '; '.join(problems)
        print(f'{name}: {time.time() - started:.0f} с, счётчики совпадают побайтно, occ_prob до {diff:.1e} '
              f'при том же порядке строк -> {status}' if not problems else f'{name}: {status}')
        failed |= bool(problems)
    print(f'файлы: {out_dir}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
