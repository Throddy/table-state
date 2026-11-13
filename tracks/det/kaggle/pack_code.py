"""Архив кода для Kaggle Dataset `ts-code`: закоммиченное состояние репозитория и файл GIT_SHA.

Запуск: python tracks/det/kaggle/pack_code.py [--out artifacts/ts-code.zip]
В архив попадает ровно HEAD (git archive), незакоммиченные правки не попадают. Файл GIT_SHA нужен,
чтобы карточки артефактов, посчитанных на Kaggle, содержали git sha (там нет каталога .git).
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'ts-code.zip')
    args = parser.parse_args()

    def git(*a: str) -> str:
        return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    sha = git('rev-parse', '--short', 'HEAD')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Есть незакоммиченные изменения: сначала коммит, иначе git sha в карточках будет неверным.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'archive', '--format=zip', '-o', str(args.out), 'HEAD', '--', '.', ':!submissions'],
                   cwd=ROOT, check=True)
    with zipfile.ZipFile(args.out, 'a') as archive:
        archive.writestr('GIT_SHA', sha + '\n')
    print(f'{args.out}: коммит {sha}, {args.out.stat().st_size / 1e6:.2f} МБ. Загрузите как Kaggle Dataset ts-code.')


if __name__ == '__main__':
    main()
