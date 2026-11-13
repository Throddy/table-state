"""Общее для скриптов tools/: корень репозитория в sys.path и импорт baseline.py организаторов."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def organizer_baseline():
    """Модуль `baseline.py` из каталога данных, без изменений (он импортирует свой `score.py`)."""
    from ts import paths
    if str(paths.DATA) not in sys.path:
        sys.path.insert(0, str(paths.DATA))
    import baseline
    return baseline


def git_sha() -> tuple[str, bool]:
    """Короткий sha текущего коммита и признак незакоммиченных изменений."""
    def run(*args: str) -> str:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return run('rev-parse', '--short', 'HEAD') or 'no-commit', bool(run('status', '--porcelain', '--untracked-files=no'))
