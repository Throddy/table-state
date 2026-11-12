"""Общие фикстуры. Тесты, которым нужен пакет организаторов, пропускаются без него."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ts import paths  # noqa: E402

HAS_DATA = (paths.DATA / 'tables.csv').is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason='нет каталога данных TS_DATA')
