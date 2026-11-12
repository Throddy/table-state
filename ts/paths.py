"""Пути к данным и артефактам. Только через TS_DATA и TS_ARTIFACTS (раздел 6)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent


def find_data_dir(search_root: str | Path = '/kaggle/input') -> Path:
    """Найти каталог данных так же, как baseline.ipynb организаторов.

    Каталог определяется по наличию `train_labels.csv`, `tables.csv` и
    `data_card.json`. Слаг соревнования не используется. Падает, если
    кандидатов не ровно один.
    """
    root = Path(search_root)
    candidates = [p.parent for p in root.rglob('train_labels.csv')
                  if (p.parent / 'tables.csv').is_file() and (p.parent / 'data_card.json').is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'В {root} найдено каталогов данных: {len(candidates)}. Укажите путь через TS_DATA.')
    return candidates[0]


def _resolve(var: str, default: str) -> Path:
    value = os.environ.get(var)
    path = Path(value) if value else ROOT / default
    return path.expanduser().resolve()


DATA: Path = _resolve('TS_DATA', 'data')
ART: Path = _resolve('TS_ARTIFACTS', 'artifacts')


def art(kind: str, name: str) -> Path:
    """Путь `ART/kind/name`. Каталог `ART/kind` создаётся при необходимости."""
    folder = ART / kind if kind else ART
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def image_path(frame_id: str) -> Path:
    """Путь к JPEG кадра: `DATA/<frame_id>.jpg`. Работает и для `aux_*`."""
    return DATA / f'{frame_id}.jpg'
