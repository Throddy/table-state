"""Скачать веса моделей трека C с зафиксированной ревизией.

Запуск (один раз, нужен интернет): python tracks/ov/fetch_weights.py [ov_gdino] [ov_owl]
Файлы кладутся в ART/models/<source>/ (или в TS_OV_MODELS/<source>/, если переменная задана).
Инференс читает веса только оттуда и в сеть не ходит.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ts import paths  # noqa: E402

# source -> репозиторий Hugging Face и коммит (обе модели Apache-2.0, см. MODELS.md)
MODELS = {
    'ov_gdino': ('IDEA-Research/grounding-dino-base', '12bdfa3120f3e7ec7b434d90674b3396eccf88eb'),
    'ov_owl': ('google/owlv2-large-patch14-ensemble', '95e26936e865f87db1742128404b3c035d47d89d'),
}
FILES = ['*.json', '*.txt', 'model.safetensors']


def model_dir(source: str) -> Path:
    """Каталог весов источника: `TS_OV_MODELS/<source>`, по умолчанию `ART/models/<source>`."""
    root = os.environ.get('TS_OV_MODELS')
    return (Path(root) if root else paths.ART / 'models') / source


def main() -> None:
    from huggingface_hub import snapshot_download

    for source in sys.argv[1:] or list(MODELS):
        repo, revision = MODELS[source]
        target = snapshot_download(repo, revision=revision, local_dir=model_dir(source), allow_patterns=FILES)
        print(f'{source}: {repo}@{revision} -> {target}')


if __name__ == '__main__':
    main()
