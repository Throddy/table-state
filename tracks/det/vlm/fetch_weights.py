"""Веса VLM-счётчика с зафиксированной ревизией Hugging Face.

Запуск (один раз, с сетью): python tracks/det/vlm/fetch_weights.py [--out ART/models/qwen3-vl-4b-instruct]
На Kaggle те же файлы берутся из Kaggle Models `qwen-lm/qwen-3-vl/transformers/4b-instruct/1`
(sha256 двух файлов safetensors совпадает с этой ревизией). При инференсе сеть не используется.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ts import paths  # noqa: E402

REPO = 'Qwen/Qwen3-VL-4B-Instruct'
REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
LICENSE = 'Apache-2.0'
# Размер и sha256 файлов весов этой ревизии (поле lfs.oid на Hugging Face).
SHA256 = {
    'model-00001-of-00002.safetensors': (4967229296, '30a01a0556622645'),
    'model-00002-of-00002.safetensors': (3908490048, '046296a2a387efb4'),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=paths.ART / 'models' / 'qwen3-vl-4b-instruct')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=REVISION, local_dir=str(args.out))
    for name, (size, prefix) in SHA256.items():
        file = args.out / name
        got = sha256(file)
        if file.stat().st_size != size or not got.startswith(prefix):
            raise SystemExit(f'{name}: размер {file.stat().st_size}, sha256 {got[:16]}, ожидалось {size} и {prefix}')
        print(f'{name}: {size} байт, sha256 {got[:16]} совпал')
    print(f'Готово: {args.out} ({REPO}@{REVISION})')


if __name__ == '__main__':
    main()
