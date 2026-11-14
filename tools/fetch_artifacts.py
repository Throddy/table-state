"""Скачивание данных соревнования и кэшей источников, нужных для сборки финала из кэша.

Запуск: python tools/fetch_artifacts.py [--data data] [--artifacts artifacts] [--skip-data]

Нужен токен Kaggle (~/.kaggle/kaggle.json или ~/.kaggle/access_token), принятые правила соревнования
rascar-hack-case-2 и доступ к датасетам и ноутбукам из таблицы ниже.

| что | откуда | куда |
|---|---|---|
| пакет организаторов | соревнование rascar-hack-case-2 | data/ |
| руки своего детектора ft_yolo_v1 | датасет victimestsate/ts-artifacts-dets | artifacts/dets/ft_yolo__v1.* |
| счётчики Grounding DINO и OWLv2 по строкам | датасет throddy/ts-artifacts-ov-pseudo | artifacts/rowfeats/ov_counts__v1.* |
| ответы Qwen3-VL-4B, шарды 0–1 и 2–3 | вывод ноутбуков throddy/ts-det-vlm-shards-0-1 и throddy/notebook15b164940c | artifacts/rowfeats/_parts/vlm_qwen__v1/ |

Ответы VLM склеиваются в artifacts/rowfeats/vlm_qwen__v1.parquet командой run_vlm.py --merge, модель не запускается.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPETITION = 'rascar-hack-case-2'
DATASET_FILES = [
    ('victimestsate/ts-artifacts-dets', 'dets/ft_yolo__v1.parquet', 'dets'),
    ('victimestsate/ts-artifacts-dets', 'dets/ft_yolo__v1.json', 'dets'),
    ('throddy/ts-artifacts-ov-pseudo', 'artifacts/rowfeats/ov_counts__v1.parquet', 'rowfeats'),
    ('throddy/ts-artifacts-ov-pseudo', 'artifacts/rowfeats/ov_counts__v1.json', 'rowfeats'),
]
VLM_KERNELS = ['throddy/ts-det-vlm-shards-0-1', 'throddy/notebook15b164940c']
VLM_NAME, VLM_SHARDS = 'vlm_qwen__v1', 4


def kaggle(*args: str) -> None:
    """Команда Kaggle CLI из того же окружения, что и этот скрипт."""
    exe = Path(sys.executable).parent / 'kaggle'
    command = [str(exe) if exe.exists() else 'kaggle', *args]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f'Ошибка Kaggle CLI ({" ".join(args[:3])}): {done.stdout[-500:]} {done.stderr[-500:]}')


def unzip_all(folder: Path) -> None:
    for archive in list(folder.rglob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            z.extractall(archive.parent)
        archive.unlink()


def fetch_data(data: Path) -> None:
    if (data / 'train_labels.csv').is_file():
        print(f'данные уже есть: {data}')
        return
    data.mkdir(parents=True, exist_ok=True)
    print(f'скачиваю пакет соревнования {COMPETITION} в {data} (около 570 МБ)')
    kaggle('competitions', 'download', '-c', COMPETITION, '-p', str(data))
    unzip_all(data)
    if not (data / 'train_labels.csv').is_file():
        raise SystemExit(f'В {data} нет train_labels.csv после распаковки')


def fetch_dataset_files(art: Path) -> None:
    for dataset, path, kind in DATASET_FILES:
        target = art / kind / Path(path).name
        if target.is_file():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            kaggle('datasets', 'download', '-d', dataset, '-f', path, '-p', tmp, '--force')
            unzip_all(Path(tmp))
            found = list(Path(tmp).rglob(Path(path).name))
            if len(found) != 1:
                raise SystemExit(f'{dataset}/{path}: файл не скачался')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(found[0]), target)
        print(f'{target.relative_to(art.parent) if art.parent in target.parents else target}  <-  {dataset}/{path}')


def fetch_vlm(art: Path, data: Path) -> None:
    merged = art / 'rowfeats' / f'{VLM_NAME}.parquet'
    if merged.is_file():
        print(f'{merged.name} уже есть')
        return
    parts = art / 'rowfeats' / '_parts' / VLM_NAME
    parts.mkdir(parents=True, exist_ok=True)
    for kernel in VLM_KERNELS:
        with tempfile.TemporaryDirectory() as tmp:
            kaggle('kernels', 'output', kernel, '-p', tmp, '--file-pattern', r'.*shard_\d+_of_\d+\.jsonl$')
            for shard in Path(tmp).rglob('shard_*_of_*.jsonl'):
                shutil.move(str(shard), parts / shard.name)
                print(f'{parts.name}/{shard.name}  <-  вывод ноутбука {kernel}')
    found = sorted(p.name for p in parts.glob('shard_*_of_*.jsonl'))
    expected = [f'shard_{k}_of_{VLM_SHARDS}.jsonl' for k in range(VLM_SHARDS)]
    if found != expected:
        raise SystemExit(f'Нужны шарды {expected}, найдены {found}')
    env = {**os.environ, 'TS_DATA': str(data), 'TS_ARTIFACTS': str(art)}
    subprocess.run([sys.executable, str(ROOT / 'tracks' / 'det' / 'vlm' / 'run_vlm.py'), '--subset', 'all',
                    '--of', str(VLM_SHARDS), '--merge'], check=True, env=env, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--skip-data', action='store_true', help='не скачивать пакет соревнования')
    args = parser.parse_args()
    data, art = args.data.resolve(), args.artifacts.resolve()
    if not args.skip_data:
        fetch_data(data)
    fetch_dataset_files(art)
    fetch_vlm(art, data)
    print(f'готово: данные {data}, кэши {art}')


if __name__ == '__main__':
    main()
