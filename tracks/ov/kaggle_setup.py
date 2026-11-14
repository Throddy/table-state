"""Подготовка сессии Kaggle для задач C2 и C5 (раздел 6.1 контракта). Ничего не считает, только настраивает.

В ноутбуке, до первого импорта ts:   %run tracks/ov/kaggle_setup.py

- находит каталог данных соревнования по train_labels.csv, tables.csv и data_card.json (слаг не используется);
- выставляет TS_DATA, TS_ARTIFACTS (/kaggle/working/artifacts) и TS_OV_MODELS, их наследуют команды `!python ...`;
- проверяет Python, torch, CUDA, GPU, transformers и scipy;
- восстанавливает готовые файлы и незавершённые порции прошлых запусков из подключённых датасетов;
- берёт веса из подключённого датасета (например, `ts-models-ov`), а если их нет, скачивает `fetch_weights.py`
  (нужен интернет). `OV_SOURCES=ov_owl` в окружении: нужны веса только этой модели (по умолчанию обеих).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get('TS_KAGGLE_INPUT', '/kaggle/input'))
WORKING = Path(os.environ.get('TS_KAGGLE_WORKING', '/kaggle/working'))
SOURCES = [s for s in os.environ.get('OV_SOURCES', 'ov_gdino,ov_owl').split(',') if s]
RESTORE = ['rows_ext.parquet', 'dets/ov_*__v*.parquet', 'dets/ov_*__v*.json', 'dets/ov_*__v*.parts',
           'labels/ov_pseudo__v*.parquet', 'labels/ov_pseudo__v*.json',
           'rowfeats/ov_counts__v*.parquet', 'rowfeats/ov_counts__v*.json']
MIN_TRANSFORMERS = (4, 40)


def data_dir() -> Path:
    """Каталог данных соревнования: так же, как в baseline.ipynb организаторов."""
    found = sorted({p.parent for p in INPUT.rglob('train_labels.csv')
                    if (p.parent / 'tables.csv').is_file() and (p.parent / 'data_card.json').is_file()})
    if len(found) != 1:
        raise RuntimeError(f'Каталогов данных соревнования в {INPUT}: {len(found)}. Подключите датасет соревнования.')
    return found[0]


def check_env() -> None:
    """Версии и GPU. Падает, если нет GPU (кроме TS_ALLOW_CPU=1) или transformers слишком старый."""
    import scipy
    import torch
    import transformers
    print(f'python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}, '
          f'transformers {transformers.__version__}, scipy {scipy.__version__}')
    if tuple(int(v) for v in transformers.__version__.split('.')[:2]) < MIN_TRANSFORMERS:
        raise RuntimeError('Нужен transformers >= 4.40: выполните `pip install -q transformers==5.22.2` (с интернетом)')
    if not torch.cuda.is_available():
        if os.environ.get('TS_ALLOW_CPU') != '1':
            raise RuntimeError('GPU не найден: Settings -> Accelerator -> GPU T4 x2')
        print('GPU нет, разрешён запуск на CPU (TS_ALLOW_CPU=1)')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'cuda:{i} {props.name}, {props.total_memory / 2 ** 30:.1f} ГБ')


def restore(art: Path) -> None:
    """Скопировать из подключённых датасетов готовые файлы и порции, которых ещё нет в `art`."""
    copied = 0
    for pattern in RESTORE:
        for src in sorted(INPUT.rglob(pattern)):
            dst = art / Path(pattern).parent / src.name
            for item in ([src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix != '.tmp')):
                target = dst if src.is_file() else dst / item.name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                    copied += 1
    print(f'восстановлено файлов из подключённых датасетов: {copied}')


def weights(art: Path) -> Path:
    """Каталог с весами обеих моделей: локальный, из подключённого датасета или свежескачанный."""
    attached = sorted({p.parents[1] for p in INPUT.rglob(f'{SOURCES[0]}/model.safetensors')})
    for folder in [art / 'models'] + attached:
        if all((folder / source / 'model.safetensors').is_file() for source in SOURCES):
            return folder
    print(f'весов {SOURCES} нет ни в artifacts/models, ни в подключённых датасетах: скачиваю (нужен интернет)')
    env = {**os.environ, 'TS_OV_MODELS': str(art / 'models')}
    if subprocess.run([sys.executable, str(ROOT / 'tracks' / 'ov' / 'fetch_weights.py'), *SOURCES], env=env).returncode:
        raise RuntimeError('Веса не скачались: включите Internet в настройках ноутбука или подключите датасет с весами')
    return art / 'models'


def main() -> None:
    data, art = data_dir(), WORKING / 'artifacts'
    art.mkdir(parents=True, exist_ok=True)
    os.environ.update(TS_DATA=str(data), TS_ARTIFACTS=str(art))
    os.chdir(ROOT)
    check_env()
    restore(art)
    os.environ['TS_OV_MODELS'] = str(weights(art))
    print(f"код: {ROOT} (TS_GIT_SHA={os.environ.get('TS_GIT_SHA') or 'не задан'})")
    for key in ['TS_DATA', 'TS_ARTIFACTS', 'TS_OV_MODELS']:
        print(f'{key}={os.environ[key]}')


main()
