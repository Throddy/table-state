"""Веса и колёса трека B с зафиксированными файлами релиза (разделы 6 и 6.1 контракта).

Запуск (один раз, с сетью: локально или в Kaggle-ноутбуке с включённым интернетом):
  python tracks/det/fetch_weights.py                 # веса в ART/ts-models/
  python tracks/det/fetch_weights.py --wheels        # ещё и колёса ultralytics для установки без сети
  python tracks/det/fetch_weights.py --check         # только сверка уже лежащих файлов, без сети

Каталог целиком публикуется как приватный Kaggle Dataset `ts-models`. Во время инференса и обучения
сеть не используется: скрипты трека принимают явный путь к файлу весов и сами ничего не скачивают.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts import paths  # noqa: E402

RELEASE_URL = 'https://github.com/ultralytics/assets/releases/download/{release}/{name}'
LICENSE = 'AGPL-3.0'
# Точное имя файла релиза, размер и sha256 (поле digest у ассета релиза на GitHub, сверено скачиванием).
WEIGHTS: dict[str, dict] = {
    'yolo11x.pt': {
        'release': 'v8.4.0', 'bytes': 114636239,
        'sha256': '7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24',
        'use': 'COCO-детектор без обучения: источники coco_person и coco_crop',
    },
    'yolo11m.pt': {
        'release': 'v8.4.0', 'bytes': 40684120,
        'sha256': 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95',
        'use': 'стартовые COCO-веса для дообучения ft_yolo (B4)',
    },
    'yolo11x-pose.pt': {
        'release': 'v8.4.0', 'bytes': 118481010,
        'sha256': '013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8',
        'use': 'запястья для источника pose (B5)',
    },
}
# Чистые python-колёса (py3-none-any). torch, torchvision, opencv и остальное берутся из образа Kaggle.
WHEELS = ['ultralytics==8.4.211', 'ultralytics-thop==2.2.5', 'nvidia-ml-py==13.615.71']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def url_of(name: str) -> str:
    return RELEASE_URL.format(release=WEIGHTS[name]['release'], name=name)


def model_ref(path: Path) -> str:
    """Строка для поля `models` карточки. Падает, если файл не совпал с зафиксированным."""
    name = Path(path).name
    if name not in WEIGHTS:
        raise SystemExit(f'{name} нет в таблице WEIGHTS (tracks/det/fetch_weights.py).')
    problem = verify(Path(path))
    if problem:
        raise SystemExit(problem)
    return f"ultralytics/assets@{WEIGHTS[name]['release']}:{name} sha256:{WEIGHTS[name]['sha256']}"


def verify(path: Path) -> str:
    """Пустая строка, если файл совпал с таблицей по размеру и sha256, иначе описание расхождения."""
    spec = WEIGHTS[path.name]
    if not path.is_file():
        return f'{path}: файла нет'
    if path.stat().st_size != spec['bytes']:
        return f"{path}: размер {path.stat().st_size}, ожидается {spec['bytes']}"
    got = sha256(path)
    return '' if got == spec['sha256'] else f"{path}: sha256 {got}, ожидается {spec['sha256']}"


def fetch(name: str, out: Path) -> Path:
    """Скачать файл релиза, если его ещё нет, и сверить размер и sha256."""
    path = out / name
    if path.is_file() and not verify(path):
        print(f'{name}: уже на месте, sha256 совпал')
        return path
    tmp = path.with_suffix('.part')
    print(f'{name}: скачиваю {url_of(name)}')
    with urllib.request.urlopen(url_of(name), timeout=120) as response, open(tmp, 'wb') as f:
        for block in iter(lambda: response.read(1 << 20), b''):
            f.write(block)
    tmp.replace(path)
    problem = verify(path)
    if problem:
        path.unlink()
        raise SystemExit(f'Скачанный файл не совпал с зафиксированным: {problem}')
    print(f'{name}: {path.stat().st_size} байт, sha256 совпал')
    return path


def fetch_wheels(out: Path) -> list[Path]:
    """Колёса из списка WHEELS в `out/wheels`, без зависимостей (их даёт образ Kaggle)."""
    folder = out / 'wheels'
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary', ':all:',
                    '--dest', str(folder), *WHEELS], check=True)
    wheels = sorted(folder.glob('*.whl'))
    wrong = [w.name for w in wheels if not w.name.endswith('py3-none-any.whl')]
    if wrong:
        raise SystemExit(f'Колёса зависят от платформы и на Kaggle могут не встать: {wrong}')
    return wheels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=paths.ART / 'ts-models', help='каталог будущего датасета ts-models')
    parser.add_argument('--only', nargs='*', choices=sorted(WEIGHTS), help='только эти файлы весов')
    parser.add_argument('--wheels', action='store_true', help='скачать колёса ultralytics для установки без сети')
    parser.add_argument('--check', action='store_true', help='только сверить лежащие файлы, сеть не трогать')
    args = parser.parse_args()
    names = args.only or sorted(WEIGHTS)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.check:
        problems = [p for p in (verify(args.out / n) for n in names) if p]
        print('\n'.join(problems) if problems else f'OK: {len(names)} файлов весов совпали с таблицей')
        sys.exit(1 if problems else 0)

    for name in names:
        fetch(name, args.out)
    wheels = fetch_wheels(args.out) if args.wheels else sorted((args.out / 'wheels').glob('*.whl'))
    manifest = {
        'license': LICENSE,
        'weights': {n: {**WEIGHTS[n], 'url': url_of(n)} for n in sorted(WEIGHTS) if (args.out / n).is_file()},
        'wheels': {w.name: sha256(w) for w in wheels},
        'wheels_spec': WHEELS,
    }
    (args.out / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Готово: {args.out} (весов: {len(manifest["weights"])}, колёс: {len(wheels)}). '
          'Каталог публикуется как Kaggle Dataset ts-models.')


if __name__ == '__main__':
    main()
