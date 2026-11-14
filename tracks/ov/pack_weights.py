"""Офлайн-веса трека C: каталог для Kaggle Dataset `ts-models-ov` (Grounding DINO base и OWLv2 large ensemble).

Запуск:
  python tracks/ov/pack_weights.py --out /kaggle/working/ts-models-ov   # скачать (нужен интернет), проверить, манифест
  python tracks/ov/pack_weights.py --check artifacts/models             # только проверить готовый каталог без сети

Что делает:
1. скачивает файлы `fetch_weights.FILES` каждой модели с зафиксированной ревизией `fetch_weights.MODELS`
   в `<out>/<source>/` (служебный `.cache` удаляется);
2. сверяет sha256 `model.safetensors` с ожидаемым значением (`SHA256`, совпадает с Hugging Face для этих ревизий);
3. в отдельном процессе с `HF_HUB_OFFLINE=1` загружает процессор, токенизатор и модель тем же кодом, что и
   `run_ov.py` (классы `GroundingDino` и `Owl`, промпт v2): так проверяется, что для работы без интернета хватает
   файлов каталога;
4. пишет `<out>/MANIFEST.json` (репозиторий, ревизия, лицензия, размер и sha256 каждого файла) и `<out>/README.md`.

Каталог подключается как Input; `kaggle_setup.py` находит его по `ov_gdino/model.safetensors` или
`ov_owl/model.safetensors` и выставляет `TS_OV_MODELS`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_weights import FILES, MODELS  # noqa: E402

LICENSE = {'ov_gdino': 'apache-2.0', 'ov_owl': 'apache-2.0'}   # карточки моделей на Hugging Face, проверено 14.11
SHA256 = {   # model.safetensors для ревизий из fetch_weights.MODELS (lfs.sha256 на Hugging Face, сверено 14.11)
    'ov_gdino': '5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21',
    'ov_owl': 'd1c2261503c55aaf400667a843a54a5167e3c696334674c4093d6d10f7f40075',
}
PROMPT_VERSION = 2
OFFLINE = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def download(out: Path, sources: list[str]) -> None:
    from huggingface_hub import snapshot_download
    for source in sources:
        repo, revision = MODELS[source]
        target = out / source
        snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=FILES)
        shutil.rmtree(target / '.cache', ignore_errors=True)
        print(f'{source}: {repo}@{revision} -> {target}', flush=True)


def check_hashes(folder: Path, sources: list[str]) -> dict:
    """sha256 весов против `SHA256` и описание файлов для манифеста."""
    manifest = {}
    for source in sources:
        weights = folder / source / 'model.safetensors'
        if not weights.is_file():
            raise SystemExit(f'Нет {weights}')
        files = {p.name: {'size': p.stat().st_size, 'sha256': sha256(p)}
                 for p in sorted((folder / source).iterdir()) if p.is_file()}
        if files['model.safetensors']['sha256'] != SHA256[source]:
            raise SystemExit(f"{weights}: sha256 {files['model.safetensors']['sha256']}, ожидается {SHA256[source]}")
        repo, revision = MODELS[source]
        manifest[source] = {'repo': repo, 'revision': revision, 'license': LICENSE[source],
                            'url': f'https://huggingface.co/{repo}/tree/{revision}', 'files': files}
        print(f'{source}: sha256 весов совпадает ({len(files)} файлов)', flush=True)
    return manifest


def load_offline(folder: Path, sources: list[str]) -> None:
    """Загрузить модели кодом `run_ov.py` без сети (вызывается в дочернем процессе с OFFLINE)."""
    import run_ov
    for source in sources:
        prompt = (run_ov.PROMPTS / run_ov.PROMPT_FILE[source].format(PROMPT_VERSION)).read_text(encoding='utf-8')
        model = (run_ov.GroundingDino if source == 'ov_gdino' else run_ov.Owl)(folder / source, prompt, 'cpu')
        classes = [run_ov.concept_class(q) for q in model.queries]
        print(f'{source}: загружен без сети, понятий промпта {len(classes)}', flush=True)
        del model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path, help='скачать в каталог, проверить и записать манифест')
    group.add_argument('--check', type=Path, help='только проверить готовый каталог (без сети)')
    parser.add_argument('--sources', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--_load', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.out or args.check

    if args._load:
        load_offline(folder, args.sources)
        return
    if args.out:
        folder.mkdir(parents=True, exist_ok=True)
        download(folder, args.sources)
    manifest = check_hashes(folder, args.sources)
    env = {**os.environ, **OFFLINE}
    command = [sys.executable, str(Path(__file__).resolve()), '--check', str(folder), '--sources', *args.sources, '--_load']
    if subprocess.run(command, env=env).returncode:
        raise SystemExit('Загрузка без сети не прошла: в каталоге не хватает файлов')
    if args.out:
        (folder / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        lines = ['# ts-models-ov', '', 'Веса трека C для работы без интернета (`tracks/ov/pack_weights.py`).', '',
                 '| каталог | модель | ревизия | лицензия | sha256 model.safetensors |', '|---|---|---|---|---|']
        lines += [f"| `{s}/` | {m['repo']} | `{m['revision']}` | {m['license']} | `{m['files']['model.safetensors']['sha256']}` |"
                  for s, m in manifest.items()]
        (folder / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'манифест -> {folder / "MANIFEST.json"}')
    print('OK: веса полные и загружаются без сети')


if __name__ == '__main__':
    main()
