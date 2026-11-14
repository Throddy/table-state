"""VLM-счётчик по кропу строки: Qwen-VL считает people, hands, cups, dishes, items для целевого стола.

Запуск (одна ячейка Kaggle на шард, по процессу на GPU):
  python tracks/det/vlm/run_vlm.py --model <каталог весов> --subset all --shard 0 --of 2 --device cuda:0
  python tracks/det/vlm/run_vlm.py --model <каталог весов> --subset all --shard 1 --of 2 --device cuda:1
  python tracks/det/vlm/run_vlm.py --subset all --of 2 --merge          # сборка rowfeats после всех шардов
Отладка и проверка на train: --subset train (231 строка, файл с суффиксом __train, без ts.check).

Вход модели: широкий кроп строки (ts.crops, kind='wide', ×2.5) с тонкой красной рамкой целевого стола,
длинная сторона приведена к 1024. Промпт: tracks/det/vlm/prompts/<--prompt-id>.txt. Генерация без сэмплирования,
не больше 128 новых токенов. Ответ, который не разобрался как JSON с пятью целыми, даёт −1 и __ok = 0.
Возобновление: каждый ответ сразу дописывается в ART/rowfeats/_parts/<имя>/shard_<i>_of_<n>.jsonl,
повторный запуск пропускает готовые строки. Сеть не используется, веса читаются из каталога.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from ts import COUNTS, paths  # noqa: E402
from ts.check import check_rowfeats  # noqa: E402
from ts.crops import crop_table, to_crop_coords  # noqa: E402
from ts.rows import load_rows_ext  # noqa: E402

SOURCE = 'vlm_qwen'
PROMPTS = Path(__file__).resolve().parent / 'prompts'
# vlm_v1: образец ответа с нулями (на нём считался шаг 1); vlm_v2: образец с буквой N вместо чисел.
LONG_SIDE = 1024
WIDE_SCALE = 2.5
MAX_NEW_TOKENS = 128


def row_image(row) -> Image.Image:
    """Широкий кроп строки с красной рамкой целевого стола, длинная сторона 1024."""
    box = (row.x1, row.y1, row.x2, row.y2)
    crop, meta = crop_table(row.src_frame_id, box, kind='wide', scale=WIDE_SCALE, slot=int(row.slot))
    ImageDraw.Draw(crop).rectangle(to_crop_coords([box], meta)[0].tolist(), outline=(255, 0, 0), width=3)
    k = LONG_SIDE / max(crop.size)
    return crop.resize((round(crop.width * k), round(crop.height * k)), Image.Resampling.LANCZOS)


def parse(text: str) -> dict | None:
    """Пять неотрицательных целых из ответа модели или None."""
    for chunk in re.findall(r'\{[^{}]*\}', text):
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if all(isinstance(data.get(c), int) and not isinstance(data.get(c), bool) and 0 <= data[c] <= 50 for c in COUNTS):
            return {c: int(data[c]) for c in COUNTS}
    return None


def work_rows(subset: str) -> pd.DataFrame:
    rows = load_rows_ext()
    rows = rows[rows['dt'] == 0]
    if subset == 'train':
        rows = rows[rows['split'] == 'train']
    return rows.sort_values('row_id', kind='stable').reset_index(drop=True)


def parts_dir(name: str) -> Path:
    folder = paths.art('rowfeats', '_parts') / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_done(folder: Path) -> dict[str, dict]:
    done = {}
    for file in sorted(folder.glob('shard_*.jsonl')):
        for line in file.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                done[record['row_id']] = record
    return done


def run(args, name: str) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not (args.model / 'config.json').is_file():
        raise SystemExit(f'В {args.model} нет config.json. Веса кладутся заранее, автоскачивание запрещено.')
    rows = work_rows(args.subset)
    mine = rows[rows.index % args.of == args.shard]
    folder = parts_dir(name)
    done = read_done(folder)
    todo = mine[~mine['row_id'].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f'[{name}] шард {args.shard}/{args.of}: строк {len(mine)}, осталось {len(todo)}, dtype {args.dtype}, батч {args.batch}', flush=True)
    if todo.empty:
        return
    torch.manual_seed(common.SEED)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(args.model), local_files_only=True,
                                                        dtype=getattr(torch, args.dtype), device_map={'': args.device})
    model.eval()
    prompt = (PROMPTS / f'{args.prompt_id}.txt').read_text(encoding='utf-8').strip()
    out = folder / f'shard_{args.shard}_of_{args.of}.jsonl'
    started = time.time()
    processor.tokenizer.padding_side = 'left'        # для батча: ответ дописывается справа от промпта
    records = list(todo.itertuples(index=False))
    done_rows = 0
    with open(out, 'a', encoding='utf-8') as sink:
        for start in range(0, len(records), args.batch):
            batch = records[start:start + args.batch]
            conversations = [[{'role': 'user', 'content': [{'type': 'image', 'image': row_image(row)},
                                                            {'type': 'text', 'text': prompt}]}] for row in batch]
            inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                   return_tensors='pt', padding=True).to(model.device)
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            texts = processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch, texts):
                sink.write(json.dumps({'row_id': row.row_id, 'dt': 0, 'prompt_id': args.prompt_id, 'text': text}, ensure_ascii=False) + '\n')
            sink.flush()
            done_rows += len(batch)
            if done_rows % 24 < args.batch or done_rows == len(records):
                rate = (time.time() - started) / done_rows
                print(f'[{name}] шард {args.shard}: {done_rows}/{len(records)}, {rate:.2f} с на строку, осталось около '
                      f'{(len(records) - done_rows) * rate / 60:.1f} мин', flush=True)
    print(f'[{name}] шард {args.shard}: время этого запуска {(time.time() - started) / 60:.1f} мин (для учёта квоты GPU)')


def merge(args, name: str) -> None:
    rows = work_rows(args.subset)
    done = read_done(parts_dir(name))
    missing = sorted(set(rows['row_id']) - set(done))
    if missing:
        raise SystemExit(f'Не хватает ответов для {len(missing)} строк из {len(rows)}. Досчитайте шарды.')
    records, raw = [], []
    for row_id in rows['row_id']:
        answer = done[row_id]
        counts = parse(answer['text'])
        records.append({'row_id': row_id, 'dt': 0,
                        **{f'{SOURCE}__{c}': float(counts[c]) if counts else -1.0 for c in COUNTS},
                        f'{SOURCE}__ok': int(counts is not None)})
        raw.append({'row_id': row_id, 'dt': 0, 'prompt_id': answer['prompt_id'], 'text': answer['text']})
    table = pd.DataFrame(records)
    path = paths.art('rowfeats', f'{name}.parquet')
    if path.exists():
        raise SystemExit(f'{path} уже существует. Артефакты неизменяемы: поднимите --version.')
    table.to_parquet(path, index=False)
    path.with_name(f'{name}__raw.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in raw), encoding='utf-8')
    card = {'name': name, 'kind': 'rowfeats', 'owner': common.OWNER, 'created_utc': common.utc_now(), 'git_sha': common.git_sha(),
            'command': common.command_line(), 'models': [args.model_ref], 'inputs': ['rows_ext.parquet'],
            'coverage': 'all' if args.subset == 'all' else 'train_only', 'rows': len(table),
            'not_ok_share': round(float((table[f'{SOURCE}__ok'] == 0).mean()), 4),
            'manual_labels_used': False, 'uses_train_counters': False,
            'params': {'prompt': sorted({f'tracks/det/vlm/prompts/{r["prompt_id"]}.txt' for r in raw}), 'crop': 'wide', 'scale': WIDE_SCALE, 'long_side': LONG_SIDE,
                       'do_sample': False, 'max_new_tokens': args.max_new_tokens, 'batch': args.batch, 'dtype': args.dtype, 'seed': common.SEED},
            'notes': 'счётчики VLM по широкому кропу с рамкой целевого стола; −1 и __ok = 0, если ответ не разобран'}
    path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.subset == 'all':
        check_rowfeats(path)
        print('ts.check rowfeats: OK')
    print(f'{name}: {len(table)} строк, доля __ok = 0: {card["not_ok_share"]}, файл {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, help='каталог весов (config.json, safetensors, токенизатор)')
    parser.add_argument('--model-ref', default='Qwen/Qwen3-VL-4B-Instruct', help='ссылка на веса для карточки')
    parser.add_argument('--subset', choices=['all', 'train'], default='all')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--of', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16', help='float32 только для отладки на CPU')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--prompt-id', default='vlm_v1', help='имя файла промпта в tracks/det/vlm/prompts без расширения')
    parser.add_argument('--batch', type=int, default=1, help='строк в одном вызове generate')
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    parser.add_argument('--limit', type=int, default=0, help='отладка: не больше N строк за запуск')
    parser.add_argument('--merge', action='store_true', help='собрать rowfeats из готовых шардов')
    args = parser.parse_args()
    common.require_data()
    name = f'{SOURCE}__v{args.version}' + ('' if args.subset == 'all' else f'__{args.subset}')
    if args.merge:
        merge(args, name)
    else:
        if args.model is None:
            parser.error('нужен --model')
        run(args, name)


if __name__ == '__main__':
    main()
