"""C2: zero-shot детекция посуды и вещей на узких кропах столов (раздел 13 контракта).

Источники: `ov_gdino` (Grounding DINO base) и `ov_owl` (OWLv2 large ensemble).
Веса скачивает `tracks/ov/fetch_weights.py`, здесь сеть не используется.

Запуск:
  python tracks/ov/run_ov.py --source ov_gdino --rows targets            # целевые кадры (dt = 0) -> ov_gdino__v1
  python tracks/ov/run_ov.py --source ov_owl --rows targets --batch 2 --fp16   # то же для OWLv2 на GPU
  python tracks/ov/run_ov.py --source ov_gdino                           # все 1718 кадров, с соседями
  python tracks/ov/run_ov.py --source ov_gdino --rows context --version 20     # только соседи (dt = ±30) к готовому
                                                                         # прогону targets, затем merge_dets.py
  python tracks/ov/run_ov.py --source ov_gdino --rows train0 --version 91 --prompt-version 1   # отладка на train
  python tracks/ov/run_ov.py --source ov_owl --smoke 8                   # проверка на 8 строках, файлы удаляются

Каждый кадр-источник режется на кропы `top` ×4 по рамкам столов из rows_ext. Кадр бывает целевым и соседом
других целевых, рамки одного стола у них почти совпадают, поэтому на каждый стол кадра берётся один кроп
(`unique_crops`). Привязка к строкам не страдает: `ts.assign` работает по координатам кадра.
У каждого бокса один класс словаря 8.2: лучшее понятие промпта, включая негативы. NMS по классу внутри кропа.
Пишет ART/dets/<source>__v<N>.parquet (формат 8.1) и карточку. Прогон возобновляемый: порции лежат в
ART/dets/<source>__v<N>.parts/, повторный запуск той же командой продолжает с места остановки.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from _bootstrap import git_sha  # noqa: E402  (добавляет корень репозитория в sys.path)
from fetch_weights import MODELS, model_dir  # noqa: E402

from ts import FRAME_H, FRAME_W, SEED, paths  # noqa: E402
from ts.check import check_dets  # noqa: E402
from ts.crops import crop_table, to_frame_coords  # noqa: E402
from ts.io import DET_COLUMNS  # noqa: E402
from ts.rows import load_rows_ext  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / 'prompts'
PROMPT_FILE = {'ov_gdino': 'gdino_v{}.txt', 'ov_owl': 'owl_v{}.txt'}   # версия промпта: --prompt-version
MIN_CONF = {'ov_gdino': 0.10, 'ov_owl': 0.05}   # низкий порог записи (8.1), рабочий порог выбирает стекер
CROP_KIND, CROP_SCALE = 'top', 4.0
NMS_IOU = 0.5
MAX_TABLE_FRAC = 0.6   # бокс не класса table крупнее этой доли площади рамки стола это сам стол, а не предмет
SAME_CROP_IOU = 0.8    # рамки одного кадра с таким IoU считаются одним столом и дают один кроп
SMOKE_VERSION = 999
BOX = ['x1', 'y1', 'x2', 'y2']

# Понятие промпта -> класс словаря 8.2. Запрос сопоставляется с самым длинным понятием, которым он заканчивается.
CONCEPT_CLASS = {
    'plate': 'plate', 'bowl': 'bowl', 'tray': 'tray',
    'cup': 'cup', 'mug': 'cup', 'glass': 'cup', 'takeaway cup': 'cup',
    'dish': 'plate', 'saucer': 'plate',
    'smartphone': 'phone', 'phone': 'phone', 'handbag': 'bag', 'backpack': 'bag', 'wallet': 'wallet',
    'earphones case': 'headphones', 'sheet of paper': 'paper', 'notebook': 'paper', 'book': 'paper',
    'keys': 'personal_item', 'sunglasses': 'personal_item',
    'laptop': 'laptop', 'tablet': 'tablet', 'bottle': 'bottle', 'can': 'can',
    'vase with dried flowers': 'vase', 'napkin': 'napkin', 'food': 'food', 'table': 'table',
}


def concept_class(query: str) -> str:
    """Класс словаря 8.2 для запроса вида `plate` или `a photo of a plate`."""
    for concept in sorted(CONCEPT_CLASS, key=len, reverse=True):
        if query == concept or query.endswith(' ' + concept):
            return CONCEPT_CLASS[concept]
    raise ValueError(f"Запрос '{query}' не оканчивается понятием из CONCEPT_CLASS")


def code_sha() -> str:
    """Коммит кода для карточки: `TS_GIT_SHA` (Kaggle, архив без .git) или текущий коммит репозитория."""
    return os.environ.get('TS_GIT_SHA') or git_sha()[0]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU боксов `a (n, 4)` и `b (m, 4)` в формате `x1, y1, x2, y2`."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return w * h / np.maximum(area_a[:, None] + area_b[None, :] - w * h, 1e-9)


def greedy_keep(scores: np.ndarray, same: np.ndarray) -> np.ndarray:
    """Жадный отбор по убыванию `scores`: элемент остаётся, если не совпадает (`same`) ни с одним оставленным."""
    keep = []
    for i in np.argsort(-scores, kind='stable'):
        if not any(same[i, j] for j in keep):
            keep.append(int(i))
    return np.array(keep, dtype=int)


class GroundingDino:
    """Grounding DINO: один текстовый промпт `a . b . c .`, оценка понятия = среднее по его токенам."""

    def __init__(self, folder: Path, prompt: str, device: str):
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
        self.processor = GroundingDinoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(folder, local_files_only=True).to(device).eval()
        self.device, self.text = device, prompt.strip()
        self.queries = [q.strip() for q in self.text.split('.') if q.strip()]
        encoded = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        self.input_ids = encoded['input_ids']
        self.tokens, start = [], 0
        for query in self.queries:   # позиции токенов каждого понятия в промпте
            begin = self.text.index(query, start)
            start = begin + len(query)
            ids = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and a >= begin and b <= start]
            if not ids:
                raise ValueError(f"Понятие '{query}' не найдено среди токенов промпта")
            self.tokens.append(ids)

    def __call__(self, images: list) -> list[tuple[np.ndarray, np.ndarray]]:
        import torch
        results = []
        for image in images:   # по одному кропу: размеры разные, без паддинга боксы однозначны
            inputs = self.processor(images=image, text=self.text, return_tensors='pt').to(self.device)
            if inputs['input_ids'][0].tolist() != self.input_ids:
                raise RuntimeError('Процессор изменил текст промпта, позиции токенов понятий неверны')
            output = self.model(**inputs)
            probs = output.logits[0].float().sigmoid()
            scores = torch.stack([probs[:, ids].mean(dim=1) for ids in self.tokens], dim=1)
            results.append((_corners(output.pred_boxes[0], image.width, image.height), scores.cpu().numpy()))
        return results


class Owl:
    """OWLv2: по запросу на понятие. Кроп дополняется до квадрата снизу и справа, боксы в долях стороны квадрата."""

    def __init__(self, folder: Path, prompt: str, device: str):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.processor = Owlv2Processor.from_pretrained(folder, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(folder, local_files_only=True).to(device).eval()
        self.device = device
        self.queries = [q.strip() for q in prompt.splitlines() if q.strip()]

    def __call__(self, images: list) -> list[tuple[np.ndarray, np.ndarray]]:
        inputs = self.processor(text=[self.queries] * len(images), images=images, return_tensors='pt').to(self.device)
        output = self.model(**inputs)
        probs = output.logits.float().sigmoid().cpu().numpy()
        return [(_corners(output.pred_boxes[i], max(image.size), max(image.size)), probs[i])
                for i, image in enumerate(images)]


def _corners(cxcywh, width: float, height: float) -> np.ndarray:
    """Нормированные `cx, cy, w, h` (тензор) -> `x1, y1, x2, y2` в пикселях."""
    b = cxcywh.float().cpu().numpy().astype(float)
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], axis=1)
    return xyxy * np.array([width, height, width, height])


def forward(model, images: list, fp16: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Проход модели по кропам. При нехватке памяти GPU пакет обрабатывается по одному кропу."""
    import torch
    amp = torch.autocast('cuda', dtype=torch.float16) if fp16 else contextlib.nullcontext()
    try:
        with torch.inference_mode(), amp:
            return model(images)
    except torch.cuda.OutOfMemoryError:
        if len(images) == 1:
            raise
        torch.cuda.empty_cache()
        print('нехватка памяти GPU: пакет обрабатывается по одному кропу', flush=True)
        return [result for image in images for result in forward(model, [image], fp16)]


def detections(boxes: np.ndarray, scores: np.ndarray, classes: list[str], min_conf: float, meta) -> pd.DataFrame:
    """Сырые боксы кропа и оценки по понятиям -> детекции в пикселях кадра (один класс на бокс, NMS по классу)."""
    names = list(dict.fromkeys(classes))
    by_class = np.stack([scores[:, [i for i, c in enumerate(classes) if c == name]].max(axis=1) for name in names], 1)
    label, conf = by_class.argmax(axis=1), by_class.max(axis=1)
    boxes = np.clip(boxes, 0, [meta.width, meta.height, meta.width, meta.height])
    width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    tx1, ty1, tx2, ty2 = meta.table_box
    too_big = width * height > MAX_TABLE_FRAC * (tx2 - tx1) * meta.sx * (ty2 - ty1) * meta.sy
    keep = (conf >= min_conf) & (width >= 2) & (height >= 2) & ~(too_big & (np.array(names)[label] != 'table'))
    boxes, label, conf = boxes[keep], label[keep], conf[keep]
    kept = []
    for k in np.unique(label):
        index = np.flatnonzero(label == k)
        kept.append(index[greedy_keep(conf[index], iou_matrix(boxes[index], boxes[index]) >= NMS_IOU)])
    kept = np.sort(np.concatenate(kept)) if kept else np.zeros(0, dtype=int)
    frame = np.clip(to_frame_coords(boxes[kept], meta), 0, [FRAME_W, FRAME_H, FRAME_W, FRAME_H])
    out = pd.DataFrame(frame, columns=BOX)
    out.insert(0, 'confidence', np.clip(conf[kept], 0, 1).astype(float))
    out.insert(0, 'class_name', [names[k] for k in label[kept]])
    return out


def unique_crops(rows: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на стол каждого кадра-источника.

    Из строк с одним `src_frame_id` остаётся строка `dt = 0` (собственные рамки кадра), иначе первая по порядку
    rows_ext. Рамка, у которой IoU с уже выбранными ниже `SAME_CROP_IOU`, считается другой областью и остаётся.
    """
    ordered = rows.assign(_neighbour=rows.dt != 0).sort_values(['src_frame_id', '_neighbour'], kind='stable')
    keep = []
    for _, g in ordered.groupby('src_frame_id', sort=False):
        boxes = g[BOX].to_numpy(float)
        same = iou_matrix(boxes, boxes) >= SAME_CROP_IOU
        keep += g.index[greedy_keep(-np.arange(len(g), dtype=float), same)].tolist()
    return rows.loc[sorted(keep)].reset_index(drop=True)


def shard_of(crops: pd.DataFrame, shard: str | None) -> pd.DataFrame:
    """Доля K/N по кадрам-источникам: кадр целиком попадает в одну долю, доли считаются на разных GPU."""
    if not shard:
        return crops
    try:
        k, n = (int(v) for v in shard.split('/'))
    except ValueError:
        raise SystemExit(f'--shard: ожидается K/N, получено {shard!r}')
    if not (n > 0 and 0 <= k < n):
        raise SystemExit(f'--shard: нужно 0 <= K < N, получено {shard!r}')
    frames = sorted(crops.src_frame_id.unique())
    return crops[crops.src_frame_id.isin(frames[k::n])].reset_index(drop=True)


def coverage_of(which: str, limit: int | None, shard: str | None = None) -> str:
    """Значение `coverage` карточки (раздел 8.1): все кадры, все целевые кадры или отладочная часть."""
    if limit is not None or shard:
        return 'train_only'
    return {'all': 'all', 'targets': 'targets'}.get(which, 'train_only')


def select_rows(which: str, limit: int | None, shard: str | None = None) -> pd.DataFrame:
    rows = load_rows_ext()
    if which == 'context':   # кропы прогона all, которых нет в прогоне targets: соседние кадры (dt = ±30)
        every, done = unique_crops(rows.reset_index(drop=True)), unique_crops(rows[rows.dt == 0].reset_index(drop=True))
        seen = set(zip(done.row_id, done.dt))
        crops = every[[key not in seen for key in zip(every.row_id, every.dt)]].reset_index(drop=True)
        return shard_of(crops, shard).head(limit)
    if which in ('train', 'train0'):
        rows = rows[rows.split == 'train']
    if which in ('targets', 'train0'):
        rows = rows[rows.dt == 0]
    if which == 'neighbours':
        # Только соседние кадры (dt = ±30), которые сами не целевые: для целевых кадров детекции уже есть
        # в прогоне --rows targets, и привязка ts.assign берёт их по src_frame_id.
        target_frames = set(rows.loc[rows.dt == 0, 'frame_id'])
        rows = rows[(rows.dt != 0) & ~rows.src_frame_id.isin(target_frames)]
    return shard_of(unique_crops(rows.reset_index(drop=True)), shard).head(limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, choices=list(MODELS))
    parser.add_argument('--version', type=int, default=1, help='суффикс артефакта: <source>__v<N>')
    parser.add_argument('--rows', default='all', choices=['all', 'targets', 'context', 'neighbours', 'train', 'train0'],
                        help='all: все 1718 кадров; targets: только целевые кадры (dt = 0, 4964 строки); '
                             'context: кропы all, которых нет в targets (соседи dt = ±30, 5234 кропа), '
                             'склейка с targets: merge_dets.py; '
                             'train: train-дни с соседями; train0: только целевые train-кадры')
    parser.add_argument('--prompt-version', type=int, default=1, help='файл промпта tracks/ov/prompts/*_v<P>.txt')
    parser.add_argument('--device', help='cuda, cuda:1 или cpu; по умолчанию cuda, если есть')
    parser.add_argument('--batch', type=int, default=1, help='кропов за один проход (для ov_owl)')
    parser.add_argument('--fp16', action='store_true', help='половинная точность на GPU (быстрее, меньше памяти)')
    parser.add_argument('--chunk', type=int, default=120, help='строк в одной порции прогресса')
    parser.add_argument('--max-minutes', type=float, help='штатно остановиться после порции, если прошло столько минут')
    parser.add_argument('--limit', type=int, help='первые N строк, только для отладки')
    parser.add_argument('--shard', metavar='K/N', help='доля K из N по кадрам-источникам (K от 0), для двух GPU')
    parser.add_argument('--smoke', type=int, metavar='N',
                        help='проверка на N train-строках: формат, скорость, память GPU; файлы удаляются')
    args = parser.parse_args()
    if args.smoke:
        args.version, args.rows, args.limit, args.max_minutes = SMOKE_VERSION, 'train0', args.smoke, None

    name = f'{args.source}__v{args.version}'
    out_path = paths.art('dets', f'{name}.parquet')
    parts = out_path.with_suffix('.parts')
    if args.smoke:
        for old in (out_path, out_path.with_suffix('.json')):
            old.unlink(missing_ok=True)
        shutil.rmtree(parts, ignore_errors=True)
    if out_path.is_file():
        print(f'{name}: уже готов ({out_path}), пропуск. Артефакты неизменяемы: для пересчёта укажите новый --version.')
        return

    import torch
    import transformers
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fp16 = args.fp16 and device.startswith('cuda')

    rows = select_rows(args.rows, args.limit, args.shard)
    prompt_file = PROMPT_FILE[args.source].format(args.prompt_version)
    prompt = (PROMPTS / prompt_file).read_text(encoding='utf-8')
    repo, revision = MODELS[args.source]
    state = {
        'source': args.source, 'rows': args.rows, 'shard': args.shard, 'limit': args.limit, 'n_rows': len(rows),
        'rows_sha1': hashlib.sha1('\n'.join(rows.row_id + ':' + rows.dt.astype(str)).encode()).hexdigest(),
        'model': f'{repo}@{revision}', 'prompt_sha1': hashlib.sha1(prompt.encode()).hexdigest(),
        'crop': [CROP_KIND, CROP_SCALE], 'min_conf': MIN_CONF[args.source], 'nms_iou': NMS_IOU,
        'max_table_frac': MAX_TABLE_FRAC, 'same_crop_iou': SAME_CROP_IOU, 'fp16': fp16, 'chunk': args.chunk,
    }
    parts.mkdir(exist_ok=True)
    state_path = parts / 'state.json'
    if state_path.is_file() and json.loads(state_path.read_text(encoding='utf-8')) != state:
        raise SystemExit(f'{parts} начат с другими параметрами. Удалите каталог или укажите новый --version.')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    starts = range(0, len(rows), args.chunk)
    todo = [k for k in starts if not (parts / f'{k:06d}.parquet').is_file()]
    print(f'{name} [{device}{", fp16" if fp16 else ""}]: кропов {len(rows)}, порций осталось {len(todo)} из {len(starts)}',
          flush=True)
    if todo:
        model_class = GroundingDino if args.source == 'ov_gdino' else Owl
        model = model_class(model_dir(args.source), prompt, device)
        classes = [concept_class(q) for q in model.queries]
        started, done, total = time.time(), 0, sum(len(rows.iloc[k:k + args.chunk]) for k in todo)
        for k in todo:
            chunk = rows.iloc[k:k + args.chunk]
            found = []
            for b in range(0, len(chunk), args.batch):
                batch = chunk.iloc[b:b + args.batch]
                crops = [crop_table(r.src_frame_id, (r.x1, r.y1, r.x2, r.y2), kind=CROP_KIND, scale=CROP_SCALE,
                                    slot=r.slot) for r in batch.itertuples()]
                raw = forward(model, [crop for crop, _ in crops], fp16)
                for r, (_, meta), (boxes, scores) in zip(batch.itertuples(), crops, raw):
                    dets = detections(boxes, scores, classes, MIN_CONF[args.source], meta)
                    dets.insert(0, 'frame_id', r.src_frame_id)
                    dets['from_row_id'], dets['from_dt'] = r.row_id, int(r.dt)
                    found.append(dets)
            tmp = parts / f'{k:06d}.tmp'
            pd.concat(found, ignore_index=True).to_parquet(tmp, index=False)
            tmp.replace(parts / f'{k:06d}.parquet')
            done += len(chunk)
            minutes = (time.time() - started) / 60
            memory = f', пик памяти GPU {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} ГБ' if device != 'cpu' else ''
            print(f'{name}: кропов {k + len(chunk)}/{len(rows)}, {60 * minutes / done:.2f} с/кроп, '
                  f'осталось ~{minutes / done * (total - done):.0f} мин{memory}', flush=True)
            if args.max_minutes is not None and minutes >= args.max_minutes and done < total:
                print(f'{name}: лимит {args.max_minutes:g} мин, остановка. Прогресс в {parts}, '
                      'для продолжения повторите ту же команду.', flush=True)
                return

    dets = pd.concat([pd.read_parquet(parts / f'{k:06d}.parquet') for k in starts], ignore_index=True)
    dets = dets[(dets.x2 > dets.x1) & (dets.y2 > dets.y1)].reset_index(drop=True)
    dets['source'] = args.source
    dets['det_id'] = args.source + '_' + dets.frame_id + '_' + dets.groupby('frame_id', sort=False).cumcount().astype(str)
    dets['from_dt'] = dets['from_dt'].astype('Int64')
    dets[DET_COLUMNS].to_parquet(out_path, index=False)

    frame_ids = sorted(rows.src_frame_id.unique())
    dts = sorted(int(v) for v in rows.dt.unique())
    card = {
        'name': name, 'kind': 'dets', 'owner': 'C', 'created_utc': utc_now(), 'git_sha': code_sha(),
        'command': 'python tracks/ov/run_ov.py ' + ' '.join(sys.argv[1:]),
        'models': [state['model']],
        'inputs': ['rows_ext.parquet'],
        'coverage': coverage_of(args.rows, args.limit, args.shard),
        'rows': (args.rows + (f', доля {args.shard}' if args.shard else '')) if args.limit is None else f'{args.rows}, первые {args.limit}',
        'dt': dts, 'context_features': coverage_of(args.rows, args.limit, args.shard) == 'all',   # prev/next/med: только all
        'frames_processed': {'n': len(frame_ids), 'sha1': hashlib.sha1('\n'.join(frame_ids).encode()).hexdigest()},
        'crops_processed': len(rows),
        'manual_labels_used': False,
        'uses_train_counters': False,
        'prompt': f'tracks/ov/prompts/{prompt_file}', 'prompt_sha1': state['prompt_sha1'],
        'crop': {'kind': CROP_KIND, 'scale': CROP_SCALE}, 'min_conf': state['min_conf'], 'nms_iou': NMS_IOU,
        'max_table_frac': MAX_TABLE_FRAC, 'same_crop_iou': SAME_CROP_IOU, 'fp16': fp16,
        'device': torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'notes': f'строки: {args.rows}; '
                 + ('посчитан только для dt = 0, контекстные признаки prev/next/med недоступны; ' if dts == [0] else '')
                 + ('часть прогона all без кропов targets, склейка: merge_dets.py; ' if args.rows == 'context' else '')
                 + 'один кроп на стол каждого кадра-источника (from_row_id это строка кропа); '
                 'один класс на бокс (лучшее понятие промпта, включая негативы и table); '
                 f'боксы не класса table крупнее {MAX_TABLE_FRAC} площади рамки стола отброшены; '
                 'объект в кропах двух столов записан дважды, дедупликация в ts.assign',
    }
    out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_dets(out_path)
    shutil.rmtree(parts)
    print(f'{name}: {len(dets)} детекций, {len(rows)} кропов, {len(frame_ids)} кадров -> {out_path}')
    print(dets.class_name.value_counts().to_string())
    if args.smoke:
        out_path.unlink()
        out_path.with_suffix('.json').unlink()
        print(f'{name}: проверка пройдена (формат 8.1), файлы проверки удалены')


if __name__ == '__main__':
    main()
