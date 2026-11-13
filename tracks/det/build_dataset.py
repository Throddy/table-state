"""Датасет своего детектора в формате YOLO (задачи B2, B3 «псевдо» и B7). Ручной разметки нет.

Источники боксов:
  1. aux как есть (aux_boxes.csv, маппинг 8.2);
  2. прод-детекции с confidence >= 0.6 (person, hand, cup, tray) на кадрах выбранных дней;
  3. уверенные детекции трека C из задачи C5 (tray, phone, plate, что есть в файле): ART/labels/ov_pseudo__v<N>.parquet
     или --ov-pseudo;
  4. только для v2 (B7): уверенные предсказания своего детектора v1 (--ft-pseudo ft_yolo__v1), confidence >= 0.6,
     для cup >= 0.8.
Других источников нет. Боксы одного объекта из разных источников сливаются по правилу 9.2.6 контракта.

Для каждого кадра делаются три вида: полный кадр, кропы зон top (×4) и wide (×2.5) каждого стола с
пересчитанными боксами. Столы на aux это боксы table из aux, на остальных кадрах детекции table прод-детектора.
Области неуверенных детекций (прод ниже 0.6, свой детектор 0.2–0.6) закрашиваются серым: Ultralytics не умеет
области игнора. 6 aux-кадров с тарелками и телефонами откладываются в holdout и в обучение не попадают
(выбор не зависит от набора классов, поэтому holdout у v1 и v2 один и тот же).

Запуск:
  v1: python tracks/det/build_dataset.py --name ft_data_v1 --classes v1 --pseudo-step 2
  v2: python tracks/det/build_dataset.py --name ft_data_v2 --classes auto --pseudo-step 2 --ft-pseudo ft_yolo__v1
      (набор классов v2: классы v1 и те из phone, plate, для которых в файле трека C есть боксы)
По умолчанию псевдометки берутся только с кадров 27–28.10: кадры 25–26.10 в обучение не идут,
поэтому правило 12.4 не применяется. Результат: ART/datasets/<name>/ (images, labels, data.yaml,
manifest.csv, card.json, preview.jpg).
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import common
from common import CLASS_SETS, OV_EXTRA, TEST_DATES

from ts import FRAME_H, FRAME_W, SEED, io, paths
from ts.crops import crop_region, to_crop_coords
from ts.geometry import clip_box, zone
from ts.slots import slot_of

VIEWS = {'full': None, 'top': 4.0, 'wide': 2.5}     # вид -> увеличение кропа (раздел 9.1)
PROD_CLASSES = ['person', 'hand', 'cup', 'tray']     # что берём из прод-детектора
OV_CLASSES = ['tray', *OV_EXTRA]                     # что берём из псевдометок трека C (задача C5), если это есть в файле
PSEUDO_CONF = 0.6                                    # B3: уверенные детекции идут в метки
UNSURE_MIN = 0.2                                     # B3: 0.2–0.6 закрашивается, ниже 0.2 не учитывается
# Порог для предсказаний своего детектора по классам (самообучение, B7). Для cup порог выше (решение роли A):
# v1 находит стаканы на пустых столах, которых нет в разметке, размножать их в метках v2 нельзя.
# Значение 0.8 выбрано заранее, по кадрам 25–26.10 не подбиралось. Боксы от 0.2 до порога закрашиваются.
FT_PSEUDO_CONF = {'cup': 0.8}
DEDUP_IOU, DEDUP_SMALL_PX, DEDUP_CENTER = 0.5, 40.0, 0.5   # правило 9.2.6: один и тот же объект
MIN_VISIBLE = 0.5                                    # бокс остаётся в кропе, если видно не меньше половины площади
MIN_SIDE_PX = 2.0                                    # и после обрезки он не тоньше 2 px кропа
MAX_PAINTED = 0.5                                    # кроп, закрашенный больше чем наполовину, не используется
N_HOLDOUT = 6
NEAR_SECONDS = 30                                    # кадры той же сцены не делятся между обучением и holdout
PAINT = (114, 114, 114)
JPEG_QUALITY = 95
BOX = ['x1', 'y1', 'x2', 'y2']
LABEL_COLUMNS = ['frame_id', 'class_name', 'confidence', *BOX, 'src']


# ---------- геометрия ----------

def view_region(view: str, table) -> tuple[int, int, int, int]:
    """Область кадра для вида, в целых пикселях, как её вырежет `ts.crops.crop_region`."""
    if view == 'full':
        return 0, 0, FRAME_W, FRAME_H
    x0, y0, x1, y1 = clip_box(zone(table, view, slot_of((table[0] + table[2]) / 2)))
    return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)


def visible(boxes: np.ndarray, region) -> np.ndarray:
    """Доля площади каждого бокса внутри области."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    iw = (np.minimum(b[:, 2], region[2]) - np.maximum(b[:, 0], region[0])).clip(min=0)
    ih = (np.minimum(b[:, 3], region[3]) - np.maximum(b[:, 1], region[1])).clip(min=0)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.where(area > 0, iw * ih / np.maximum(area, 1e-9), 0.0)


def paint_mask(uncertain: np.ndarray, confident: np.ndarray) -> np.ndarray:
    """Маска закраски кадра: объединение неуверенных боксов минус уверенные (их пиксели не трогаем)."""
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    for x1, y1, x2, y2 in np.asarray(uncertain, dtype=float).reshape(-1, 4):
        mask[max(0, math.floor(y1)):min(FRAME_H, math.ceil(y2)), max(0, math.floor(x1)):min(FRAME_W, math.ceil(x2))] = True
    for x1, y1, x2, y2 in np.asarray(confident, dtype=float).reshape(-1, 4):
        mask[max(0, math.floor(y1)):min(FRAME_H, math.ceil(y2)), max(0, math.floor(x1)):min(FRAME_W, math.ceil(x2))] = False
    return mask


def same_object(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Правило 9.2.6: IoU >= 0.5, а для боксов меньше 40 px расстояние между центрами меньше половины меньшей стороны."""
    others = np.asarray(others, dtype=float).reshape(-1, 4)
    if not len(others):
        return np.zeros(0, dtype=bool)
    small = np.minimum(min(box[2] - box[0], box[3] - box[1]),
                       np.minimum(others[:, 2] - others[:, 0], others[:, 3] - others[:, 1]))
    dist = np.hypot((box[0] + box[2] - others[:, 0] - others[:, 2]) / 2, (box[1] + box[3] - others[:, 1] - others[:, 3]) / 2)
    return np.where(small < DEDUP_SMALL_PX, dist < DEDUP_CENTER * small,
                    common.iou_one_to_many(np.asarray(box, dtype=float), others) >= DEDUP_IOU)


def merge_frame_labels(sure: pd.DataFrame, unsure: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Метки и области закраски одного кадра.

    Уверенные боксы одного класса, попавшие на один объект, сливаются: остаётся самый уверенный.
    Сливаются боксы разных источников, а у своего детектора (`ft`) ещё и между собой: один объект
    виден в кропах соседних столов и соседних кадров. Боксы прод-детектора между собой и боксы
    трека C между собой не сливаются: внутри источника это разные объекты.
    Неуверенный бокс, совпавший по тому же правилу с уверенным того же класса, не закрашивается:
    объект уже размечен.
    """
    def duplicates(box, src, boxes, sources) -> np.ndarray:
        merge_allowed = (sources != src) | (src == 'ft')
        return same_object(box, boxes) & merge_allowed

    kept: list = []
    for _, group in sure.groupby('class_name', sort=True):
        group = group.sort_values('confidence', ascending=False, kind='stable')
        boxes, sources = group[BOX].to_numpy(float), group['src'].to_numpy()
        chosen: list[int] = []
        for i in range(len(group)):
            if not duplicates(boxes[i], sources[i], boxes[chosen], sources[chosen]).any():
                chosen.append(i)
        kept.extend(group.index[chosen].tolist())
    labels = sure.loc[sorted(kept)]      # исходный порядок боксов сохраняется
    paint = []
    for row in unsure.itertuples(index=False):
        box = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
        same_class = labels[labels['class_name'] == row.class_name]
        if not duplicates(box, row.src, same_class[BOX].to_numpy(float), same_class['src'].to_numpy()).any():
            paint.append(box)
    return labels, (np.array(paint) if paint else np.zeros((0, 4)))


# ---------- источники боксов ----------

def aux_boxes(classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Боксы aux как есть: (объекты классов детектора, столы, все боксы для выбора holdout)."""
    aux = io.load_aux_boxes()
    return aux[aux['class_name'].isin(classes)], aux[aux['class_name'] == 'table'], aux


def pick_holdout(objects: pd.DataFrame, aux_frames: pd.DataFrame) -> tuple[list[str], list[str]]:
    """6 aux-кадров holdout и кадры той же сцены, которые нельзя оставлять в обучении.

    Сначала кадры, где есть и тарелки, и телефоны; если их меньше шести, добираются кадры с тарелками,
    затем с телефонами. Внутри группы порядок случайный с сидом 419. Кадр, отстоящий от уже выбранного
    не больше чем на 30 с, в holdout не берётся, а из обучения убирается (та же сцена, аналог правила R4).
    """
    counts = objects.pivot_table(index='frame_id', columns='class_name', values='det_id', aggfunc='size', fill_value=0)
    counts = counts.reindex(index=sorted(aux_frames['frame_id']), columns=['plate', 'phone'], fill_value=0)
    ts = pd.to_datetime(aux_frames.set_index('frame_id')['timestamp'], utc=True)
    rng = np.random.default_rng(SEED)
    both = counts.index[(counts['plate'] > 0) & (counts['phone'] > 0)].tolist()
    plate = counts.index[(counts['plate'] > 0) & (counts['phone'] == 0)].tolist()
    phone = counts.index[(counts['plate'] == 0) & (counts['phone'] > 0)].tolist()
    order = [f for group in (both, plate, phone) for f in rng.permutation(sorted(group)).tolist()]

    def near(a: str, b: str) -> bool:
        return abs((ts[a] - ts[b]).total_seconds()) <= NEAR_SECONDS

    holdout: list[str] = []
    for frame_id in order:
        if len(holdout) == N_HOLDOUT:
            break
        if not any(near(frame_id, h) for h in holdout):
            holdout.append(frame_id)
    excluded = [f for f in counts.index if f not in holdout and any(near(f, h) for h in holdout)]
    return sorted(holdout), sorted(excluded)


def dets_file(value: str | Path | None, folder: str, pattern: str) -> Path | None:
    """Файл формата 8.1 по пути или имени. Без значения берётся последний файл по шаблону в `ART/<folder>`."""
    if value is None:
        found = sorted((paths.ART / folder).glob(pattern)) if pattern else []
        return found[-1] if found else None
    path = Path(value)
    if not path.is_file():
        path = paths.ART / folder / f'{value}.parquet'
    if not path.is_file():
        raise SystemExit(f'Нет файла {value} (искал и в {paths.ART / folder}).')
    return path


def read_labels(path: Path | None, src: str, classes: list[str], frames: list[str]) -> pd.DataFrame:
    """Детекции формата 8.1 как кандидаты в метки: нужные классы и кадры, колонка источника `src`."""
    if path is None:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    table = pd.read_parquet(path)
    missing = [c for c in LABEL_COLUMNS[:-1] if c not in table.columns]
    if missing:
        raise SystemExit(f'{path}: нет колонок {missing}, нужен формат 8.1.')
    table = table[table['class_name'].isin(classes) & table['frame_id'].isin(frames)].copy()
    table['src'] = src
    return table[LABEL_COLUMNS]


# ---------- сборка ----------

def frame_jobs(frame_id: str, origin: str, split: str, labels: pd.DataFrame, tables: np.ndarray,
               uncertain: np.ndarray) -> list[dict]:
    """Все виды одного кадра и боксы, которые попадают в каждый."""
    boxes = labels[BOX].to_numpy(float).reshape(-1, 4)
    classes = labels['class_name'].to_numpy()
    mask = paint_mask(uncertain, boxes) if len(uncertain) else None
    jobs = []
    for view, scale in VIEWS.items():
        for i, table in enumerate([None] if view == 'full' else tables):
            region = view_region(view, table)
            if region[2] <= region[0] or region[3] <= region[1]:
                continue
            keep = np.flatnonzero(visible(boxes, region) >= MIN_VISIBLE)
            painted = float(mask[region[1]:region[3], region[0]:region[2]].mean()) if mask is not None else 0.0
            if painted > MAX_PAINTED:
                continue
            name = f'{frame_id}__{view}' + ('' if view == 'full' else f'{i}')
            jobs.append({'name': name, 'frame_id': frame_id, 'origin': origin, 'split': split, 'view': view,
                         'scale': scale or 1.0, 'region': region,
                         'table': None if table is None else tuple(float(v) for v in table),
                         'boxes': boxes[keep], 'classes': classes[keep], 'painted': painted})
    return jobs


def render(frame_id: str, jobs: list[dict], uncertain: np.ndarray, confident: np.ndarray, out: Path,
           classes: list[str]) -> list[dict]:
    """Записать картинки и файлы меток выбранных видов одного кадра. Возвращает строки манифеста."""
    image = io.read_image(frame_id)
    if len(uncertain):
        pixels = np.asarray(image).copy()
        pixels[paint_mask(uncertain, confident)] = PAINT
        image = Image.fromarray(pixels)
    records = []
    for job in jobs:
        if job['view'] == 'full':
            crop = image
            in_crop = job['boxes'].copy()
        else:
            crop, meta = crop_region(image, zone(job['table'], job['view'], slot_of((job['table'][0] + job['table'][2]) / 2)),
                                     job['scale'], frame_id=frame_id, kind=job['view'], table_box=job['table'])
            if (meta.x0, meta.y0, meta.x1, meta.y1) != tuple(job['region']):
                raise RuntimeError(f'{job["name"]}: область кропа {meta} не совпала с расчётной {job["region"]}')
            in_crop = to_crop_coords(job['boxes'], meta)
        width, height = crop.size
        in_crop[:, [0, 2]] = in_crop[:, [0, 2]].clip(0, width)
        in_crop[:, [1, 3]] = in_crop[:, [1, 3]].clip(0, height)
        lines, counts = [], {c: 0 for c in classes}
        for (x1, y1, x2, y2), name in zip(in_crop, job['classes']):
            if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
                continue
            lines.append(f'{classes.index(name)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} '
                         f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}')
            counts[name] += 1
        crop.save(out / 'images' / job['split'] / f'{job["name"]}.jpg', quality=JPEG_QUALITY)
        (out / 'labels' / job['split'] / f'{job["name"]}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''),
                                                                       encoding='utf-8')
        records.append({'image': f'images/{job["split"]}/{job["name"]}.jpg', 'split': job['split'],
                        'origin': job['origin'], 'view': job['view'], 'frame_id': frame_id, 'width': width,
                        'height': height, 'painted': round(job['painted'], 4), 'n_boxes': len(lines),
                        **{f'n_{c}': n for c, n in counts.items()}})
    return records


def preview(out: Path, manifest: pd.DataFrame, classes: list[str], n: int = 10) -> None:
    """Лист из `n` случайных примеров aux (сид 419) с нарисованными боксами: визуальная проверка B2."""
    pool = manifest[(manifest['origin'] == 'aux') & (manifest['n_boxes'] > 0)]
    if pool.empty:
        return
    picks = pool.sample(n=min(n, len(pool)), random_state=SEED).sort_values('image')
    colors = ['#ff4d4d', '#ffd60a', '#00e5ff', '#22c55e', '#c77dff', '#ff9f1c', '#ffffff', '#ff66c4']
    tiles = []
    for r in picks.itertuples():
        image = Image.open(out / r.image).convert('RGB')
        draw = ImageDraw.Draw(image)
        label_file = out / r.image.replace('images/', 'labels/', 1).replace('.jpg', '.txt')
        for line in label_file.read_text(encoding='utf-8').splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * image.width, float(cy) * image.height, float(w) * image.width, float(h) * image.height
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=colors[int(c) % len(colors)], width=2)
            draw.text((cx - w / 2 + 2, max(0, cy - h / 2 - 12)), classes[int(c)], fill=colors[int(c) % len(colors)])
        image.thumbnail((640, 400))
        tile = Image.new('RGB', (640, 420), 'black')
        tile.paste(image, (0, 20))
        ImageDraw.Draw(tile).text((4, 4), f'{r.split} {Path(r.image).stem} ({r.n_boxes} boxes)', fill='white')
        tiles.append(tile)
    sheet = Image.new('RGB', (1280, 420 * math.ceil(len(tiles) / 2)), 'black')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (640 * (i % 2), 420 * (i // 2)))
    sheet.save(out / 'preview.jpg', quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='ft_data_v1', help='каталог ART/datasets/<name>')
    parser.add_argument('--classes', choices=[*sorted(CLASS_SETS), 'auto'], default='v1',
                        help='v1: person, hand, cup, tray; auto (v2): классы v1 и те из phone, plate, что есть в файле трека C')
    parser.add_argument('--pseudo-dates', nargs='*', default=list(TEST_DATES),
                        help='дни UTC, с которых берутся псевдометки (по умолчанию 27–28.10; пусто = только aux)')
    parser.add_argument('--ov-pseudo', default=None,
                        help='файл псевдометок трека C (путь или имя в ART/labels); по умолчанию последний ov_pseudo__v*.parquet')
    parser.add_argument('--ft-pseudo', default=None,
                        help='детекции своего детектора прошлой версии (путь или имя в ART/dets), источник меток для v2')
    parser.add_argument('--bg-frac', type=float, default=0.1,
                        help='сколько псевдо-картинок без боксов оставить, доля от числа картинок с боксами того же вида')
    parser.add_argument('--pseudo-step', type=int, default=1,
                        help='брать каждый N-й кадр эпизода (цепочки с шагом 30 с); 2 = каждый второй')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    common.require_data()
    out = paths.ART / 'datasets' / args.name
    if out.exists():
        raise SystemExit(f'{out} уже существует. Датасет неизменяем: выберите другое --name.')
    frames = io.load_frames()
    pseudo = frames[frames['date_utc'].isin(args.pseudo_dates)].sort_values('ts')
    # эпизод это цепочка кадров с шагом 30 с; соседние кадры почти дубликаты, поэтому можно брать каждый N-й
    episode = (pseudo['ts'].diff().dt.total_seconds().fillna(1e9) > NEAR_SECONDS + 5).cumsum()
    pseudo_ids = sorted(pseudo.loc[pseudo.groupby(episode).cumcount() % args.pseudo_step == 0, 'frame_id'])

    ov_path = dets_file(args.ov_pseudo, 'labels', 'ov_pseudo__v*.parquet') if (args.classes == 'auto' or args.ov_pseudo) else None
    ov_found: dict[str, int] = {}
    if ov_path is not None:
        ov_all = pd.read_parquet(ov_path)
        ov_found = {str(c): int(n) for c, n in ov_all.loc[ov_all['frame_id'].isin(pseudo_ids), 'class_name']
                    .value_counts().sort_index().items()}
    if args.classes == 'auto':
        if ov_path is None:
            raise SystemExit('Для --classes auto нужен файл псевдометок трека C (задача C5): положите '
                             'ART/labels/ov_pseudo__v<N>.parquet или укажите --ov-pseudo.')
        classes = CLASS_SETS['v1'] + [c for c in OV_EXTRA if ov_found.get(c, 0) > 0]
    else:
        classes = CLASS_SETS[args.classes]
    ov_classes = [c for c in OV_CLASSES if c in classes and ov_found.get(c, 0) > 0]
    print(f'классы датасета ({args.classes}): {classes}')
    print(f'файл трека C: {ov_path}; классы на псевдо-кадрах: {ov_found or "нет"}; берём: {ov_classes or "ничего"}')
    ft_path = dets_file(args.ft_pseudo, 'dets', '') if args.ft_pseudo else None

    # 1. aux: holdout и обучение
    objects, aux_tables, aux_all = aux_boxes(classes)
    aux_frames = pd.read_csv(paths.DATA / 'aux_frames.csv')
    holdout, excluded = pick_holdout(aux_all, aux_frames)
    plan: dict[str, dict] = {}
    none = np.zeros((0, 4))
    for frame_id in sorted(aux_frames['frame_id']):
        if frame_id in excluded:
            continue
        split = 'holdout' if frame_id in holdout else 'train'
        labels = objects[objects['frame_id'] == frame_id]
        tables = aux_tables.loc[aux_tables['frame_id'] == frame_id, BOX].to_numpy(float)
        plan[frame_id] = {'jobs': frame_jobs(frame_id, 'aux', split, labels, tables, none), 'uncertain': none,
                          'confident': none}

    # 2. псевдометки: уверенные детекции источников идут в метки, неуверенные области под закраску
    prod = io.load_prod_dets()
    prod_labels = prod[prod['class_name'].isin([c for c in PROD_CLASSES if c in classes]) & prod['frame_id'].isin(pseudo_ids)]
    prod_labels = prod_labels.assign(src='prod')[LABEL_COLUMNS]
    ov = read_labels(ov_path, 'ov_pseudo', ov_classes, pseudo_ids)
    ft = read_labels(ft_path, 'ft', classes, pseudo_ids)
    ft_thr = ft['class_name'].map(FT_PSEUDO_CONF).fillna(PSEUDO_CONF)
    sure = pd.concat([prod_labels[prod_labels['confidence'] >= PSEUDO_CONF], ov, ft[ft['confidence'] >= ft_thr]],
                     ignore_index=True)
    unsure = pd.concat([prod_labels[prod_labels['confidence'] < PSEUDO_CONF],
                        ft[(ft['confidence'] >= UNSURE_MIN) & (ft['confidence'] < ft_thr)]], ignore_index=True)
    sure_by = dict(tuple(sure.groupby('frame_id')))
    unsure_by = dict(tuple(unsure.groupby('frame_id')))
    tables_by = dict(tuple(prod[prod['class_name'] == 'table'].groupby('frame_id')))
    pseudo_jobs, merged, n_painted = [], [], 0
    for frame_id in pseudo_ids:
        labels, uncertain = merge_frame_labels(sure_by.get(frame_id, sure.iloc[:0]), unsure_by.get(frame_id, unsure.iloc[:0]))
        merged.append(labels)
        n_painted += len(uncertain)
        tables = tables_by[frame_id][BOX].to_numpy(float) if frame_id in tables_by else none
        tables = tables[np.argsort(tables[:, 0], kind='stable')]
        pseudo_jobs += frame_jobs(frame_id, 'pseudo', 'train', labels, tables, uncertain)
        plan[frame_id] = {'jobs': [], 'uncertain': uncertain, 'confident': labels[BOX].to_numpy(float).reshape(-1, 4)}
    merged = pd.concat(merged, ignore_index=True) if merged else sure.iloc[:0]

    # 3. отбор псевдо-картинок: все с боксами и доля пустых по каждому виду (сид 419)
    rng = np.random.default_rng(SEED)
    for view in VIEWS:
        of_view = [j for j in pseudo_jobs if j['view'] == view]
        positive = [j for j in of_view if len(j['boxes'])]
        background = sorted((j for j in of_view if not len(j['boxes'])), key=lambda j: j['name'])
        n_bg = min(len(background), int(round(args.bg_frac * len(positive))))
        chosen = [background[i] for i in sorted(rng.choice(len(background), size=n_bg, replace=False))] if n_bg else []
        for job in positive + chosen:
            plan[job['frame_id']]['jobs'].append(job)

    # 4. запись
    for split in ['train', 'holdout']:
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
    work = [(frame_id, p) for frame_id, p in sorted(plan.items()) if p['jobs']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(lambda item: render(item[0], item[1]['jobs'], item[1]['uncertain'],
                                                  item[1]['confident'], out, classes), work))
    manifest = pd.DataFrame([r for part in parts for r in part]).sort_values('image').reset_index(drop=True)
    manifest.to_csv(out / 'manifest.csv', index=False)
    (out / 'data.yaml').write_text(
        '# Датасет своего детектора (трек B). Путь к каталогу подставляет tracks/det/train.py.\n'
        'train: images/train\nval: images/holdout\nnames:\n'
        + ''.join(f'  {i}: {c}\n' for i, c in enumerate(classes)), encoding='utf-8')
    preview(out, manifest, classes)

    by_group = manifest.groupby(['split', 'origin', 'view'])[[f'n_{c}' for c in classes]].sum()
    labels_by_source = {src: {c: int(n) for c, n in g['class_name'].value_counts().sort_index().items()}
                        for src, g in merged.groupby('src')}
    card = {
        'name': args.name, 'kind': 'dataset', 'owner': common.OWNER, 'created_utc': common.utc_now(),
        'git_sha': common.git_sha(), 'command': common.command_line(),
        'inputs': ['aux_boxes.csv', 'aux_frames.csv', 'detections.csv', 'frames.csv']
                  + [p.name for p in (ov_path, ft_path) if p is not None],
        'manual_labels_used': False,
        'uses_train_counters': False,
        'class_set': args.classes,
        'classes': classes,
        'box_sources': {
            'aux': {'frames': int(aux_frames['frame_id'].nunique()), 'boxes_used': int(len(objects)),
                    'note': 'aux_boxes.csv как есть, без правок'},
            'prod': {'classes': PROD_CLASSES, 'min_confidence': PSEUDO_CONF, 'candidates': int(len(prod_labels))},
            'ov_pseudo': {'file': None if ov_path is None else ov_path.name, 'classes': ov_classes,
                          'classes_in_file_on_pseudo_frames': ov_found, 'candidates': int(len(ov))},
            'ft': {'file': None if ft_path is None else ft_path.name, 'min_confidence': PSEUDO_CONF,
                   'min_confidence_by_class': FT_PSEUDO_CONF,
                   'painted_from': UNSURE_MIN, 'candidates': int(len(ft))},
            'pseudo_frames': len(pseudo_ids),
            'labels_after_merge_by_source': labels_by_source,
            'painted_boxes': int(n_painted),
        },
        'holdout_frames': holdout,
        'excluded_near_holdout': excluded,
        'pseudo_dates': list(args.pseudo_dates),
        'params': {'views': VIEWS, 'min_visible': MIN_VISIBLE, 'min_side_px': MIN_SIDE_PX, 'max_painted': MAX_PAINTED,
                   'bg_frac': args.bg_frac, 'pseudo_step': args.pseudo_step, 'paint_rgb': PAINT,
                   'dedup': {'iou': DEDUP_IOU, 'small_px': DEDUP_SMALL_PX, 'center_ratio': DEDUP_CENTER},
                   'jpeg_quality': JPEG_QUALITY, 'seed': SEED},
        'images': {f'{s}/{o}/{v}': int(n) for (s, o, v), n in manifest.groupby(['split', 'origin', 'view']).size().items()},
        'boxes_by_class': {s: {c: int(manifest.loc[manifest['split'] == s, f'n_{c}'].sum()) for c in classes}
                           for s in ['train', 'holdout']},
        'boxes_by_group': {f'{s}/{o}/{v}': {c[2:]: int(n) for c, n in row.items()}
                           for (s, o, v), row in by_group.iterrows()},
        'notes': 'кадры 25–26.10 в датасет не входят (правило 12.4 не применяется), если они не указаны в --pseudo-dates; '
                 'неуверенные детекции закрашены серым, уверенные боксы при этом не затронуты',
    }
    (out / 'card.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f'{args.name} ({args.classes}): {len(manifest)} картинок -> {out}')
    print('holdout aux:', ', '.join(holdout), '| убраны как соседи holdout:', ', '.join(excluded) or 'нет')
    print(manifest.groupby(['split', 'origin', 'view']).agg(images=('image', 'size'), boxes=('n_boxes', 'sum')).to_string())
    print('боксы по классам:', json.dumps(card['boxes_by_class'], ensure_ascii=False))
    print('метки псевдо-кадров после слияния, по источникам:', json.dumps(labels_by_source, ensure_ascii=False),
          '| закрашено боксов:', n_painted)
    print(f'размер каталога: {sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6:.0f} МБ; '
          f'проверка глазами: {out / "preview.jpg"}')


if __name__ == '__main__':
    main()
