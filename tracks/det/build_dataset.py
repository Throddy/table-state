"""Датасет своего детектора в формате YOLO (задачи B2 и B3, вариант «псевдо»). Ручной разметки нет.

Источников боксов ровно три:
  1. aux как есть (aux_boxes.csv, маппинг 8.2);
  2. прод-детекции с confidence >= 0.6 (person, hand, cup, tray) на кадрах выбранных дней;
  3. уверенные детекции трека C из задачи C5 (plate, phone), если есть файл ART/labels/ov_pseudo__v<N>.parquet.

Для каждого кадра делаются три вида: полный кадр, кропы зон top (×4) и wide (×2.5) каждого стола с
пересчитанными боксами. Столы на aux это боксы table из aux, на остальных кадрах детекции table прод-детектора.
Области неуверенных прод-детекций (confidence < 0.6) закрашиваются серым: Ultralytics не умеет области игнора.
6 aux-кадров с тарелками и телефонами откладываются в holdout и в обучение не попадают.

Запуск:
  python tracks/det/build_dataset.py --name ft_data_v1
По умолчанию псевдометки берутся только с кадров 27–28.10: кадры 25–26.10 в обучение не идут,
поэтому правило 12.4 не применяется. Результат: ART/datasets/<name>/ (images, labels, data.yaml,
manifest.csv, card.json, preview.jpg). Каталог публикуется отдельным Kaggle Dataset.
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
from common import FT_CLASSES, TEST_DATES

from ts import FRAME_H, FRAME_W, SEED, io, paths
from ts.crops import crop_region, to_crop_coords
from ts.geometry import clip_box, zone
from ts.slots import slot_of

VIEWS = {'full': None, 'top': 4.0, 'wide': 2.5}     # вид -> увеличение кропа (раздел 9.1)
PROD_CLASSES = ['person', 'hand', 'cup', 'tray']     # что берём из прод-детектора
OV_CLASSES = ['plate', 'phone']                      # что берём из псевдометок трека C (задача C5)
PSEUDO_CONF = 0.6                                    # B3: уверенные прод-детекции
MIN_VISIBLE = 0.5                                    # бокс остаётся в кропе, если видно не меньше половины площади
MIN_SIDE_PX = 2.0                                    # и после обрезки он не тоньше 2 px кропа
MAX_PAINTED = 0.5                                    # кроп, закрашенный больше чем наполовину, не используется
N_HOLDOUT = 6
NEAR_SECONDS = 30                                    # кадры той же сцены не делятся между обучением и holdout
PAINT = (114, 114, 114)
JPEG_QUALITY = 95


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


# ---------- источники боксов ----------

def aux_boxes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Боксы aux как есть: (объекты классов детектора, столы). Классы вне списка B2 не используются."""
    aux = io.load_aux_boxes()
    return aux[aux['class_name'].isin(FT_CLASSES)], aux[aux['class_name'] == 'table']


def pick_holdout(objects: pd.DataFrame, aux_frames: pd.DataFrame) -> tuple[list[str], list[str]]:
    """6 aux-кадров holdout и кадры той же сцены, которые нельзя оставлять в обучении.

    Сначала кадры, где есть и тарелки, и телефоны; если их меньше шести, добираются кадры с тарелками,
    затем с телефонами. Внутри группы порядок случайный с сидом 419. Кадр, отстоящий от уже выбранного
    не больше чем на 30 с, в holdout не берётся, а из обучения убирается (та же сцена, аналог правила R4).
    """
    counts = objects.pivot_table(index='frame_id', columns='class_name', values='det_id', aggfunc='size', fill_value=0)
    counts = counts.reindex(index=sorted(aux_frames['frame_id']), columns=FT_CLASSES, fill_value=0)
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


def ov_pseudo(path: Path | None) -> tuple[pd.DataFrame, str | None]:
    """Псевдометки трека C (формат 8.1), только классы plate и phone, и путь к файлу. Без файла таблица пустая."""
    if path is None:
        found = sorted((paths.ART / 'labels').glob('ov_pseudo__v*.parquet'))
        path = found[-1] if found else None
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']), None
    table = pd.read_parquet(path)
    return table[table['class_name'].isin(OV_CLASSES)].copy(), str(path)


# ---------- сборка ----------

def frame_jobs(frame_id: str, origin: str, split: str, labels: pd.DataFrame, tables: np.ndarray,
               uncertain: np.ndarray) -> list[dict]:
    """Все виды одного кадра: какие боксы попадают в каждый и нужен ли вид без боксов."""
    boxes = labels[['x1', 'y1', 'x2', 'y2']].to_numpy(float).reshape(-1, 4)
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


def render(frame_id: str, jobs: list[dict], uncertain: np.ndarray, confident: np.ndarray, out: Path) -> list[dict]:
    """Записать картинки и файлы меток выбранных видов одного кадра. Возвращает строки манифеста."""
    image = io.read_image(frame_id)
    if len(uncertain):
        pixels = np.asarray(image).copy()
        pixels[paint_mask(uncertain, confident)] = PAINT
        image = Image.fromarray(pixels)
    records = []
    for job in jobs:
        if job['view'] == 'full':
            crop, sx, sy, x0, y0 = image, 1.0, 1.0, 0, 0
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
        lines, counts = [], {c: 0 for c in FT_CLASSES}
        for (x1, y1, x2, y2), name in zip(in_crop, job['classes']):
            if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
                continue
            lines.append(f'{FT_CLASSES.index(name)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} '
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


def preview(out: Path, manifest: pd.DataFrame, n: int = 10) -> None:
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
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=colors[int(c)], width=2)
            draw.text((cx - w / 2 + 2, max(0, cy - h / 2 - 12)), FT_CLASSES[int(c)], fill=colors[int(c)])
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
    parser.add_argument('--pseudo-dates', nargs='*', default=list(TEST_DATES),
                        help='дни UTC, с которых берутся псевдометки (по умолчанию 27–28.10; пусто = только aux)')
    parser.add_argument('--ov-pseudo', type=Path, default=None,
                        help='файл псевдометок трека C; по умолчанию последний ART/labels/ov_pseudo__v*.parquet')
    parser.add_argument('--bg-frac', type=float, default=0.1,
                        help='сколько псевдо-картинок без боксов оставить, доля от числа картинок с боксами того же вида')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    common.require_data()
    out = paths.ART / 'datasets' / args.name
    if out.exists():
        raise SystemExit(f'{out} уже существует. Датасет неизменяем: выберите другое --name.')

    # 1. aux: holdout и обучение
    objects, aux_tables = aux_boxes()
    aux_frames = pd.read_csv(paths.DATA / 'aux_frames.csv')
    holdout, excluded = pick_holdout(objects, aux_frames)
    plan: dict[str, dict] = {}
    for frame_id in sorted(aux_frames['frame_id']):
        if frame_id in excluded:
            continue
        split = 'holdout' if frame_id in holdout else 'train'
        labels = objects[objects['frame_id'] == frame_id]
        tables = aux_tables.loc[aux_tables['frame_id'] == frame_id, ['x1', 'y1', 'x2', 'y2']].to_numpy(float)
        none = np.zeros((0, 4))
        plan[frame_id] = {'jobs': frame_jobs(frame_id, 'aux', split, labels, tables, none), 'uncertain': none,
                          'confident': none}

    # 2. псевдометки: прод-детекции >= 0.6 и метки трека C, неуверенные области под закраску
    frames = io.load_frames()
    pseudo_ids = sorted(frames.loc[frames['date_utc'].isin(args.pseudo_dates), 'frame_id'])
    prod = io.load_prod_dets()
    ov, ov_path = ov_pseudo(args.ov_pseudo)
    prod_objects = prod[prod['class_name'].isin(PROD_CLASSES) & prod['frame_id'].isin(pseudo_ids)]
    ov = ov[ov['frame_id'].isin(pseudo_ids)]
    sure_prod = prod_objects[prod_objects['confidence'] >= PSEUDO_CONF]
    sure = pd.concat([sure_prod[['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']],
                      ov[['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']]], ignore_index=True)
    unsure = prod_objects[prod_objects['confidence'] < PSEUDO_CONF]
    sure_by = dict(tuple(sure.groupby('frame_id')))
    unsure_by = dict(tuple(unsure.groupby('frame_id')))
    tables_by = dict(tuple(prod[prod['class_name'] == 'table'].groupby('frame_id')))
    empty = sure.iloc[:0]
    pseudo_jobs = []
    for frame_id in pseudo_ids:
        labels = sure_by.get(frame_id, empty)
        tables = tables_by[frame_id][['x1', 'y1', 'x2', 'y2']].to_numpy(float) if frame_id in tables_by else np.zeros((0, 4))
        tables = tables[np.argsort(tables[:, 0], kind='stable')]
        uncertain = (unsure_by[frame_id][['x1', 'y1', 'x2', 'y2']].to_numpy(float) if frame_id in unsure_by
                     else np.zeros((0, 4)))
        jobs = frame_jobs(frame_id, 'pseudo', 'train', labels, tables, uncertain)
        plan[frame_id] = {'jobs': [], 'uncertain': uncertain,
                          'confident': labels[['x1', 'y1', 'x2', 'y2']].to_numpy(float).reshape(-1, 4)}
        pseudo_jobs += jobs

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
                                                  item[1]['confident'], out), work))
    manifest = pd.DataFrame([r for part in parts for r in part]).sort_values('image').reset_index(drop=True)
    manifest.to_csv(out / 'manifest.csv', index=False)
    (out / 'data.yaml').write_text(
        '# Датасет своего детектора (трек B). Путь к каталогу подставляет tracks/det/train.py.\n'
        'train: images/train\nval: images/holdout\nnames:\n'
        + ''.join(f'  {i}: {c}\n' for i, c in enumerate(FT_CLASSES)), encoding='utf-8')
    preview(out, manifest)

    box_columns = [f'n_{c}' for c in FT_CLASSES]
    by_group = manifest.groupby(['split', 'origin', 'view'])[box_columns].sum()
    card = {
        'name': args.name, 'kind': 'dataset', 'owner': common.OWNER, 'created_utc': common.utc_now(),
        'git_sha': common.git_sha(), 'command': common.command_line(),
        'inputs': ['aux_boxes.csv', 'aux_frames.csv', 'detections.csv', 'frames.csv']
                  + ([Path(ov_path).name] if ov_path else []),
        'manual_labels_used': False,
        'uses_train_counters': False,
        'classes': FT_CLASSES,
        'box_sources': {
            'aux': {'frames': int(aux_frames['frame_id'].nunique()), 'boxes_used': int(len(objects)),
                    'note': 'aux_boxes.csv как есть, без правок'},
            'prod': {'classes': PROD_CLASSES, 'min_confidence': PSEUDO_CONF, 'dates': list(args.pseudo_dates),
                     'frames': len(pseudo_ids), 'boxes_confident': int(len(sure_prod)),
                     'boxes_painted': int(len(unsure))},
            'ov_pseudo': {'file': ov_path, 'classes': OV_CLASSES, 'boxes': int(len(ov))},
        },
        'holdout_frames': holdout,
        'excluded_near_holdout': excluded,
        'pseudo_dates': list(args.pseudo_dates),
        'params': {'views': VIEWS, 'min_visible': MIN_VISIBLE, 'min_side_px': MIN_SIDE_PX, 'max_painted': MAX_PAINTED,
                   'bg_frac': args.bg_frac, 'paint_rgb': PAINT, 'jpeg_quality': JPEG_QUALITY, 'seed': SEED},
        'images': {f'{s}/{o}/{v}': int(n) for (s, o, v), n in manifest.groupby(['split', 'origin', 'view']).size().items()},
        'boxes_by_class': {s: {c: int(manifest.loc[manifest['split'] == s, f'n_{c}'].sum()) for c in FT_CLASSES}
                           for s in ['train', 'holdout']},
        'boxes_by_group': {f'{s}/{o}/{v}': {c[2:]: int(n) for c, n in row.items()}
                           for (s, o, v), row in by_group.iterrows()},
        'notes': 'кадры 25–26.10 в датасет не входят (правило 12.4 не применяется), если они не указаны в --pseudo-dates; '
                 'неуверенные прод-детекции закрашены серым, уверенные боксы при этом не затронуты',
    }
    (out / 'card.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f'{args.name}: {len(manifest)} картинок -> {out}')
    print('holdout aux:', ', '.join(holdout), '| убраны как соседи holdout:', ', '.join(excluded) or 'нет')
    print(manifest.groupby(['split', 'origin', 'view']).agg(images=('image', 'size'), boxes=('n_boxes', 'sum')).to_string())
    print('боксы по классам:', json.dumps(card['boxes_by_class'], ensure_ascii=False))
    print(f'размер каталога: {sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6:.0f} МБ; '
          f'проверка глазами: {out / "preview.jpg"}')


if __name__ == '__main__':
    main()
