"""Чтение таблиц и изображений пакета организаторов (раздел 7.2)."""
from __future__ import annotations

import pandas as pd
from PIL import Image

from . import FRAME_H, FRAME_W
from . import paths
from .slots import slot_of

DET_COLUMNS = ['det_id', 'frame_id', 'source', 'class_name', 'confidence',
               'x1', 'y1', 'x2', 'y2', 'from_row_id', 'from_dt']

PROD_CLASS_MAP = {'nude_hands': 'hand', 'tray': 'tray', 'cup': 'cup',
                  'person': 'person', 'table': 'table', 'napkin': 'napkin'}
AUX_CLASS_MAP = {'nude_hands': 'hand', 'personal_items': 'personal_item',
                 'empty_plate': 'plate', 'full_plate': 'plate'}


def _csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(paths.DATA / name, **kwargs)


def read_csv_exact(path) -> pd.DataFrame:
    """Прочитать csv с точным разбором float (значение после записи и чтения не меняется)."""
    if int(pd.__version__.split('.')[0]) >= 3:
        return pd.read_csv(path)
    return pd.read_csv(path, float_precision='round_trip')


def load_frames() -> pd.DataFrame:
    """`frames.csv` + `ts` (datetime UTC), `date_utc` (YYYY-MM-DD), `hour_msk` (int)."""
    df = _csv('frames.csv')
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True)
    df['date_utc'] = df['ts'].dt.strftime('%Y-%m-%d')
    df['hour_msk'] = ((df['ts'].dt.hour + 3) % 24).astype(int)
    return df


def load_targets() -> pd.DataFrame:
    """`targets.csv` + `context`: список id соседних кадров (0–2 штуки)."""
    df = _csv('targets.csv', keep_default_na=False)
    df['context'] = [[n for n in s.split(';') if n] for s in df['context_frames']]
    return df


def load_tables() -> pd.DataFrame:
    """`tables.csv` + `slot, cx, cy, w, h, split, date_utc`."""
    df = _csv('tables.csv')
    df['cx'] = (df['x1'] + df['x2']) / 2
    df['cy'] = (df['y1'] + df['y2']) / 2
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']
    df['slot'] = df['cx'].map(slot_of).astype(int)
    df = df.merge(load_targets()[['frame_id', 'split']], on='frame_id', how='left', validate='many_to_one')
    df = df.merge(load_frames()[['frame_id', 'date_utc']], on='frame_id', how='left', validate='many_to_one')
    return df


def _as_dets(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Привести таблицу боксов к формату 8.1 (пиксели кадра, канонические классы)."""
    df = df.copy()
    df['source'] = source
    n = df.groupby('frame_id', sort=False).cumcount()
    df['det_id'] = source + '_' + df['frame_id'] + '_' + n.astype(str)
    df['from_row_id'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df['from_dt'] = pd.array([pd.NA] * len(df), dtype='Int64')
    for c in ['confidence', 'x1', 'y1', 'x2', 'y2']:
        df[c] = df[c].astype(float)
    return df[DET_COLUMNS].reset_index(drop=True)


def load_prod_dets() -> pd.DataFrame:
    """`detections.csv` в формате 8.1, `source='prod'`, классы из словаря 8.2."""
    df = _csv('detections.csv')
    unknown = set(df['class_name']) - set(PROD_CLASS_MAP)
    if unknown:
        raise ValueError(f'Неизвестные классы прод-детектора: {sorted(unknown)}')
    df['class_name'] = df['class_name'].map(PROD_CLASS_MAP)
    return _as_dets(df, 'prod')


def load_labels() -> pd.DataFrame:
    """`train_labels.csv` как есть (231 строка)."""
    return _csv('train_labels.csv')


def load_aux_boxes() -> pd.DataFrame:
    """`aux_boxes.csv` в пикселях кадра 1280×720, формат 8.1, `source='aux_gt'`.

    Нормированные `cx, cy, w, h` переводятся в `x1, y1, x2, y2` и обрезаются
    по кадру. Классы приводятся к словарю 8.2. `confidence = 1.0`.
    Дополнительная колонка `batch_id` сохраняется.
    """
    df = _csv('aux_boxes.csv')
    df['class_name'] = df['class_name'].replace(AUX_CLASS_MAP)
    df['x1'] = ((df['cx'] - df['w'] / 2) * FRAME_W).clip(0, FRAME_W)
    df['x2'] = ((df['cx'] + df['w'] / 2) * FRAME_W).clip(0, FRAME_W)
    df['y1'] = ((df['cy'] - df['h'] / 2) * FRAME_H).clip(0, FRAME_H)
    df['y2'] = ((df['cy'] + df['h'] / 2) * FRAME_H).clip(0, FRAME_H)
    df['confidence'] = 1.0
    out = _as_dets(df, 'aux_gt')
    out['batch_id'] = df['batch_id'].to_numpy()
    return out


def read_image(frame_id: str) -> Image.Image:
    """Кадр как PIL.Image в RGB. Работает для `im_*` и `aux_*`."""
    with Image.open(paths.image_path(frame_id)) as im:
        return im.convert('RGB')
