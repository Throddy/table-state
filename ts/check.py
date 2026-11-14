"""Проверка форматов артефактов (разделы 7.9 и 8). Не прошёл проверку = не существует."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import COUNTS, FRAME_H, FRAME_W, PRED_COLUMNS, io
from .io import DET_COLUMNS
from .rows import ROWS_EXT_COLUMNS, ROWS_EXT_NAME
from . import paths

# Канонический словарь классов (раздел 8.2): класс -> счётчик или None.
CLASS_TO_COUNTER: dict[str, str | None] = {
    'person': 'people', 'hand': 'hands', 'cup': 'cups',
    'plate': 'dishes', 'bowl': 'dishes', 'tray': 'dishes',
    'phone': 'items', 'bag': 'items', 'wallet': 'items', 'headphones': 'items',
    'paper': 'items', 'personal_item': 'items',
    'laptop': None, 'tablet': None, 'bottle': None, 'can': None, 'vase': None,
    'napkin': None, 'food': None, 'glove_hand': None, 'table': None,
}
CARD_REQUIRED = ['name', 'kind', 'owner', 'manual_labels_used', 'uses_train_counters']
TRAIN_PRED_MODES = {'zero_shot', 'oof_by_date'}
DT_VALUES = {-30, 0, 30}
N_FRAMES = 1718
N_TARGET_FRAMES = 839    # целевые кадры (dt = 0): 39 train + 800 test
N_ROWS = 4964


class CheckError(ValueError):
    """Артефакт не соответствует формату контракта."""


def _fail(path: Path, message: str) -> None:
    raise CheckError(f'{path}: {message}')


def _need_columns(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        _fail(path, f'нет колонок {missing}')


def _card(path: Path, required: list[str]) -> dict:
    """Прочитать карточку `<имя>.json` рядом с файлом и проверить обязательные поля."""
    card_path = path.with_suffix('.json')
    if not card_path.is_file():
        _fail(path, f'нет карточки {card_path.name} (раздел 8.6)')
    try:
        card = json.loads(card_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        _fail(path, f'карточка {card_path.name} не читается как json: {error}')
    missing = [k for k in required if k not in card]
    if missing:
        _fail(path, f'в карточке {card_path.name} нет полей {missing}')
    for key in ['manual_labels_used', 'uses_train_counters']:
        if key in required and not isinstance(card[key], bool):
            _fail(path, f'поле карточки {key} должно быть true/false')
    if card.get('manual_labels_used'):
        _fail(path, 'manual_labels_used = true: ручная разметка любых кадров пакета запрещена (правило R7)')
    return card


def _source_of(path: Path) -> str:
    stem = path.stem
    if '__v' not in stem:
        _fail(path, 'имя файла должно быть <source>__v<N>.parquet')
    return stem.split('__v')[0]


def _numeric(path: Path, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    for c in columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            _fail(path, f'колонка {c} должна быть числовой, сейчас {df[c].dtype}')
    values = df[columns].to_numpy(float)
    if not np.isfinite(values).all():
        _fail(path, f'NaN или бесконечность в колонках {columns}')
    return values


def check_dets(path) -> None:
    """Проверить файл детекций формата 8.1 и его карточку. Падает с `CheckError`."""
    path = Path(path)
    if not path.is_file():
        _fail(path, 'файл не найден')
    source = _source_of(path)
    df = pd.read_parquet(path)
    _need_columns(path, df, DET_COLUMNS)
    if df['det_id'].isna().any() or df['det_id'].duplicated().any():
        _fail(path, 'det_id должен быть непустым и уникальным')
    if set(df['source'].unique()) - {source}:
        _fail(path, f"колонка source должна быть равна '{source}', найдено {sorted(df['source'].unique())}")
    unknown = sorted(set(df['class_name']) - set(CLASS_TO_COUNTER))
    if unknown:
        _fail(path, f'классы не из словаря 8.2: {unknown}')
    frames = set(io.load_frames()['frame_id'])
    bad = set(df['frame_id']) - frames
    if bad:
        _fail(path, f'{len(bad)} frame_id не из 1718 основных кадров, например {sorted(bad)[0]}')
    v = _numeric(path, df, ['confidence', 'x1', 'y1', 'x2', 'y2'])
    if (v[:, 0] < 0).any() or (v[:, 0] > 1).any():
        _fail(path, 'confidence должен быть в [0, 1]')
    if (v[:, 3] <= v[:, 1]).any() or (v[:, 4] <= v[:, 2]).any():
        _fail(path, 'нужно x2 > x1 и y2 > y1')
    if (v[:, 1] < 0).any() or (v[:, 2] < 0).any() or (v[:, 3] > FRAME_W).any() or (v[:, 4] > FRAME_H).any():
        _fail(path, f'боксы выходят за кадр {FRAME_W}×{FRAME_H}')
    dts = set(pd.to_numeric(df['from_dt'].dropna()).astype(int).unique())
    if dts - DT_VALUES:
        _fail(path, f'from_dt допускает только -30, 0, 30 или null, найдено {sorted(dts - DT_VALUES)}')
    card = _card(path, CARD_REQUIRED + ['coverage', 'frames_processed'])
    if card['coverage'] not in ('all', 'targets', 'train_only'):
        _fail(path, "coverage в карточке: 'all', 'targets' или 'train_only'")
    n = card['frames_processed'].get('n') if isinstance(card['frames_processed'], dict) else None
    if card['coverage'] == 'all' and n != N_FRAMES:
        _fail(path, f'coverage=all требует frames_processed.n = {N_FRAMES}, сейчас {n}')
    if card['coverage'] == 'targets' and n != N_TARGET_FRAMES:
        _fail(path, f'coverage=targets требует frames_processed.n = {N_TARGET_FRAMES} (все целевые кадры), сейчас {n}')
    if n is not None and df['frame_id'].nunique() > n:
        _fail(path, 'кадров с детекциями больше, чем frames_processed.n')
    # Заявленное покрытие сверяется с данными, а не принимается на веру.
    expected = {'all': frames, 'targets': set(io.load_tables()['frame_id'])}.get(card['coverage'])
    if expected is not None:
        extra = set(df['frame_id']) - expected
        if extra:
            _fail(path, f"coverage={card['coverage']}: {len(extra)} frame_id вне заявленного набора кадров, "
                        f'например {sorted(extra)[0]}')
        declared = card['frames_processed'].get('sha1')
        actual = hashlib.sha1('\n'.join(sorted(expected)).encode('utf-8')).hexdigest()
        if declared != actual:
            _fail(path, f"frames_processed.sha1 не совпадает с sha1 отсортированного списка кадров покрытия "
                        f"'{card['coverage']}' (id через перевод строки): в карточке {declared}, должно быть {actual}")


def check_rowfeats(path) -> None:
    """Проверить файл признаков строк формата 8.4 и его карточку."""
    path = Path(path)
    if not path.is_file():
        _fail(path, 'файл не найден')
    source = _source_of(path)
    df = pd.read_parquet(path)
    _need_columns(path, df, ['row_id', 'dt'])
    features = [c for c in df.columns if c not in ('row_id', 'dt')]
    wrong = [c for c in features if not c.startswith(source + '__')]
    if not features or wrong:
        _fail(path, f"нужны колонки признаков вида '{source}__<имя>', лишние: {wrong}")
    _numeric(path, df, features)
    if set(pd.unique(df['dt'])) - DT_VALUES:
        _fail(path, 'dt допускает только -30, 0, 30')
    if df.duplicated(['row_id', 'dt']).any():
        _fail(path, 'пары (row_id, dt) должны быть уникальны')
    expected = set(io.load_tables()['row_id'])
    current = set(df.loc[df['dt'] == 0, 'row_id'])
    if current != expected:
        _fail(path, f'строки dt=0: не хватает {len(expected - current)}, лишних {len(current - expected)} '
                    f'(нужны все {N_ROWS})')
    if set(df['row_id']) - expected:
        _fail(path, 'есть row_id не из tables.csv')
    if source.startswith('vlm'):
        _need_columns(path, df, [f'{source}__{c}' for c in COUNTS + ['ok']])
    ok = f'{source}__ok'
    if ok in df.columns and not df[ok].isin([0, 1]).all():
        _fail(path, f'{ok} допускает только 0 и 1')
    _card(path, CARD_REQUIRED)


def check_pred(path) -> None:
    """Проверить файл предсказаний формата 8.5 (все 4964 строки) и его карточку."""
    path = Path(path)
    if not path.is_file():
        _fail(path, 'файл не найден')
    df = io.read_csv_exact(path)
    if list(df.columns) != PRED_COLUMNS:
        _fail(path, f'колонки должны быть ровно {PRED_COLUMNS}, сейчас {list(df.columns)}')
    if df['row_id'].isna().any() or df['row_id'].duplicated().any():
        _fail(path, 'row_id должен быть непустым и уникальным')
    expected = set(io.load_tables()['row_id'])
    current = set(df['row_id'])
    if current != expected:
        _fail(path, f'не хватает {len(expected - current)} строк, лишних {len(current - expected)} '
                    f'(нужны все {N_ROWS}: 231 train + 4733 test)')
    v = _numeric(path, df, COUNTS + ['occ_prob'])
    if (v < 0).any() or (v[:, :5] > 50).any() or (v[:, 5] > 1).any():
        _fail(path, 'счётчики должны быть в [0, 50], occ_prob в [0, 1]')
    card = _card(path, ['name', 'train_pred', 'uses_train_counters'])
    if card['train_pred'] not in TRAIN_PRED_MODES:
        _fail(path, f'train_pred в карточке: одно из {sorted(TRAIN_PRED_MODES)}')
    if card['uses_train_counters'] and card['train_pred'] != 'oof_by_date':
        _fail(path, 'uses_train_counters=true требует train_pred=oof_by_date (раздел 8.5)')


def check_rows_ext(path=None) -> None:
    """Проверить `rows_ext.parquet` (формат 8.3). По умолчанию `ART/rows_ext.parquet`."""
    path = Path(path) if path is not None else paths.ART / ROWS_EXT_NAME
    if not path.is_file():
        _fail(path, 'файл не найден, запустите: python -m ts.rows')
    df = pd.read_parquet(path)
    if list(df.columns) != ROWS_EXT_COLUMNS:
        _fail(path, f'колонки должны быть ровно {ROWS_EXT_COLUMNS}')
    if df.isna().any().any():
        _fail(path, 'пропуски в таблице')
    if set(df['dt'].unique()) - DT_VALUES:
        _fail(path, 'dt допускает только -30, 0, 30')
    if df.duplicated(['row_id', 'dt']).any():
        _fail(path, 'пары (row_id, dt) должны быть уникальны')
    current = df[df['dt'] == 0]
    tables = io.load_tables()
    if len(current) != N_ROWS or set(current['row_id']) != set(tables['row_id']):
        _fail(path, f'нужно ровно {N_ROWS} строк с dt = 0 (все row_id из tables.csv), сейчас {len(current)}')
    if not (current['frame_id'] == current['src_frame_id']).all():
        _fail(path, 'при dt = 0 src_frame_id должен совпадать с frame_id')
    if (df.loc[df['dt'] != 0, 'frame_id'] == df.loc[df['dt'] != 0, 'src_frame_id']).any():
        _fail(path, 'при dt != 0 src_frame_id должен быть соседним кадром')
    if df.groupby('row_id')[['x1', 'y1', 'x2', 'y2', 'slot', 'table_id']].nunique().max().max() != 1:
        _fail(path, 'рамка стола у соседей должна совпадать с рамкой целевого кадра')
    if not set(df['split'].unique()) <= {'train', 'test'}:
        _fail(path, 'split допускает только train и test')
    if not df['slot'].between(1, 6).all() or not df['is_ir'].isin([0, 1]).all():
        _fail(path, 'slot должен быть 1..6, is_ir 0 или 1')
    if not set(df['src_frame_id']) <= set(io.load_frames()['frame_id']):
        _fail(path, 'src_frame_id не из frames.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Проверка артефакта по контракту.')
    parser.add_argument('kind', choices=['dets', 'rowfeats', 'pred', 'rows_ext'])
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    if args.kind == 'rows_ext':
        check_rows_ext(args.path)
    else:
        if not args.path:
            parser.error('нужен путь к файлу')
        {'dets': check_dets, 'rowfeats': check_rowfeats, 'pred': check_pred}[args.kind](args.path)
    print('OK')
