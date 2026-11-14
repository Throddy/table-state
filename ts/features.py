"""Сборка признаков строк из источников (раздел 7.7)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import io, paths
from .assign import assign
from .check import check_dets, check_rowfeats
from .rows import load_rows_ext
from .temporal import aggregate, merge_sources

META_COLUMNS = ['slot', 'is_ir', 'hour_msk', 'n_tables']


def load_dets(source: str) -> pd.DataFrame:
    """Детекции источника: `'prod'` из пакета, остальные из `ART/dets/<source>.parquet` (с проверкой)."""
    if source == 'prod':
        return io.load_prod_dets()
    path = paths.ART / 'dets' / f'{source}.parquet'
    check_dets(path)
    return pd.read_parquet(path)


@lru_cache(maxsize=8)
def _assigned_cached(source: str, key: str) -> pd.DataFrame:
    import json
    return assign(load_dets(source), load_rows_ext(), {'assign': json.loads(key)})


def assigned_for(source: str, cfg: dict | None = None) -> pd.DataFrame:
    """Привязанные объекты источника по всем строкам `rows_ext` (кэшируется в памяти)."""
    import json
    return _assigned_cached(source, json.dumps((cfg or {}).get('assign', {}), sort_keys=True))


def split_sources(sources: list[str]) -> tuple[list[str], list[str]]:
    """Разделить имена источников на детекции и признаки строк (по наличию файла в `ART/rowfeats`)."""
    dets, rowfeats = [], []
    for source in sources:
        is_rowfeats = source != 'prod' and (paths.ART / 'rowfeats' / f'{source}.parquet').is_file()
        (rowfeats if is_rowfeats else dets).append(source)
    return dets, rowfeats


def source_parts(sources: list[str], cfg: dict | None = None) -> list[pd.DataFrame]:
    """Привязанные объекты по источникам детекций с учётом `cfg['sources']['counters']`.

    `sources.counters: {<source>: [hands, cups]}` оставляет от источника только объекты перечисленных
    счётчиков: так новый источник меняет ровно одну колонку, а остальные считаются по прежним.
    `sources.classes: {<source>: [plate, bowl]}` оставляет только перечисленные классы словаря 8.2.
    """
    dets, _ = split_sources(sources)
    limits = ((cfg or {}).get('sources') or {}).get('counters') or {}
    classes = ((cfg or {}).get('sources') or {}).get('classes') or {}
    parts = []
    for source in dets:
        part = assigned_for(source, cfg)
        if source in limits:
            part = part[part['counter'].isin(limits[source])]
        if source in classes:
            part = part[part['class_name'].isin(classes[source])]
        parts.append(part)
    return parts


def merged_assigned(sources: list[str], cfg: dict | None = None) -> pd.DataFrame:
    """Объединённый список объектов всех источников детекций (`source = 'all'`)."""
    return merge_sources(source_parts(sources, cfg), cfg)


def build_features(sources: list[str], cfg: dict) -> pd.DataFrame:
    """Признаки всех 4964 строк по списку источников.

    `sources`: `'prod'`, имена файлов `ART/dets/<source>__v<N>` или
    `ART/rowfeats/<source>__v<N>` без расширения (каждый файл проходит
    `ts.check`). Источники детекций проходят `assign`; их объекты объединяются
    с дедупликацией (`merge_sources`) и сворачиваются `aggregate` в колонки
    `all__<counter>__<stat>`. Признаки строк берутся при `dt = 0` как есть.
    Возвращает таблицу с индексом `row_id` (4964 строки): колонки источников и
    `slot, is_ir, hour_msk, n_tables, n_ctx` (число доступных соседей).
    """
    rows = load_rows_ext()
    current = rows[rows['dt'] == 0].set_index('row_id')
    out = current[META_COLUMNS].astype(float).copy()
    out['n_ctx'] = (rows.groupby('row_id').size() - 1).reindex(out.index).astype(float)
    dets, rowfeats = split_sources(sources)
    merged = merge_sources(source_parts(sources, cfg), cfg)
    out = out.join(aggregate(merged, {**(cfg or {}), 'source': 'all'}, rows).reindex(out.index), how='left')
    for source in rowfeats:
        path = paths.ART / 'rowfeats' / f'{source}.parquet'
        check_rowfeats(path)
        table = pd.read_parquet(path)
        out = out.join(table[table['dt'] == 0].drop(columns='dt').set_index('row_id').reindex(out.index), how='left')
    if len(out) != 4964:
        raise ValueError(f'Ожидалось 4964 строки признаков, получено {len(out)}')
    return out
