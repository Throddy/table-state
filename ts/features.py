"""Сборка признаков строк из источников (раздел 7.7)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import io, paths
from .assign import assign
from .check import check_dets, check_rowfeats
from .rows import load_rows_ext
from .temporal import aggregate

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


def build_features(sources: list[str], cfg: dict) -> pd.DataFrame:
    """Признаки всех 4964 строк по списку источников.

    `sources`: `'prod'`, имена файлов `ART/dets/<source>__v<N>` или
    `ART/rowfeats/<source>__v<N>` без расширения. Источники детекций проходят
    `assign` и `aggregate`; признаки строк берутся при `dt = 0` как есть.
    Возвращает таблицу с индексом `row_id` (4964 строки): колонки источников и
    `slot, is_ir, hour_msk, n_tables, n_ctx` (число доступных соседей).
    """
    rows = load_rows_ext()
    current = rows[rows['dt'] == 0].set_index('row_id')
    out = current[META_COLUMNS].astype(float).copy()
    out['n_ctx'] = (rows.groupby('row_id').size() - 1).reindex(out.index).astype(float)
    for source in sources:
        rowfeats = paths.ART / 'rowfeats' / f'{source}.parquet'
        if source != 'prod' and rowfeats.is_file():
            check_rowfeats(rowfeats)
            table = pd.read_parquet(rowfeats)
            part = table[table['dt'] == 0].drop(columns='dt').set_index('row_id')
        else:
            part = aggregate(assigned_for(source, cfg), cfg, rows)
        out = out.join(part.reindex(out.index), how='left')
    if len(out) != 4964:
        raise ValueError(f'Ожидалось 4964 строки признаков, получено {len(out)}')
    return out
