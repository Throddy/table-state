"""Главный индекс строк `rows_ext.parquet` (разделы 7.4 и 8.3)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from . import io, paths

ROWS_EXT_NAME = 'rows_ext.parquet'
ROWS_EXT_COLUMNS = ['row_id', 'split', 'date_utc', 'frame_id', 'src_frame_id', 'dt',
                    'table_id', 'slot', 'x1', 'y1', 'x2', 'y2', 'n_tables', 'hour_msk', 'is_ir']
# Средняя насыщенность HSV (0..1) уменьшенного кадра. Ниже порога кадр считается ИК (серым).
# Порог подобран по гистограмме всех 1718 кадров, см. experiments/LOG.md.
IR_SAT_THRESHOLD = 0.05
_SAT_SIZE = (160, 90)


def frame_saturation(frame_id: str) -> float:
    """Средняя насыщенность HSV кадра, уменьшенного до 160×90, в диапазоне 0..1."""
    with Image.open(paths.image_path(frame_id)) as im:
        im.draft('RGB', (_SAT_SIZE[0] * 2, _SAT_SIZE[1] * 2))
        small = im.convert('RGB').resize(_SAT_SIZE, Image.Resampling.BILINEAR)
    saturation = np.asarray(small.convert('HSV'), dtype=np.float32)[..., 1]
    return float(saturation.mean() / 255.0)


def frame_saturations(frame_ids: list[str]) -> pd.Series:
    """`frame_saturation` для списка кадров (читает JPEG в несколько потоков)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(frame_saturation, frame_ids))
    return pd.Series(values, index=pd.Index(frame_ids, name='frame_id'), name='saturation')


def build_rows_ext() -> pd.DataFrame:
    """Построить и записать `ART/rows_ext.parquet` (формат 8.3).

    Для каждой строки `tables.csv` создаётся строка `dt = 0` и по строке на
    каждого доступного соседа (`dt = -30` или `30`). Рамка стола у соседа та
    же, что у целевого кадра; `table_id` между кадрами не сопоставляется.
    `hour_msk` относится к целевому кадру, `is_ir` считается по `src_frame_id`.
    """
    frames = io.load_frames().set_index('frame_id')
    targets = io.load_targets().set_index('frame_id')
    tables = io.load_tables()
    tables['n_tables'] = tables.groupby('frame_id')['row_id'].transform('size').astype(int)
    tables['hour_msk'] = tables['frame_id'].map(frames['hour_msk']).astype(int)

    links = [(frame_id, frame_id, 0) for frame_id in targets.index]
    for frame_id, context in targets['context'].items():
        for neighbour in context:
            delta = (frames.at[neighbour, 'ts'] - frames.at[frame_id, 'ts']).total_seconds()
            dt = int(round(delta / 30.0)) * 30
            if dt not in (-30, 30):
                raise ValueError(f'Сосед {neighbour} кадра {frame_id} отстоит на {delta} с')
            links.append((frame_id, neighbour, dt))
    links = pd.DataFrame(links, columns=['frame_id', 'src_frame_id', 'dt'])

    rows = tables.merge(links, on='frame_id', how='left', validate='many_to_many')
    saturation = frame_saturations(sorted(rows['src_frame_id'].unique()))
    rows['is_ir'] = (rows['src_frame_id'].map(saturation) < IR_SAT_THRESHOLD).astype('int8')
    rows['dt'] = rows['dt'].astype('int16')
    rows['slot'] = rows['slot'].astype('int8')
    rows = rows.sort_values(['frame_id', 'table_id', 'row_id', 'dt'], kind='stable')
    rows = rows[ROWS_EXT_COLUMNS].reset_index(drop=True)
    rows.to_parquet(paths.art('', ROWS_EXT_NAME), index=False)
    return rows


def load_rows_ext(build: bool = True) -> pd.DataFrame:
    """Прочитать `rows_ext.parquet`; если файла нет и `build=True`, построить."""
    path = paths.ART / ROWS_EXT_NAME
    if path.is_file():
        return pd.read_parquet(path)
    if not build:
        raise FileNotFoundError(f'Нет {path}. Запустите: python -m ts.rows')
    return build_rows_ext()


if __name__ == '__main__':
    out = build_rows_ext()
    cur = out[out['dt'] == 0]
    print(f'rows_ext: {len(out)} строк, из них dt=0: {len(cur)}; '
          f'ИК-кадров: {out.loc[out.is_ir == 1, "src_frame_id"].nunique()} '
          f'из {out["src_frame_id"].nunique()} (порог {IR_SAT_THRESHOLD})')
    print(f'записано: {paths.ART / ROWS_EXT_NAME}')
