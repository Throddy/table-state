"""Общие функции трека B: Ultralytics без сети, NMS, запись порциями, артефакт детекций (8.1) и карточка (8.6)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts import FRAME_H, FRAME_W, SEED, paths  # noqa: E402
from ts.check import check_dets  # noqa: E402
from ts.io import DET_COLUMNS  # noqa: E402

OWNER = 'B'
CONF_MIN = 0.05   # раздел 8.1: сохраняем от 0.05, порог выбирает стекер
NMS_IOU = 0.5     # раздел 8.1: NMS по классу внутри источника
TRAIN_DATES = ('2026-10-25', '2026-10-26')
TEST_DATES = ('2026-10-27', '2026-10-28')
# Классы своего детектора v1, порядок это номера классов YOLO. Решение роли A от 13.11: v1 учится на 4 классах,
# plate, phone, personal_item, headphones добавляются в v2 после меток C5 (на псевдо-кадрах они без меток).
FT_CLASSES = ['person', 'hand', 'cup', 'tray']
# Сырые детекции одной порции: det_id и source добавляются при сборке артефакта.
RAW_COLUMNS = ['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'from_row_id', 'from_dt']


# ---------- окружение ----------

def require_data() -> None:
    """Остановиться с понятным сообщением, если пакета организаторов нет."""
    if not (paths.DATA / 'frames.csv').is_file():
        raise SystemExit(f'Нет пакета организаторов в {paths.DATA} (нет frames.csv). Локально распакуйте пакет '
                         'в data/, на Kaggle выставьте TS_DATA и TS_ARTIFACTS по README до запуска скрипта.')


def setup_ultralytics() -> str:
    """Ultralytics без сети (разделы 6 и 16.6). Вызывать до первого `import ultralytics`.

    `YOLO_OFFLINE` выключает проверки обновлений, загрузку шрифтов и аналитику, `sync=False` и флаги
    интеграций это то же самое, что `yolo settings sync=False ...`. Возвращает версию пакета.
    """
    os.environ['YOLO_OFFLINE'] = 'true'
    import ultralytics
    from ultralytics.utils import SETTINGS
    SETTINGS.update({'sync': False, 'clearml': False, 'comet': False, 'dvc': False, 'mlflow': False,
                     'raytune': False, 'tensorboard': False, 'wandb': False})
    return ultralytics.__version__


def load_model(weights: str | Path):
    """Модель Ultralytics из явного файла весов. Автоскачивание запрещено, поэтому файл обязан существовать."""
    weights = Path(weights)
    if not weights.is_file():
        raise SystemExit(f'Нет файла весов {weights}. Автоскачивание запрещено: файл кладётся заранее '
                         'скриптом tracks/det/fetch_weights.py (датасет ts-models).')
    from ultralytics import YOLO
    return YOLO(str(weights))


def seed_everything(seed: int = SEED) -> None:
    """Сид 419 и детерминированный cudnn (раздел 6)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_device() -> str:
    import torch
    return '0' if torch.cuda.is_available() else 'cpu'


def device_name(device: str) -> str:
    import torch
    if device not in ('cpu', 'mps') and torch.cuda.is_available():
        return torch.cuda.get_device_name(int(str(device).split(',')[0]))
    return str(device)


def git_sha() -> str:
    """Короткий sha коммита. На Kaggle (код без .git) читается файл GIT_SHA, который кладёт kaggle/pack_code.py."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ''
    if (ROOT / '.git').exists():
        sha = run('rev-parse', '--short', 'HEAD')
        if sha:
            return sha + ('-dirty' if run('status', '--porcelain') else '')
    marker = ROOT / 'GIT_SHA'
    return marker.read_text(encoding='utf-8').strip() if marker.is_file() else 'no-commit'


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sha1_ids(ids) -> str:
    """sha1 от отсортированного списка id, соединённого переводом строки (поле `frames_processed.sha1`)."""
    return hashlib.sha1('\n'.join(sorted(str(i) for i in ids)).encode('utf-8')).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def command_line() -> str:
    """Команда запуска для карточки: путь скрипта относительно корня репозитория и аргументы как есть."""
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(ROOT)
    except ValueError:
        pass
    return ' '.join(['python', str(script), *sys.argv[1:]])


# ---------- боксы ----------

def clip_to_frame(boxes: np.ndarray) -> np.ndarray:
    """Обрезать боксы `(n, 4)` по кадру 1280×720."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, FRAME_W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, FRAME_H)
    return b


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU одного бокса с каждым из `boxes` (формат `x1, y1, x2, y2`)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    iw = (np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])).clip(min=0)
    ih = (np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])).clip(min=0)
    inter = iw * ih
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = NMS_IOU) -> np.ndarray:
    """Жадный NMS отдельно по каждому классу: бокс с IoU ≥ `iou_thr` к более уверенному боксу того же класса убирается.

    Возвращает индексы оставленных боксов по убыванию confidence.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=object)
    keep: list[int] = []
    for name in sorted(set(classes.tolist())):
        idx = np.flatnonzero(classes == name)
        order = idx[np.argsort(-scores[idx], kind='stable')]
        while len(order):
            best, rest = order[0], order[1:]
            keep.append(int(best))
            order = rest[iou_one_to_many(boxes[best], boxes[rest]) < iou_thr]
    return np.array(sorted(keep, key=lambda i: (-scores[i], i)), dtype=int)


# ---------- запись порциями ----------

class PartStore:
    """Результат по порциям с пропуском уже обработанных (раздел 16.9).

    Каждая порция это `part_<n>.parquet` и отметка `part_<n>.done.json` со списком обработанных единиц
    (кадров или строк). Отметка пишется последней, поэтому оборванная порция пересчитывается заново.
    `fingerprint` защищает от смешивания порций, посчитанных с разными параметрами.
    """

    def __init__(self, folder: Path, fingerprint: dict):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, indent=1)
        marker = self.folder / 'fingerprint.json'
        if marker.is_file():
            if marker.read_text(encoding='utf-8') != text:
                raise SystemExit(f'В {self.folder} лежат порции прогона с другими параметрами. '
                                 'Удалите каталог или поднимите --version.')
        else:
            marker.write_text(text, encoding='utf-8')

    def _files(self, index: int) -> tuple[Path, Path]:
        return self.folder / f'part_{index:05d}.parquet', self.folder / f'part_{index:05d}.done.json'

    def is_done(self, index: int) -> bool:
        return self._files(index)[1].is_file()

    def save(self, index: int, table: pd.DataFrame, unit_ids: list[str]) -> None:
        data, done = self._files(index)
        tmp = data.with_suffix('.tmp')
        table.to_parquet(tmp, index=False)
        tmp.replace(data)
        tmp = done.with_suffix('.tmp')
        tmp.write_text(json.dumps(unit_ids), encoding='utf-8')
        tmp.replace(done)

    def load(self, n_parts: int) -> tuple[pd.DataFrame, list[str]]:
        """Все порции `0..n_parts-1`: таблица и список обработанных единиц. Падает, если порции не хватает."""
        tables, units = [], []
        for index in range(n_parts):
            data, done = self._files(index)
            if not done.is_file():
                raise SystemExit(f'Порция {index} из {n_parts} не посчитана ({self.folder}). Запустите команду ещё раз.')
            units.extend(json.loads(done.read_text(encoding='utf-8')))
            table = pd.read_parquet(data)
            if len(table):
                tables.append(table)
        return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), units


# ---------- артефакт детекций ----------

def raw_table(rows: list[tuple]) -> pd.DataFrame:
    """Таблица сырых детекций порции с устойчивыми типами колонок (пустая порция тоже записывается)."""
    table = pd.DataFrame(rows, columns=RAW_COLUMNS)
    for c in ['frame_id', 'class_name']:
        table[c] = table[c].astype(str)
    for c in ['confidence', 'x1', 'y1', 'x2', 'y2']:
        table[c] = table[c].astype(float)
    table['from_row_id'] = table['from_row_id'].astype(object)
    table['from_dt'] = table['from_dt'].astype('Int64')
    return table


def finalize_dets(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Сырые детекции всех порций -> таблица формата 8.1 с `det_id = <source>_<frame_id>_<n>`.

    Номер `n` идёт внутри кадра в устойчивом порядке: строка кропа, `dt`, класс, убывание confidence,
    координаты. Порядок не зависит от разбиения на порции и батчи.
    """
    if not len(raw):
        raw = raw_table([])
    df = raw.copy()
    df['source'] = source
    df['_row'] = df['from_row_id'].astype(object).where(df['from_row_id'].notna(), '').astype(str)
    df['_dt'] = pd.to_numeric(df['from_dt'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values(['frame_id', '_row', '_dt', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'],
                        ascending=[True, True, True, True, False, True, True, True, True], kind='stable')
    df['det_id'] = source + '_' + df['frame_id'].astype(str) + '_' + df.groupby('frame_id', sort=False).cumcount().astype(str)
    df['from_row_id'] = df['from_row_id'].astype(object).where(df['from_row_id'].notna(), None)
    df['from_dt'] = pd.to_numeric(df['from_dt'], errors='coerce').astype('Int64')
    for c in ['confidence', 'x1', 'y1', 'x2', 'y2']:
        df[c] = df[c].astype(float)
    return df[DET_COLUMNS].reset_index(drop=True)


def write_dets_artifact(dets: pd.DataFrame, source: str, version: int, card: dict) -> Path:
    """Записать `dets/<source>__v<N>.parquet` и карточку, затем проверить `ts.check.check_dets`.

    Артефакты неизменяемы (раздел 5): существующий файл не перезаписывается. Файл, не прошедший
    проверку, удаляется (по контракту он не существует), порции при этом остаются на месте.
    """
    import pyarrow as pa

    name = f'{source}__v{version}'
    path = paths.art('dets', f'{name}.parquet')
    if path.exists():
        raise SystemExit(f'{path} уже существует. Артефакты неизменяемы: поднимите --version.')
    schema = pa.schema([('det_id', pa.string()), ('frame_id', pa.string()), ('source', pa.string()),
                        ('class_name', pa.string()), ('confidence', pa.float64()),
                        ('x1', pa.float64()), ('y1', pa.float64()), ('x2', pa.float64()), ('y2', pa.float64()),
                        ('from_row_id', pa.string()), ('from_dt', pa.int64())])
    dets[DET_COLUMNS].to_parquet(path, index=False, schema=schema)
    card_path = path.with_suffix('.json')
    card_path.write_text(json.dumps({**card, 'name': name}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        check_dets(path)
    except Exception:
        path.unlink(missing_ok=True)
        card_path.unlink(missing_ok=True)
        raise
    return path


def dets_card(models: list[str], inputs: list[str], coverage: str, frame_ids, notes: str, **extra) -> dict:
    """Карточка артефакта детекций по разделу 8.6 (поле `name` добавляет `write_dets_artifact`)."""
    frame_ids = sorted(set(frame_ids))
    return {
        'name': '',
        'kind': 'dets',
        'owner': OWNER,
        'created_utc': utc_now(),
        'git_sha': git_sha(),
        'command': command_line(),
        'models': models,
        'inputs': inputs,
        'coverage': coverage,
        'frames_processed': {'n': len(frame_ids), 'sha1': sha1_ids(frame_ids)},
        'manual_labels_used': False,
        'uses_train_counters': False,
        'notes': notes,
        **extra,
    }
