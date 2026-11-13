# table-state

Решение кейса 2 «Table State» (RASCAR.HACK): для каждой рамки стола на кадре предсказываем
`people, hands, cups, dishes, items` и `occ_prob`.

Правила работы, форматы артефактов и задачи треков: **[CONTRACT.md](CONTRACT.md)** (читать целиком перед началом).
План и роли: [PLAN.md](PLAN.md). Журнал экспериментов: [experiments/LOG.md](experiments/LOG.md).

## Окружение

Python 3.12, ядро работает на CPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # или: uv venv --python 3.12 .venv
pip install -r requirements.txt
# треки B и C дополнительно: pip install -r requirements-gpu.txt
```

## Данные и переменные окружения

Распакуйте пакет организаторов как есть в `data/` (в git не попадает, файлы не переименовывать).
В `data/` должны лежать `frames.csv`, `tables.csv`, `train_labels.csv`, `im_*.jpg`, `aux_*.jpg` и остальное.

| Переменная | По умолчанию | Что это |
|---|---|---|
| `TS_DATA` | `<корень репозитория>/data` | пакет организаторов |
| `TS_ARTIFACTS` | `<корень репозитория>/artifacts` | общие артефакты (раздел 8 контракта), в git не попадают |

Абсолютных путей в коде нет: всё идёт через `ts.paths.DATA`, `ts.paths.ART`, `ts.paths.art(kind, name)`.

## Проверка после клонирования

```bash
python data/check_data.py --data data   # валидатор пакета от организаторов
python -m ts.rows                       # строит artifacts/rows_ext.parquet (около 5 секунд)
python tools/sanity.py                  # факты раздела 4 и опорные скоры таблицы 4.5
python -m pytest -q                     # тесты ts
```

`tools/sanity.py` должен напечатать `SANITY OK` и таблицу: detector 0.489 / 0.490, calibrated 0.551, нули 0.020.

## Команды

```bash
# оценка файла предсказаний по схеме 12.2 (oof_all, f25, f26, robust, ИК, bootstrap)
python tools/eval_pred.py artifacts/preds/<name>.csv

# проверка артефакта по контракту
python -m ts.check dets artifacts/dets/<source>__v<N>.parquet
python -m ts.check rowfeats artifacts/rowfeats/<source>__v<N>.parquet
python -m ts.check pred artifacts/preds/<name>.csv
python -m ts.check rows_ext

# контактные листы train-кадров (рамки, слоты, прод-детекции, ручные счётчики) -> report/sheets/
python tools/contact_sheet.py [--frames im_...] [--context]

# опорное предсказание бейзлайна организаторов (режим calibrated)
python tracks/core/base_calibrated.py

# сборка предсказания по конфигу (источники из sources, привязка, контекст, стекер, постобработка)
python tracks/core/build_pred.py --name <name> [--config configs/default.yaml]

# абляция «минус один источник» и проверка отдельных параметров (честный out-of-fold по дням)
python tools/ablate.py [--set post.items_prior=false] [--set temporal.people_max_area=36000]

# гистограммы площади боксов person -> report/person_area_hist.png
python tools/person_area_hist.py

# проверка и регистрация сабмита (только роль A, на Kaggle не отправляет)
python tools/submit_gate.py --pred artifacts/preds/<name>.csv --note "<вопрос, на который отвечает сабмит>"

# единый воспроизводимый запуск (задача A7, пока заглушка)
python run.py --data $TS_DATA --artifacts $TS_ARTIFACTS --config configs/submissions/<name>.yaml --out submission.csv
```

## Библиотека `ts` в двух словах

```python
from ts import io, paths
from ts.rows import load_rows_ext
from ts.crops import crop_table, to_frame_coords

rows = load_rows_ext()                       # индекс строк, формат 8.3
r = rows.iloc[0]
crop, meta = crop_table(r.src_frame_id, (r.x1, r.y1, r.x2, r.y2), kind='top', scale=4.0, slot=r.slot)
# boxes: np.ndarray (n, 4) в пикселях кропа -> пиксели кадра 1280×720
boxes_frame = to_frame_coords(boxes, meta)
out = paths.art('dets', 'ov_gdino__v1.parquet')   # куда писать артефакт
```

Готово: `paths, io, slots, rows, geometry, crops, metric, cv, check, assign, temporal, features, stack, post`.
Заглушка: `viz`.

## Как подключить новый источник (делает A)

1. Положить файл и карточку в `artifacts/dets/<source>__v<N>.parquet` или `artifacts/rowfeats/<source>__v<N>.parquet`.
2. Проверить: `python -m ts.check dets artifacts/dets/<source>__v<N>.parquet` (или `rowfeats`).
3. Дописать имя без расширения в `sources.dets` или `sources.rowfeats` в `configs/default.yaml`. Для счётчика, который раньше был нулевым (items), включить его в `stack.counts` (`items: cur`); для признаков строк задать `stack.vote`.
4. `python tools/ablate.py` покажет полную сборку и варианты без каждого источника. Пороги confidence нового источника (по одному на счётчик) подбираются на дне подгонки автоматически.
5. Принять источник по правилу 12.3, собрать `python tracks/core/build_pred.py --name S<N>` и записать строку в журнал.

## Структура

```
ts/            общая библиотека (владелец A), ts/vendor/score.py это копия score.py организаторов
tracks/core/   трек A      tracks/det/   трек B      tracks/ov/   трек C
tools/         sanity, eval_pred, submit_gate, contact_sheet
configs/       default.yaml и замороженные конфиги сабмитов
experiments/   LOG.md, PROPOSALS.md, ORG_ANSWERS.md
submissions/   <name>.csv + <name>.json        report/   отчёт, листы, слайды
data/, artifacts/   не в git
```

Зоны владения описаны в разделе 5 контракта: каждый трек пишет только в `tracks/<свой трек>/`,
свои файлы в `artifacts/`, добавляет строки в конец `experiments/LOG.md` и `MODELS.md`.

## Kaggle

Соревнование: <https://www.kaggle.com/competitions/rascar-hack-case-2/overview> (слаг `rascar-hack-case-2`).
Эталонный прогон решения идёт в Kaggle Notebook (раздел 6.1 контракта).

Каталог данных в ноутбуке ищется так же, как в `baseline.ipynb` организаторов: по наличию
`train_labels.csv`, `tables.csv` и `data_card.json` внутри `/kaggle/input`. Слаг в коде не хардкодится.
Переменные надо выставить до первого импорта `ts`:

```python
import os, sys
sys.path.insert(0, '/kaggle/working/table-state')        # куда положен код репозитория
from pathlib import Path
candidates = [p.parent for p in Path('/kaggle/input').rglob('train_labels.csv')
              if (p.parent / 'tables.csv').is_file() and (p.parent / 'data_card.json').is_file()]
assert len(candidates) == 1, 'Подключите датасет соревнования или распакуйте архив и укажите TS_DATA'
os.environ['TS_DATA'] = str(candidates[0])
os.environ['TS_ARTIFACTS'] = '/kaggle/working/artifacts'
```

То же самое делает `ts.paths.find_data_dir()`. Скрипты запускаются одной ячейкой:
`!python tracks/<трек>/<скрипт>.py ...`.

Общие артефакты команды это приватные Kaggle Datasets с доступом всем участникам:

| Датасет | Что внутри |
|---|---|
| `ts-artifacts-dets` | `dets/<source>__v<N>.parquet` + карточки, `rows_ext.parquet` |
| `ts-artifacts-rowfeats` | `rowfeats/<source>__v<N>.parquet` + карточки и `__raw.jsonl` |
| `ts-models` | веса моделей и колёса зависимостей для запуска без интернета |

Новая версия артефакта это новая версия датасета, номер версии указывается в карточке.
Финальный ноутбук `notebooks/final_inference.ipynb` запускается с выключенным интернетом.
Отправляет сабмиты только A: `kaggle competitions submit -c rascar-hack-case-2 -f submissions/<name>.csv -m "<name> <git sha>"`.
