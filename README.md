# table-state

Решение кейса 2 «Table State» (RASCAR.HACK): для каждой рамки стола на кадре предсказываем
`people, hands, cups, dishes, items` и `occ_prob`.

Правила работы, форматы артефактов и задачи треков: **[CONTRACT.md](CONTRACT.md)** (читать целиком перед началом).
План и роли: [PLAN.md](PLAN.md). Журналы экспериментов по трекам: [experiments/LOG.md](experiments/LOG.md), одной таблицей: `python tools/log_all.py`.

## Окружение

Python 3.12, ядро работает на CPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # или: uv venv --python 3.12 .venv
pip install -r requirements.txt
# трек B дополнительно: pip install -r tracks/det/requirements.txt (у каждого трека свой файл зависимостей)
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

# абляция заменой колонок: базовый файл с отдельными колонками из другого предсказания, затем gate (раздел 12.5)
python tools/swap_cols.py --base base_calibrated --donor R1c --cols occ_prob --name AB_occ

# новый источник одной командой: ts.check -> привязка -> счётчики -> замена колонок в базовом файле -> gate
python tools/try_source.py --source coco_person__v1 --cols people --base <лучший файл> --name AB_people_coco

# колонка из признаков строк (rowfeats) вместо счётчика: проверка -> сборка -> замена в базовом файле -> gate
python tools/try_rowfeat.py --rowfeats ov_counts__v1 --column ov_counts__items_gdino_and_owl --col items --base NB1 --name OV_items_and

# разбор ошибки на train по колонкам: недосчёт и пересчёт, ИК и цвет, столы с людьми и без
python tools/error_breakdown.py COMBO

# демо очереди оператора -> report/demo_queue.jpg, report/demo_queue.csv
python tracks/core/demo_queue.py

# три журнала одной таблицей
python tools/log_all.py --tail 30

# гистограммы площади боксов person -> report/person_area_hist.png
python tools/person_area_hist.py

# проверка и регистрация сабмита (только роль A, на Kaggle не отправляет)
python tools/submit_gate.py --pred artifacts/preds/<name>.csv --note "<вопрос, на который отвечает сабмит>"

# единый воспроизводимый запуск: детекции треков B и C берутся из artifacts/, 15 секунд на CPU
python run.py --data $TS_DATA --artifacts $TS_ARTIFACTS --config configs/submissions/COMBO.yaml --out submission.csv

# сборка сабмита из колонок нескольких файлов с проверкой, что конфиг даёт тот же файл
python tools/combo.py --base NB1 --take hands=AB_ft_hands_pp items=OV_items_and_pp dishes=OV_dishes_owl_max --config configs/COMBO.yaml --name COMBO
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

## Текущая основа

`COMBO` = бейзлайн организаторов + `people` с добором по нижней трети бокса (NB1, Public 0.64519) + руки `ft_yolo__v1` (0.65585)
+ вещи при согласии двух моделей трека C (0.66401) + посуда OWLv2 вне подносов (0.67045). Собирается по `configs/default.yaml`:
блок `compose` говорит, какая колонка берётся из своей сборки (`stack`), а какая из бейзлайна (`baseline`), `post.tray_mult` задаёт множитель подносов для dishes.
Колонка меняется только после проверки заменой на Public (раздел 12.5 контракта).

## Как подключить новый источник (делает A)

1. Положить файл и карточку в `artifacts/dets/<source>__v<N>.parquet` или `artifacts/rowfeats/<source>__v<N>.parquet`.
2. Проверить: `python -m ts.check dets artifacts/dets/<source>__v<N>.parquet` (или `rowfeats`).
3. Дописать имя без расширения в `sources.dets` или `sources.rowfeats` в `configs/default.yaml`. Для счётчика, который раньше был нулевым (items), включить его в `stack.counts` (`items: cur`); для признаков строк задать `stack.vote`.
4. Быстрый путь: `python tools/try_source.py --source <source>__v<N> --cols <колонки> --base <лучший файл> --name AB_<...>` делает шаги 2–3 во временном конфиге, собирает предсказание и подставляет указанные колонки в базовый файл (проверка по 12.5).
   `python tools/ablate.py` покажет полную сборку и варианты без каждого источника. Пороги confidence нового источника (по одному на счётчик) подбираются на дне подгонки автоматически.
5. Принять источник по правилу 12.3, собрать `python tracks/core/build_pred.py --name S<N>` и записать строку в журнал.

## Структура

```
ts/            общая библиотека (владелец A), ts/vendor/score.py это копия score.py организаторов
tracks/core/   трек A      tracks/det/   трек B      tracks/ov/   трек C
tools/         sanity, eval_pred, submit_gate, contact_sheet
configs/       default.yaml и замороженные конфиги сабмитов
experiments/   LOG_core.md, LOG_det.md, LOG_ov.md (журналы треков), PROPOSALS.md, ORG_ANSWERS.md
submissions/   <name>.csv + <name>.json        report/   отчёт, листы, слайды
data/, artifacts/   не в git
```

Зоны владения описаны в разделе 5 контракта: каждый трек пишет только в `tracks/<свой трек>/`,
свои файлы в `artifacts/`, добавляет строки в конец журнала своего трека `experiments/LOG_<core|det|ov>.md` и в `MODELS.md`.

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
| `victimestsate/ts-artifacts-dets` | детекции трека B: `dets/coco_person__v1`, `coco_crop__v1`, `ft_yolo__v1`, `ft_yolo_tta__v1`, `pose__v1` + карточки, `rows_ext.parquet` |
| `throddy/ts-artifacts-ov-pseudo` | трек C: `artifacts/dets/ov_gdino__v1`, `ov_owl__v1`, `artifacts/labels/ov_pseudo__v1`, `artifacts/rowfeats/ov_counts__v1` + карточки, веса моделей |
| `ts-artifacts-rowfeats` | `rowfeats/<source>__v<N>.parquet` + карточки и `__raw.jsonl` |
| `ts-models` | веса моделей и колёса зависимостей для запуска без интернета |
| `ts-code` | архив закоммиченного кода репозитория и файл `GIT_SHA` (`python tracks/det/kaggle/pack_code.py`); новая версия после каждого изменения кода |

Новая версия артефакта это новая версия датасета, номер версии указывается в карточке.
Финальный ноутбук `notebooks/final_inference.ipynb` запускается с выключенным интернетом.
Отправляет сабмиты только A: `kaggle competitions submit -c rascar-hack-case-2 -f submissions/<name>.csv -m "<name> <git sha>"`.
