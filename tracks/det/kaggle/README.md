# Запуск трека B на Kaggle

Обучение и тяжёлый инференс идут только на Kaggle, через Save Version (Run All), интернет выключен.

## Что подключить к ноутбуку

| Вход | Откуда | Что внутри |
|---|---|---|
| данные соревнования | `rascar-hack-case-2`; при запуске через API соревнование не подключается, поэтому используется приватная копия пакета `ts-data-private` | пакет организаторов |
| `ts-models` (приватный Dataset) | каталог `artifacts/ts-models/` после `python tracks/det/fetch_weights.py --wheels` | `yolo11x.pt`, `yolo11m.pt`, `yolo11x-pose.pt`, `wheels/*.whl`, `MANIFEST.json` |
| `ts-code` (приватный Dataset) | `artifacts/ts-code.zip` после `python tracks/det/kaggle/pack_code.py` | закоммиченный код репозитория и файл `GIT_SHA` |

После каждого изменения кода: коммит, `pack_code.py`, новая версия датасета `ts-code`.

## Ноутбуки

| Файл | Ячейки со скриптами | Выход в `/kaggle/working/artifacts` |
|---|---|---|
| `det_coco_sources.ipynb` | `infer.py --source coco_person`, `infer.py --source coco_crop` | `dets/coco_person__v1.*`, `dets/coco_crop__v1.*` |
| `det_train_ft_yolo.ipynb` | `build_dataset.py --classes v1`, `train.py`, `infer.py --source ft_yolo --half` | `datasets_ft_data_v1.zip`, `models/ft_yolo_v1/`, `dets/ft_yolo__v1.*` |
| `det_ft_yolo_tta.ipynb` | `infer.py --source ft_yolo --tta --half` | `dets/ft_yolo_tta__v1.*` |
| `det_train_ft_yolo_v2.ipynb` | `build_dataset.py --classes auto`, `train.py`, `infer.py --source ft_yolo --version 2 --half` | `datasets_ft_data_v2.zip`, `models/ft_yolo_v2/`, `dets/ft_yolo__v2.*` |
| `det_pose.ipynb` | `infer.py --source pose` (источник выключен решением A, оставлен для воспроизводимости журнала) | `dets/pose__v1.*` |

Первая ячейка всех ноутбуков находит входы по содержимому (слаги не зашиты), копирует код в
`/kaggle/working/table-state`, выставляет `TS_DATA` и `TS_ARTIFACTS` и ставит колёса без сети.

## Команды воспроизведения

Обучение v1 (4 класса: person, hand, cup, tray):

```bash
python tracks/det/build_dataset.py --name ft_data_v1 --classes v1 --pseudo-step 2
python tracks/det/train.py --config tracks/det/configs/ft_yolo_v1.yaml \
    --data $TS_ARTIFACTS/datasets/ft_data_v1 --weights <ts-models>/yolo11m.pt --hours 2.7
```

Обучение v2 (самообучение, задача B7; классы v1 и те из phone, plate, что есть в файле трека C):

```bash
python tracks/det/build_dataset.py --name ft_data_v2 --classes auto --pseudo-step 2 \
    --ft-pseudo <путь>/ft_yolo__v1.parquet --ov-pseudo <путь>/ov_pseudo__v1.parquet
python tracks/det/train.py --config tracks/det/configs/ft_yolo_v2.yaml \
    --data $TS_ARTIFACTS/datasets/ft_data_v2 --weights <ts-models>/yolo11m.pt --hours 2.7
```

Инференс (все 1718 кадров и все строки `rows_ext`, `coverage=all`):

```bash
python tracks/det/infer.py --source coco_crop --weights <ts-models>/yolo11x.pt
python tracks/det/infer.py --source ft_yolo --weights <ts-models>/ft_yolo_v1.pt --half
python tracks/det/infer.py --source ft_yolo --tta --weights <ts-models>/ft_yolo_v1.pt --half     # ft_yolo_tta__v1
```

Точка входа для `run.py` (этап 2 раздела 15.1): та же команда с `--from-cache`. Если
`$TS_ARTIFACTS/dets/<source>__v<N>.parquet` уже лежит и проходит `ts.check`, скрипт ничего не считает
и завершается с кодом 0; иначе считает источник заново.

Recall на aux-holdout (6 кадров, хватает CPU, для сравнения версий без сабмитов):

```bash
python tracks/det/eval_holdout.py --weights <путь>/ft_yolo_v1.pt [--tta]
```

## После прогона

1. `dets/*.parquet` и карточки `*.json` выложить новой версией датасета `ts-artifacts-dets`, номер версии вписать в поле `kaggle_dataset_version` карточки.
2. `datasets_ft_data_v1.zip` выложить отдельным датасетом, `models/ft_yolo_v1/ft_yolo_v1.pt` и `.json` добавить в `ts-models`.
3. Время GPU из последней строки каждого скрипта записать в `experiments/LOG_det.md`.
4. Оборванный прогон: инференс продолжается с готовых порций (`dets/_parts`), обучение с `--resume .../weights/last.pt`. Для этого вывод прошлой версии ноутбука подключается как входные данные; порции нужно скопировать в `/kaggle/working/artifacts/dets/_parts` до запуска.
