# Запуск трека B на Kaggle

Обучение и тяжёлый инференс идут только на Kaggle, через Save Version (Run All), интернет выключен.

## Что подключить к ноутбуку

| Вход | Откуда | Что внутри |
|---|---|---|
| данные соревнования | `rascar-hack-case-2` | пакет организаторов |
| `ts-models` (приватный Dataset) | каталог `artifacts/ts-models/` после `python tracks/det/fetch_weights.py --wheels` | `yolo11x.pt`, `yolo11m.pt`, `yolo11x-pose.pt`, `wheels/*.whl`, `MANIFEST.json` |
| `ts-code` (приватный Dataset) | `artifacts/ts-code.zip` после `python tracks/det/kaggle/pack_code.py` | закоммиченный код репозитория и файл `GIT_SHA` |

После каждого изменения кода: коммит, `pack_code.py`, новая версия датасета `ts-code`.

## Ноутбуки

| Файл | Ячейки со скриптами | Выход в `/kaggle/working/artifacts` |
|---|---|---|
| `det_coco_sources.ipynb` | `infer.py --source coco_person`, `infer.py --source coco_crop` | `dets/coco_person__v1.*`, `dets/coco_crop__v1.*` |
| `det_train_ft_yolo.ipynb` | `build_dataset.py`, `train.py`, `infer.py --source ft_yolo` | `datasets_ft_data_v1.zip`, `models/ft_yolo_v1/`, `dets/ft_yolo__v1.*` |

Первая ячейка обоих ноутбуков находит входы по содержимому (слаги не зашиты), копирует код в
`/kaggle/working/table-state`, выставляет `TS_DATA` и `TS_ARTIFACTS` и ставит колёса без сети.

## После прогона

1. `dets/*.parquet` и карточки `*.json` выложить новой версией датасета `ts-artifacts-dets`, номер версии вписать в поле `kaggle_dataset_version` карточки.
2. `datasets_ft_data_v1.zip` выложить отдельным датасетом, `models/ft_yolo_v1/ft_yolo_v1.pt` и `.json` добавить в `ts-models`.
3. Время GPU из последней строки каждого скрипта записать в `experiments/LOG_det.md`.
4. Оборванный прогон: инференс продолжается с готовых порций (`dets/_parts`), обучение с `--resume .../weights/last.pt`. Для этого вывод прошлой версии ноутбука подключается как входные данные; порции нужно скопировать в `/kaggle/working/artifacts/dets/_parts` до запуска.
