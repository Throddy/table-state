# Table State

Решение кейса 2 RASCAR.HACK «Оператор зала». Для каждой рамки стола на кадре кафе предсказываются число людей, рук, чашек, посуды и личных вещей и вероятность того, что стол занят. Отмечены два финальных сабмита: `VLM_hands_avg` с Public 0.75957 и `VLM_items` с Public 0.74357.

## Воспроизведение

Есть три пути, от самого простого к полному. Первые два собирают финал из сохранённых выходов моделей за минуту, третий пересчитывает модели на GPU.

### а) Kaggle, около минуты

1. Открыть ноутбук https://www.kaggle.com/code/throddy/table-state-final-inference.
2. Нажать Copy & Edit, затем Run All. Интернет и GPU не нужны.
3. На выходе `/kaggle/working/submission.csv` (это `VLM_hands_avg`) и `submission_VLM_items.csv`. Последняя ячейка сверяет оба файла с `submissions/` и падает при расхождении.

К ноутбуку подключены данные соревнования, код этого репозитория (`throddy/table-state-code`), кэши источников (`victimestsate/ts-artifacts-dets`, `throddy/ts-artifacts-ov-pseudo`) и вывод двух ноутбуков VLM (`throddy/ts-det-vlm-shards-0-1`, `throddy/notebook15b164940c`). Ответы VLM ноутбук склеивает из шардов сам, модель не запускается.

### б) Локально из кэша

Нужны Python 3.12, токен Kaggle в `~/.kaggle/` и принятые правила соревнования `rascar-hack-case-2`.

1. Склонировать репозиторий и поставить зависимости:

   ```bash
   git clone https://github.com/Throddy/table-state.git
   cd table-state
   python3.12 -m venv .venv          # или: uv venv --python 3.12 --seed .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Скачать пакет соревнования в `data/` и кэши источников в `artifacts/`. Скрипт сверяет пакет с `MANIFEST.sha256` организаторов и склеивает ответы VLM из шардов:

   ```bash
   python tools/fetch_artifacts.py
   ```

3. Собрать оба финала и сверить их с сохранёнными сабмитами:

   ```bash
   python tools/verify_final.py --out out
   ```

   Ожидаемый вывод: по обоим файлам `OK`, счётчики совпадают побайтно, `occ_prob` расходится не больше чем на 2.2e-16 при том же порядке строк. Такое расхождение даёт бейзлайн организаторов (ExtraTrees с `n_jobs=2` складывает деревья в двух потоках), скор от него не меняется.

4. Только сабмит, без сверки:

   ```bash
   python run.py --data data --artifacts artifacts --config configs/submissions/VLM_hands_avg.yaml --out submission.csv
   ```

Проверено на чистом клоне 15.11.2026 (MacBook на Apple Silicon, окружение через `uv`): шаг 1 занял 17 с, шаг 2 1 мин 30 с (559 МБ данных и 2.4 МБ кэшей), шаг 3 1 мин 6 с, оба файла `OK`; шаг 4 занял 17 с, его `submission.csv` совпал с `submissions/VLM_hands_avg.csv` по всем счётчикам, `occ_prob` отличается не больше чем на 1.1e-16.

### в) Полный пересчёт источников

Все модели запускались в ноутбуках Kaggle на Tesla T4 с выключенным интернетом. Веса закреплены по ревизии, ссылки в [MODELS.md](MODELS.md).

| этап | что получается | команда или ноутбук | железо | время |
|---|---|---|---|---|
| 1 | свой детектор `ft_yolo_v1`: датасет из aux-кадров и псевдометок прод-детектора, обучение, детекции `dets/ft_yolo__v1` | [tracks/det/kaggle/det_train_ft_yolo.ipynb](tracks/det/kaggle/det_train_ft_yolo.ipynb) (`build_dataset.py`, `train.py`, `infer.py`) | T4 | обучение 2.7 ч, инференс 25 мин |
| 2 | Grounding DINO и OWLv2 на кропах столов, детекции `dets/ov_gdino__v1`, `dets/ov_owl__v1` | [tracks/ov/kaggle_c2_c5.ipynb](tracks/ov/kaggle_c2_c5.ipynb), прогон https://www.kaggle.com/code/throddy/notebookc45eeb95e0?scriptVersionId=369282247 | 2 × T4 | около 64 и 45 мин |
| 3 | счётчики по строкам `rowfeats/ov_counts__v1` | `python tracks/ov/eval_dets.py ov_gdino__v1 ov_owl__v1 --rowfeats 1` | CPU | меньше минуты |
| 4 | ответы Qwen3-VL-4B по 4964 строкам, 4 шарда | [tracks/det/kaggle/det_vlm_all_a.ipynb](tracks/det/kaggle/det_vlm_all_a.ipynb) и [det_vlm_all_b.ipynb](tracks/det/kaggle/det_vlm_all_b.ipynb), прогоны https://www.kaggle.com/code/throddy/ts-det-vlm-shards-0-1 и https://www.kaggle.com/code/throddy/notebook15b164940c?scriptVersionId=369589586 | 2 × T4 на двух аккаунтах | 62–70 мин на шард, всего 4.35 GPU-часа |
| 5 | склейка ответов VLM в `rowfeats/vlm_qwen__v1` | `python tracks/det/vlm/run_vlm.py --subset all --of 4 --merge` | CPU | секунды |
| 6 | сабмит | `python run.py ...` из пути б | CPU | 15 с |

Подробности по трекам: [tracks/det/README.md](tracks/det/README.md), [tracks/ov/README.md](tracks/ov/README.md).

## Как устроено решение

Предсказание собирается по колонкам. Основа это бейзлайн организаторов, отдельные колонки заменены своими, если замена подняла Public. Метрика складывается из вкладов колонок, поэтому каждая замена проверялась отдельным сабмитом.

| колонка | источник | правило | вклад на Public |
|---|---|---|---|
| `people` | прод-детектор | правило бейзлайна и добор людей, сидящих за дальней стороной стола (нижняя треть бокса в зоне столешницы) | +0.0055 |
| `hands` | прод-детектор, свой детектор `ft_yolo_v1`, VLM | руки детекторов, затем половина шага вверх к ответу VLM; без людей рук нет, не больше двух рук на человека | +0.0107, затем +0.0160 |
| `cups` | бейзлайн организаторов | чашки прод-детектора | замены не помогли |
| `dishes` | прод-детектор, OWLv2 | max(2 × подносы, посуда OWLv2 на кропе стола) | +0.0421 за множитель 2, +0.0253 за OWLv2 |
| `items` | Grounding DINO и OWLv2, VLM | max(вещи, найденные обеими моделями за занятым столом; ответ VLM) | +0.0188, затем +0.0436 |
| `occ_prob` | бейзлайн организаторов | ExtraTrees по признакам прод-детекций | AP на Public уже 1.0 |

Второй финал `VLM_items` отличается только колонкой `hands`, в нём нет шага к VLM. История всех сабмитов в [experiments/README.md](experiments/README.md), разбор и ограничения в [report/REPORT.md](report/REPORT.md).

## Где что лежит

```
run.py                 сборка сабмита по конфигу из data/ и artifacts/
configs/               default.yaml (финальный конфиг) и submissions/ (замороженный конфиг каждого сабмита)
ts/                    библиотека: чтение данных, привязка объектов к столам, контекст ±30 с, стекер, метрика, проверки
tracks/core/           сборка предсказаний, демо очереди оператора, разбор ошибок по людям
tracks/det/            свой детектор (YOLO11m) и VLM-счётчик, ноутбуки Kaggle
tracks/ov/             Grounding DINO и OWLv2 на кропах столов, ноутбуки Kaggle
tracks/*/experiments/  разовые разборы, на которые ссылаются журналы
tools/                 скачивание кэшей, сверка финалов, оценка и регистрация сабмитов
notebooks/             финальный ноутбук Kaggle
submissions/           все собранные сабмиты с карточками
experiments/           таблица сабмитов и журналы трёх треков
report/                отчёт и демо очереди оператора
tests/                 тесты библиотеки (python -m pytest tests)
```

`data/` и `artifacts/` в git не входят, их заполняет `tools/fetch_artifacts.py`.

## Ссылки

- [report/REPORT.md](report/REPORT.md), отчёт: подход, валидация по дням, история на Public, ошибки по классам, время работы, соблюдение правил.
- [MODELS.md](MODELS.md), модели, ссылки на веса, ревизии и лицензии.
- Данные: соревнование https://www.kaggle.com/competitions/rascar-hack-case-2.
- Кэши и код на Kaggle: датасеты `throddy/table-state-code`, `victimestsate/ts-artifacts-dets`, `throddy/ts-artifacts-ov-pseudo`, ноутбуки `throddy/ts-det-vlm-shards-0-1`, `throddy/notebook15b164940c`, `throddy/table-state-final-inference`. Они приватные, доступ выдаётся на аккаунт Kaggle проверяющего.
