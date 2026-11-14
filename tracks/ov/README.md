# Трек C. Open-vocab и VLM (владелец: Никита)

Задача трека: посуда и вещи без обучения, а также независимый счётчик по кропу от VLM.
Полные формулировки: раздел 13 [CONTRACT.md](../../CONTRACT.md). Время МСК. Статус задачи фиксируется строкой `TASK <id> done` в журнале трека `experiments/LOG_ov.md` (в другие журналы не писать).

| ID | Срок | Выход | DoD |
|---|---|---|---|
| C1 | пт 03:00 | `tracks/ov/make_crops.py`, кропы `top` ×4 и `wide` ×2.5 для 231 train-строки, контактный лист кропов занятых столов с ручными счётчиками | глазами видно чашки и телефоны на кропах; если нет, увеличить масштаб или применить резкость и сообщить A |
| C2 | первая версия пт 13:00, все строки пт 16:00 | `dets/ov_gdino__v1.parquet`, `dets/ov_owl__v1.parquet` (confidence от 0.05), промпты в `tracks/ov/prompts/` с негативами | на train при лучшем пороге items ≥ 0.30 и dishes ≥ 0.40 на `oof_all`, оба лучше нулей на каждом из дней; иначе разбор причин в журнале |
| C3 | train пт 14:00, полный прогон к сб 02:00 | `rowfeats/vlm_qwen__v1.parquet` + `__raw.jsonl`, промпт в `tracks/ov/prompts/vlm_v1.txt` | 4964 строки `dt = 0`, доля `__ok = 0` меньше 1%, карточка с ревизией весов; тест запускается только после проверки на 231 train-строке |
| C4 (необяз.) | пт 23:00 | `dets/ov_sam3__v1.parquet`; только если доступ к весам получен до пт 12:00 и лицензия подтверждена | по правилу 12.3 лучше лучшего из C2 хотя бы по одному классу |
| C5 | пт 17:00, обновление сб 10:00 | `artifacts/labels/ov_pseudo__v1.parquet` в формате 8.1, только уверенные согласованные детекции | B подтверждает, что файл читается его `build_dataset.py` |
| C6 | сб 18:00 | `report/demo_queue.jpg`, `report/demo_queue.csv`, 4–6 показательных кадров, слайды защиты, вычитка `MODELS.md` | на слайдах есть таблица ошибок по классам на двух днях и раздел «ограничения» |

Что уже есть в `ts` для трека:

- `ts.crops.crop_table(src_frame_id, box, kind, scale, slot)` возвращает кроп и `CropMeta`; `to_frame_coords` переводит боксы кропа в пиксели кадра. Своя временная реализация кропов не нужна.
- `ts.rows.load_rows_ext()`: индекс строк с соседями (`dt`), рамкой, слотом и `is_ir`.
- `python -m ts.check dets <файл>` и `python -m ts.check rowfeats <файл>`: проверка формата перед публикацией артефакта.

Зависимости трека кладутся в `tracks/ov/requirements.txt` (с точными версиями), модели в `MODELS.md`.

## Запуск C2 и C5

Состояние на сб 14.11: `ov_gdino__v1`, `ov_owl__v1`, `ov_pseudo__v1`, `ov_counts__v1` посчитаны на Kaggle GPU
(промпт v2, 4964 кропа на модель, только `dt = 0`) и лежат в Kaggle Dataset `throddy/ts-artifacts-ov-pseudo`.
Проверка файлов и результаты на Public: `experiments/LOG_core.md` (сб 14.11 04:55–05:20), записи трека:
`experiments/LOG_ov.md`. Коротко о состоянии и оставшихся шагах: [HANDOFF.md](HANDOFF.md).

```bash
pip install -r requirements.txt -r tracks/ov/requirements.txt
python tracks/ov/fetch_weights.py                                  # один раз, с интернетом: веса в artifacts/models/
python tracks/ov/run_ov.py --source ov_gdino --smoke 8             # короткая проверка, файлы удаляются
python tracks/ov/run_ov.py --source ov_gdino --rows targets --prompt-version 2                  # -> dets/ov_gdino__v1
python tracks/ov/run_ov.py --source ov_owl --rows targets --prompt-version 2 --batch 2 --fp16   # -> dets/ov_owl__v1
python tracks/ov/make_pseudo.py ov_gdino__v1 ov_owl__v1            # C5 -> labels/ov_pseudo__v1.parquet (боксы 27–28.10)
python tracks/ov/eval_dets.py ov_gdino__v1 ov_owl__v1 --rowfeats 1 # оценка на train и rowfeats/ov_counts__v1
pytest tracks/ov                                                   # тесты трека
```

- Инференс только на GPU: на CPU уходит 12–24 с на кроп. `run_ov.py` возобновляемый: при обрыве или после
  `--max-minutes` повторить ту же команду.
- `--rows targets`: 4964 кропа на модель (целевые кадры), карточка `coverage: targets`; `--rows all`: 10198 кропов,
  все 1718 кадров с соседями, `coverage: all`.
- `make_pseudo.py` пишет боксы только на кадрах дней `--dates` (по умолчанию 27–28.10): пороги подобраны по
  счётчикам 25–26.10, боксы этих дней в обучение не отдаются. Дни записаны в карточке (`dates`).
- Отладка на train: `--rows train0 --version 91 --prompt-version 1`. Промпты: `prompts/*_v1.txt` … `*_v3.txt`,
  таблица «понятие -> класс 8.2»: `CONCEPT_CLASS` в `run_ov.py`.
- `eval_dets.py` это диагностика трека с временной привязкой; числа по контракту даёт стекер трека A.

### Kaggle

1. Datasets → New Dataset: загрузить `artifacts/kaggle/table-state-code.zip` (или архив своего коммита:
   `git archive --format=zip --prefix=table-state/ -o table-state.zip HEAD`), сделать приватным.
2. Code → New Notebook → File → Import Notebook: `tracks/ov/kaggle_c2_c5.ipynb`.
3. Add Input: датасет соревнования и датасет с кодом. Settings: Accelerator `GPU T4 x2`, Internet `On`.
4. В первой ячейке вписать `TS_GIT_SHA`. `PROMPT_VERSION = 2` уже выбран по train; чтобы сравнить промпты заново,
   поставить `RUN_TRAIN_COMPARE = True` и выполнить ячейки 0–2 интерактивно.
5. Save Version → Save & Run All.
6. Из Output скачать `artifacts/labels/ov_pseudo__v1.*`, `artifacts/rowfeats/ov_counts__v1.*`, `artifacts/dets/ov_*__v1.*`;
   Output сохранить как версию датасета команды. Каталог `artifacts/models` из Output сохранить как датасет с весами:
   с ним следующие запуски идут без интернета.

### Соседние кадры `dt = ±30` (эксперимент, на GPU не запускался)

Нужен для медианы по трём кадрам (раздел 10 контракта): `ts.temporal` строит `prev/next/med` только для источника
с `coverage: all`. Целевые кадры не пересчитываются: считаются 5234 недостающих кропа на модель (у 95 строк из 4964
сосед один, у остальных два) и склеиваются с готовым `v1`.

```bash
python tracks/ov/run_ov.py --source ov_gdino --rows context --version 20 --prompt-version 2
python tracks/ov/run_ov.py --source ov_owl --rows context --version 20 --prompt-version 2 --batch 2 --fp16
python tracks/ov/merge_dets.py ov_gdino__v1 ov_gdino__v20 --version 2     # -> dets/ov_gdino__v2, coverage: all
python tracks/ov/merge_dets.py ov_owl__v1 ov_owl__v20 --version 2         # -> dets/ov_owl__v2, coverage: all
```

На Kaggle: тот же ноутбук, Add Input ещё и датасет с готовыми `v1`, в первой ячейке `RUN_CONTEXT = True`,
Save & Run All (ячейка 4б). `merge_dets.py` отказывается клеить части с разными моделью, промптом или параметрами.
Проверка источника потом командой трека A: `python tools/try_source.py --source ov_owl__v2 --cols dishes ...`.

### Материалы для защиты (C6)

Текст слайдов и источники чисел: [C6_SLIDES.md](C6_SLIDES.md). Очередь оператора (`report/demo_queue.*`) и два
сложных train-кадра на том же листе собирает трек A (`python tracks/core/demo_queue.py`), трек C свою версию демо
не делает. Вспомогательный лист ошибок для любого файла с out-of-fold предсказаниями на 231 train-строке
(в файле сабмита их нет), в демо не входит:

```bash
python tracks/ov/demo_errors.py artifacts/preds/COMBO.csv   # report/demo_errors.jpg и report/demo_errors.csv
```

## TODO от ревью

Замечания роли A по pull request #3 (C2 и C5), 13.11. PR влит, пункты не блокирующие, но их нужно закрыть до первого полного прогона.

1. `make_pseudo.py`: добавить фильтр по дням, по умолчанию только 27–28.10, и записать дни в карточку. Сейчас боксы пишутся на всех кадрах входных файлов, включая 25–26.10, а пороги подбираются по счётчикам train. Утечки пока нет только потому, что `tracks/det/build_dataset.py` берёт псевдо-кадры тестовых дней.
2. Карточка при прогоне `--rows targets` (839 целевых кадров) должна иметь `"coverage": "targets"`, а не `train_only`: в файле есть тестовые кадры. `ts.check` принимает `targets` при `frames_processed.n = 839`.
3. В карточке указать, что источник посчитан только для `dt = 0`: контекстные признаки `prev/next/med` для него недоступны.
4. Журнал трека: строки о C2 и C5 пишутся в `experiments/LOG_ov.md` (в общий `experiments/LOG.md` больше не пишем).
5. Префикс коммитов по контракту `ov: ...`, а не `feat(ov): ...`. Перед следующим PR сделать `git pull`: ветка отставала от main, были конфликты в `MODELS.md` и в этом файле.
6. `HANDOFF.md` устарел («файлы не закоммичены»): актуальное перенести сюда, файл удалить или обновить.
7. Посуда: тарелка или миска, перекрытая подносом той же строки (IoU >= 0.3 или центр внутри бокса подноса), в dishes не считается, потому что поднос уже идёт с множителем 2 (`post.tray_mult`, подтверждено на Public). Это делает `ts.temporal`, в `run_ov.py` ничего фильтровать не нужно; при оценке в `eval_dets.py` сравнивать с базой «2 × подносы», а не «1 × подносы».
8. Проверка источника делается одной командой роли A: `python tools/try_source.py --source ov_owl__v1 --cols dishes --base NB1 --name OV_dishes` (для вещей `--cols items`).
9. `make_pseudo.py`: покрытие выходного файла выводить из самого слабого входного (`train_only` < `targets` < `all`), а не из двух значений; `eval_dets.py --rowfeats` должен требовать `coverage` из `all` или `targets`. `ts.check` теперь сверяет `frames_processed.sha1` с sha1 отсортированного списка кадров заявленного покрытия (id через перевод строки) и отклоняет детекции на кадрах вне него.

### Статус замечаний (трек C, сб 14.11)

| № | Статус | Что сделано |
|---|---|---|
| 1 | закрыто | `make_pseudo.py`: `--dates`, по умолчанию 27–28.10 (`TEST_DATES`); в карточке `dates` и `boxes_dropped_other_dates`. Пороги и листы train не изменились |
| 2 | закрыто в коде | `run_ov.py`: `--rows targets` даёт `coverage: targets` (`coverage_of`); тест сверяет список кадров и sha1 с `ts.check`. В уже опубликованных карточках `v1` осталось `train_only`: файлы не пересоздавались |
| 3 | закрыто в коде | `run_ov.py`: в карточке `dt`, `context_features` и фраза в `notes` о том, что `prev/next/med` недоступны |
| 4 | закрыто | строки о C2, C5 и этих правках добавлены в `experiments/LOG_ov.md` |
| 5 | за Никитой | префикс коммитов `ov: ...` и `git pull` перед PR: git-операции делает только Никита |
| 6 | закрыто | `HANDOFF.md` переписан под текущее состояние, актуальное перенесено в этот файл |
| 7 | закрыто | `eval_dets.py`: база сравнения dishes это `2 × подносы` (на train 0.421), `1 × подносы` печатается для справки; `run_ov.py` ничего не фильтрует |
| 8 | к сведению | проверка источника командой трека A, со стороны трека C правок не требует |
| 9 | закрыто | `make_pseudo.py`: покрытие от самого слабого входа (`weakest_coverage`); `eval_dets.py --rowfeats` требует `coverage` из `all` или `targets` |

Тесты: `pytest tracks/ov` (11 тестов). Перегенерация `ov_pseudo__v1` не требуется: `tracks/det/build_dataset.py`
и так берёт псевдометки только с кадров 27–28.10, новый фильтр убрал бы из файла лишь боксы 39 train-кадров.
