# Трек C. Open-vocab и VLM (владелец: участник 3)

Задача трека: посуда и вещи без обучения, а также независимый счётчик по кропу от VLM.
Полные формулировки: раздел 13 [CONTRACT.md](../../CONTRACT.md). Время МСК. Статус задачи фиксируется строкой `TASK <id> done` в `experiments/LOG.md`.

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

Зависимости трека дописываются в `requirements-gpu.txt`, модели в `MODELS.md`.

## Запуск C2 и C5

Текущее состояние, измерения, оставшиеся шаги и блокеры: [HANDOFF.md](HANDOFF.md).

```bash
pip install -r requirements.txt -r tracks/ov/requirements.txt
python tracks/ov/fetch_weights.py                                  # один раз, с интернетом: веса в artifacts/models/
python tracks/ov/run_ov.py --source ov_gdino --smoke 8             # короткая проверка, файлы удаляются
python tracks/ov/run_ov.py --source ov_gdino --rows targets        # целевые кадры (dt = 0) -> dets/ov_gdino__v1
python tracks/ov/run_ov.py --source ov_owl --rows targets --batch 2 --fp16      # -> dets/ov_owl__v1
python tracks/ov/make_pseudo.py ov_gdino__v1 ov_owl__v1            # C5 -> labels/ov_pseudo__v1.parquet
python tracks/ov/eval_dets.py ov_gdino__v1 ov_owl__v1 --rowfeats 1 # оценка на train и rowfeats/ov_counts__v1
```

- Инференс только на GPU: на CPU уходит 12–24 с на кроп. `run_ov.py` возобновляемый: при обрыве или после
  `--max-minutes` повторить ту же команду.
- `--rows targets`: 4964 кропа на модель (целевые кадры); `--rows all`: 10198 кропов, все 1718 кадров с соседями.
- Отладка на train: `--rows train0 --version 91 --prompt-version 1`. Промпты: `prompts/*_v1.txt` … `*_v3.txt`,
  таблица «понятие -> класс 8.2»: `CONCEPT_CLASS` в `run_ov.py`.
- `eval_dets.py` это диагностика трека с временной привязкой; числа по контракту даёт стекер трека A.

### Kaggle

1. Datasets → New Dataset: загрузить `artifacts/kaggle/table-state-code.zip` (или архив своего коммита:
   `git archive --format=zip --prefix=table-state/ -o table-state.zip HEAD`), сделать приватным.
2. Code → New Notebook → File → Import Notebook: `tracks/ov/kaggle_c2_c5.ipynb`.
3. Add Input: датасет соревнования и датасет с кодом. Settings: Accelerator `GPU T4 x2`, Internet `On`.
4. В первой ячейке вписать `TS_GIT_SHA`. Выполнить ячейки 0–2 интерактивно, по таблицам выбрать `PROMPT_VERSION`.
5. Поставить `RUN_TRAIN_COMPARE = False`, Save Version → Save & Run All.
6. Из Output скачать `artifacts/labels/ov_pseudo__v1.*`, `artifacts/rowfeats/ov_counts__v1.*`, `artifacts/dets/ov_*__v1.*`;
   Output сохранить как версию датасета команды. Каталог `artifacts/models` из Output сохранить как датасет с весами:
   с ним следующие запуски идут без интернета.
