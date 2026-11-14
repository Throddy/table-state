# Трек C: open-vocabulary детекция посуды и вещей

Grounding DINO base и OWLv2 large работают на кропах столешниц ×4 по текстовым запросам с негативами (ваза с сухоцветом, ноутбук, бутылка, салфетка, еда). В финал вошли вещи, найденные обеими моделями за занятым столом, и посуда OWLv2 на столах без подноса.

| файл | что делает |
|---|---|
| `run_ov.py` | детекции на 4964 кропах целевых кадров (`--rows targets`) или на соседних кадрах (`--rows context`, `--rows neighbours`), с возобновлением |
| `prompts/` | наборы запросов v1–v3; в финале набор v2 |
| `eval_dets.py` | пороги по train и счётчики по строкам `rowfeats/ov_counts__v1`, которые читает сборка |
| `make_pseudo.py` | псевдометки из детекций, согласованных двумя моделями, только для кадров 27–28.10 |
| `make_crops.py` | кропы train-строк для просмотра |
| `merge_dets.py` | склейка прогонов по целевым и соседним кадрам в один источник |
| `fetch_weights.py`, `pack_weights.py`, `kaggle_weights_ov.ipynb` | веса с закреплёнными ревизиями и датасет весов для запуска без сети |
| `kaggle_c2_c5.ipynb`, `kaggle_setup.py` | ноутбук, которым посчитаны детекции и счётчики финала, и настройка его окружения |
| `experiments/` | разбор ошибок на train-кадрах, таблица ошибок по дням, ноутбук для соседних кадров (не запускался) |
| `test_ov.py` | тесты трека, `python -m pytest tracks/ov` |

Детекции финала получены прогоном https://www.kaggle.com/code/throddy/notebookc45eeb95e0?scriptVersionId=369282247 на 2 × T4 и лежат в `throddy/ts-artifacts-ov-pseudo`. Повторить прогон:

1. `pip install -r requirements.txt -r tracks/ov/requirements.txt`
2. `python tracks/ov/fetch_weights.py` (один раз, с интернетом)
3. `python tracks/ov/run_ov.py --source ov_gdino --rows targets --prompt-version 2`
4. `python tracks/ov/run_ov.py --source ov_owl --rows targets --prompt-version 2 --batch 2 --fp16`
5. `python tracks/ov/eval_dets.py ov_gdino__v1 ov_owl__v1 --rowfeats 1`

На CPU один кроп занимает 12–24 с, поэтому шаги 3 и 4 рассчитаны на GPU. На T4 Grounding DINO тратит 0.77 с на кроп, OWLv2 0.54 с.
