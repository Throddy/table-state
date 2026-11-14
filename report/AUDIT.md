# Аудит репозитория перед сдачей

Состояние `main` на коммите `a33f6c6`, 294 файла в git. Ниже найденное и что с этим сделано в ветке `release/final`.

## Мёртвый код, дубли, следы экспериментов

| что | сколько | решение |
|---|---|---|
| `ts/viz.py` | две функции-заглушки с `NotImplementedError`, нигде не импортируются | удалён |
| `tracks/det/vlm/apply_vlm.py` | дублирует `tools/vlm_files.py`, нигде не упоминается | удалён |
| рабочие копии конфигов в `configs/*.yaml` | 20 файлов рядом с замороженными копиями в `configs/submissions/`; 17 совпадают с замороженными побайтно, 3 (`VLM_items`, `VLM_people`, `VLM_both`) отличаются оформлением, а замороженная копия записана из фактически использованного конфига | перенесены в `configs/variants/`, в корне `configs/` остался `default.yaml` |
| разовые разборы трека B: `avg_cols.py`, `check_people.py`, `combo_checks.py`, `eval_holdout.py`, `make_figs.py` | на них ссылаются журнал и отчёт трека | перенесены в `tracks/det/experiments/` |
| разовые разборы трека C: `c6_errors.py`, `demo_errors.py`, `kaggle_owl_neighbours.ipynb` (ноутбук для соседних кадров, ни разу не запускался) | ссылаются журнал и тесты трека | перенесены в `tracks/ov/experiments/` |
| `tracks/core/ov_and_counts.py` | счётчик чашек для отклонённого кандидата `CUPS_ov` | перенесён в `tracks/core/experiments/` |
| `requirements-gpu.txt` | пустой шаблон, зависимости треков лежат в `tracks/det/requirements.txt` и `tracks/ov/requirements.txt` | удалён |
| `pytest.ini` | нужен был только из-за локальной копии кода в `artifacts/` | удалён, тесты запускаются командой `python -m pytest tests` |
| `notebooks/.gitkeep`, `report/.gitkeep`, `submissions/.gitkeep` | каталоги уже не пустые | удалены |
| `experiments/LOG.md` | файл-указатель на три журнала | заменён на `experiments/README.md` с таблицей всех сабмитов |

## Документы для внутренней координации

`CONTRACT.md`, `PLAN.md`, `experiments/PROPOSALS.md`, `experiments/ORG_ANSWERS.md`, `tracks/ov/HANDOFF.md`, `tracks/ov/C6_SLIDES.md` и раздел «TODO от ревью» в `tracks/ov/README.md` сохранены в ветке `team-internal` и убраны из `main`. Ссылки на контракт в коде, ноутбуках и журналах заменены пояснениями.

## Секреты и личные данные

В текущих файлах нет токенов, ключей, `kaggle.json`, почт и абсолютных путей с домашним каталогом.

В истории git (историю не переписывали) есть:

| что | где | сколько |
|---|---|---|
| путь `/Users/maksimvolkov` | ранние версии карточек `submissions/*.json` и `artifacts`-путей в командах | 106 вхождений |
| путь `/Users/klippxt` | карточки `FT2*.json` до очистки | 29 вхождений |
| личные почты авторов в метаданных коммитов | `maxim2007volkov@yandex.ru` (91 коммит), `nikishulga@gmail.com` (5 коммитов); у третьего автора адрес `noreply` GitHub | |

Токенов, ключей API и файлов `kaggle.json` в истории нет.

## Ветки, не влитые в main

| ветка | что в ней | решение |
|---|---|---|
| `det/infer-train-kit` | один коммит `1c95ebf`: пробные файлы VLM по шардам 2–3 (`VLM_*_partial`), до полного прогона VLM | не вливается, полные файлы VLM уже в `main` |
| `ov/feature-extraction`, `ov/review-fixes` | всё влито | |
| `team-internal` | внутренние документы команды | остаётся отдельной веткой |

## Зависимости

В `requirements.txt` стояли диапазоны версий. Закреплены точные версии, на которых собраны финальные файлы: numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, joblib 1.6.0, Pillow 13.0.0, pyarrow 26.0.1, PyYAML 6.0.3, плюс kaggle 2.2.4 для скачивания кэшей. В `tracks/ov/requirements.txt` версия scipy не указана и не восстанавливается по журналам; на сборку финала это не влияет, scipy нужен только OWLv2 на GPU.

## Тесты и проверки

`python -m pytest tests` 85 тестов, `python -m pytest tracks/ov` 13 тестов, все зелёные. `tools/sanity.py` воспроизводит опорные скоры бейзлайна. Два прогона `run.py` на одном коде дают одинаковые счётчики, а `occ_prob` расходится в 21–29 строках не больше чем на 2.2e-16: ExtraTrees организаторов складывает деревья в двух потоках. Порядок строк по `occ_prob` и значит скор от этого не меняются, поэтому побайтная сверка финалов делается по пяти счётчикам и по порядку `occ_prob` (`tools/verify_final.py`).
