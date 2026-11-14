# Модели

Все модели открытые, веса закреплены по файлу релиза, коммиту Hugging Face или sha256. При инференсе сеть не используется.

| модель | роль в решении | веса | ревизия | лицензия |
|---|---|---|---|---|
| Ultralytics YOLO11m (COCO) | стартовые веса своего детектора | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt | релиз v8.4.0, sha256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` | AGPL-3.0 |
| `ft_yolo_v1`, YOLO11m, дообученный командой | руки в финале | Kaggle Dataset `victimestsate/ts-models`, файл `ft_yolo_v1.pt` | sha256 `5bf40b516688e9e448fcb7ab8c9de69d438eb39a85055e89f620a4ded50a3235` | AGPL-3.0, производная от весов Ultralytics |
| Grounding DINO base | вещи в финале (вместе с OWLv2) | https://huggingface.co/IDEA-Research/grounding-dino-base | коммит `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` | Apache-2.0 |
| OWLv2 large patch14 ensemble | вещи и посуда в финале | https://huggingface.co/google/owlv2-large-patch14-ensemble | коммит `95e26936e865f87db1742128404b3c035d47d89d` | Apache-2.0 |
| Qwen3-VL-4B-Instruct | вещи и руки в финале | Kaggle Models `qwen-lm/qwen-3-vl/Transformers/4b-instruct/1`, исходник https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | по sha256 совпадает с коммитом `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Apache-2.0 |
| Ultralytics YOLO11x (COCO) | проверялся для людей и посуды, в финал не вошёл | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt | релиз v8.4.0, sha256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` | AGPL-3.0 |
| Ultralytics YOLO11x-pose | проверялся для рук, в финал не вошёл | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-pose.pt | релиз v8.4.0, sha256 `013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8` | AGPL-3.0 |
| `ft_yolo_v2`, самообучение на предсказаниях v1 | проверялся для рук, в финал не вошёл | Kaggle Dataset `victimestsate/ts-models`, файл `ft_yolo_v2.pt` | sha256 `487695494bb436832882842c7b2c8f1fca431a60b5ab92772e3a7b7cd8ef5abb` | AGPL-3.0, производная от весов Ultralytics |

Сборка финала на CPU использует ещё бейзлайн организаторов (`baseline.py` из пакета соревнования, ExtraTrees из scikit-learn) и их `score.py`, копия которого лежит в `ts/vendor/`.

## Где посчитаны выходы моделей

Финальная сборка читает выходы моделей из сохранённых файлов. Каждый файл получен ноутбуком Kaggle на Tesla T4 с выключенным интернетом.

| файл | модели | ноутбук | где лежит |
|---|---|---|---|
| `dets/ft_yolo__v1` | `ft_yolo_v1` | [tracks/det/kaggle/det_train_ft_yolo.ipynb](tracks/det/kaggle/det_train_ft_yolo.ipynb) | `victimestsate/ts-artifacts-dets` |
| `dets/ov_gdino__v1`, `dets/ov_owl__v1`, `rowfeats/ov_counts__v1` | Grounding DINO, OWLv2 | https://www.kaggle.com/code/throddy/notebookc45eeb95e0?scriptVersionId=369282247 | `throddy/ts-artifacts-ov-pseudo` |
| ответы VLM, шарды 0–1 | Qwen3-VL-4B | https://www.kaggle.com/code/throddy/ts-det-vlm-shards-0-1 | вывод ноутбука |
| ответы VLM, шарды 2–3 | Qwen3-VL-4B | https://www.kaggle.com/code/throddy/notebook15b164940c?scriptVersionId=369589586 | вывод ноутбука |
