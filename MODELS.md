# MODELS.md

Каждая модель заносится сюда в момент первого использования (раздел 15.2 контракта).
Только добавление строк в конец.

| модель | для чего | ссылка на веса | ревизия / файл | лицензия | кто добавил |
|---|---|---|---|---|---|
| Ultralytics YOLO11x (COCO) | детекции без обучения: `coco_person` (person на полном кадре), `coco_crop` (кропы столов) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt, копия в Kaggle Dataset `ts-models` | релиз `v8.4.0`, файл `yolo11x.pt`, sha256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` | AGPL-3.0 | B |
| Ultralytics YOLO11m (COCO) | стартовые веса дообучения `ft_yolo_v1` (B4) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt, копия в Kaggle Dataset `ts-models` | релиз `v8.4.0`, файл `yolo11m.pt`, sha256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` | AGPL-3.0 | B |
| Grounding DINO base | zero-shot детекция посуды и вещей на кропах `top` ×4, источник `ov_gdino` (C2), псевдометки (C5) | https://huggingface.co/IDEA-Research/grounding-dino-base | коммит `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`, `model.safetensors`; скачивает `tracks/ov/fetch_weights.py` в `artifacts/models/ov_gdino/`; Kaggle Dataset пока не опубликован | Apache-2.0 (карточка модели, проверено 13.11) | C |
| OWLv2 large patch14 ensemble | zero-shot детекция посуды и вещей на кропах `top` ×4, источник `ov_owl` (C2), псевдометки (C5) | https://huggingface.co/google/owlv2-large-patch14-ensemble | коммит `95e26936e865f87db1742128404b3c035d47d89d`, `model.safetensors`; скачивает `tracks/ov/fetch_weights.py` в `artifacts/models/ov_owl/`; Kaggle Dataset пока не опубликован | Apache-2.0 (карточка модели, проверено 13.11) | C |
| Ultralytics YOLO11x-pose (COCO keypoints) | запястья для источника `pose` (руки, B5) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-pose.pt, копия в Kaggle Dataset `ts-models` | релиз `v8.4.0`, файл `yolo11x-pose.pt`, sha256 `013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8` | AGPL-3.0 | B |
| ft_yolo_v1 (свой детектор, дообученный YOLO11m) | детекции `ft_yolo__v1`: person, hand, cup, tray | Kaggle Dataset `ts-models`, файл `ft_yolo_v1.pt` с карточкой `ft_yolo_v1.json`; обучение `python tracks/det/train.py --config tracks/det/configs/ft_yolo_v1.yaml` | sha256 в карточке `ft_yolo_v1.json`; стартовые веса `yolo11m.pt` релиза `v8.4.0` | AGPL-3.0 (производные от весов Ultralytics) | B |
