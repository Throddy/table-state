# MODELS.md

Каждая модель заносится сюда в момент первого использования (раздел 15.2 контракта).
Только добавление строк в конец.

| модель | для чего | ссылка на веса | ревизия / файл | лицензия | кто добавил |
|---|---|---|---|---|---|
| Ultralytics YOLO11x (COCO) | детекции без обучения: `coco_person` (person на полном кадре), `coco_crop` (кропы столов) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt, копия в Kaggle Dataset `ts-models` | релиз `v8.4.0`, файл `yolo11x.pt`, sha256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` | AGPL-3.0 | B |
| Ultralytics YOLO11m (COCO) | стартовые веса дообучения `ft_yolo_v1` (B4) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt, копия в Kaggle Dataset `ts-models` | релиз `v8.4.0`, файл `yolo11m.pt`, sha256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` | AGPL-3.0 | B |
| Ultralytics YOLO11x-pose (COCO keypoints) | запястья для источника `pose` (руки, B5) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-pose.pt, копия в Kaggle Dataset `ts-models` | релиз `v8.4.0`, файл `yolo11x-pose.pt`, sha256 `013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8` | AGPL-3.0 | B |
