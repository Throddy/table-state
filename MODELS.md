# MODELS.md

Каждая модель заносится сюда в момент первого использования (раздел 15.2 контракта).
Только добавление строк в конец.

| модель | для чего | ссылка на веса | ревизия / файл | лицензия | кто добавил |
|---|---|---|---|---|---|
| Grounding DINO base | zero-shot детекция посуды и вещей на кропах `top` ×4, источник `ov_gdino` (C2), псевдометки (C5) | https://huggingface.co/IDEA-Research/grounding-dino-base | коммит `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`, `model.safetensors`; скачивает `tracks/ov/fetch_weights.py` в `artifacts/models/ov_gdino/`; Kaggle Dataset пока не опубликован | Apache-2.0 (карточка модели, проверено 13.11) | C |
| OWLv2 large patch14 ensemble | zero-shot детекция посуды и вещей на кропах `top` ×4, источник `ov_owl` (C2), псевдометки (C5) | https://huggingface.co/google/owlv2-large-patch14-ensemble | коммит `95e26936e865f87db1742128404b3c035d47d89d`, `model.safetensors`; скачивает `tracks/ov/fetch_weights.py` в `artifacts/models/ov_owl/`; Kaggle Dataset пока не опубликован | Apache-2.0 (карточка модели, проверено 13.11) | C |
