"""Общая библиотека команды для кейса «Table State» (владелец: трек A).

Сигнатуры зафиксированы в разделе 7 CONTRACT.md.
"""

SEED = 419
FRAME_W = 1280
FRAME_H = 720
COUNTS = ['people', 'hands', 'cups', 'dishes', 'items']
PRED_COLUMNS = ['row_id'] + COUNTS + ['occ_prob']
