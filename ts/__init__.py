"""Общая библиотека команды для кейса «Table State» (владелец: трек A).

Модули: paths, io, slots, rows, geometry, crops, assign, temporal, features, stack, post, metric, cv, check.
"""

SEED = 419
FRAME_W = 1280
FRAME_H = 720
COUNTS = ['people', 'hands', 'cups', 'dishes', 'items']
PRED_COLUMNS = ['row_id'] + COUNTS + ['occ_prob']
