"""Гистограммы площади боксов person прод-детектора по дням и по ИК/цвет.

Запуск: python tools/person_area_hist.py  ->  report/person_area_hist.png
Это статистика detections.csv по всем 1718 кадрам. Метки и просмотр кадров не используются.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT

from ts import io
from ts.rows import IR_SAT_THRESHOLD, frame_saturations

THRESHOLD = 45.0   # тыс. px², текущий порог temporal.people_max_area


def main() -> None:
    dets = io.load_prod_dets()
    frames = io.load_frames().set_index('frame_id')
    p = dets[dets.class_name == 'person'].copy()
    p['area'] = (p.x2 - p.x1) * (p.y2 - p.y1) / 1000.0
    p['date'] = p.frame_id.map(frames.date_utc)
    p['ir'] = p.frame_id.map(frame_saturations(sorted(p.frame_id.unique()))) < IR_SAT_THRESHOLD
    panels = [(f'{d[8:]}.{d[5:7]} ({"train" if d < "2026-10-27" else "test"})', p[p.date == d]) for d in sorted(p.date.unique())]
    panels += [('цветные кадры, все дни', p[~p.ir]), ('ИК-кадры, все дни', p[p.ir])]
    edges = np.arange(0, 102.5, 2.5)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for ax, (title, part) in zip(axes.ravel(), panels):
        ax.hist(part.area.clip(upper=99.9), bins=edges, color='#4c78a8', edgecolor='white', linewidth=0.4)
        ax.axvline(THRESHOLD, color='#d62728', linewidth=1.5, linestyle='--')
        share = float((part.area > THRESHOLD).mean()) if len(part) else 0.0
        ax.set_title(f'{title}: {len(part)} боксов, крупнее порога {share:.1%}', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('площадь бокса person, тыс. px² (последний столбец: 97.5 и больше)')
    for ax in axes[:, 0]:
        ax.set_ylabel('число боксов')
    fig.suptitle('Площадь боксов person прод-детектора. Пунктир: порог 45 тыс. px²', fontsize=13)
    fig.tight_layout()
    out = ROOT / 'report' / 'person_area_hist.png'
    fig.savefig(out, dpi=110)
    print(f'сохранено: {out}')
    for title, part in panels:
        hist, _ = np.histogram(part.area, bins=np.arange(30, 65, 5))
        print(f'{title:<28} n={len(part):>5}  30..60 по 5: {hist.tolist()}  >45k: {(part.area > THRESHOLD).mean():.3f}')


if __name__ == '__main__':
    main()
