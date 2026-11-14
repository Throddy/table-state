"""Разбор ошибки предсказания на train по колонкам: недосчёт и пересчёт, ИК и цвет, столы с людьми и без.

Запуск: python tools/error_breakdown.py COMBO [ITEMS_or_pp ...]
Читает ART/preds/<name>.csv (train-строки там out-of-fold) и train_labels.csv.
"""
from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401

from ts import COUNTS, io
from ts.rows import load_rows_ext
from swap_cols import pred_path


def breakdown(name: str) -> None:
    labels = io.load_labels().set_index('row_id')
    pred = io.read_csv_exact(pred_path(name)).set_index('row_id').loc[labels.index]
    rows = load_rows_ext()
    is_ir = rows[rows['dt'] == 0].set_index('row_id')['is_ir'].reindex(labels.index) == 1
    occupied = labels['people'] > 0
    groups = [('все', np.ones(len(labels), dtype=bool)), ('ИК', is_ir.to_numpy()), ('цвет', (~is_ir).to_numpy()),
              ('с людьми', occupied.to_numpy()), ('без людей', (~occupied).to_numpy())]
    print(f'\n{name}: train, out-of-fold. В ячейке: ошибка = недосчёт + пересчёт (объектов в группе)')
    print(f"{'колонка':<8}" + ''.join(f'{title + " (" + str(int(m.sum())) + ")":>26}' for title, m in groups))
    for column in COUNTS:
        diff = (pred[column] - labels[column]).to_numpy(float)
        cells = []
        for _, m in groups:
            under, over = -diff[m & (diff < 0)].sum(), diff[m & (diff > 0)].sum()
            cells.append(f'{under + over:.1f} = {under:.1f} + {over:.1f} ({int(labels[column].to_numpy()[m].sum())})')
        print(f'{column:<8}' + ''.join(f'{c:>26}' for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='+')
    for name in parser.parse_args().names:
        breakdown(name)


if __name__ == '__main__':
    main()
