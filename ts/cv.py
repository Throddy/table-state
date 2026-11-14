"""Валидация по дням: фолды f25 и f26, объединённая оценка oof_all."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import COUNTS, PRED_COLUMNS, SEED, io
from .metric import fast_score, score_frame
from .rows import load_rows_ext

# имя фолда: (дата подгонки, дата проверки)
FOLDS = {'f25': ('2026-10-25', '2026-10-26'),   # fit на 25, val на 26 (официальная схема)
         'f26': ('2026-10-26', '2026-10-25')}   # fit на 26, val на 25
OUTLIER_FRAME = 'im_cf13426b6f011b9cb35c'

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def _read_pred(pred) -> pd.DataFrame:
    if isinstance(pred, (str, Path)):
        pred = io.read_csv_exact(pred)
    elif pred.index.name == 'row_id' and 'row_id' not in pred:
        pred = pred.reset_index()
    return pred[PRED_COLUMNS].copy()


def oof_predictions(fit_predict: FitPredict) -> pd.DataFrame:
    """Out-of-fold предсказания на 231 train-строку по схеме `FOLDS`.

    `fit_predict(train_rows, predict_rows) -> pred`: `train_rows` это строки
    `train_labels.csv` дня подгонки (со счётчиками), `predict_rows` это строки
    `ts.io.load_tables()` дня проверки (без счётчиков). `pred` содержит
    `row_id`, пять счётчиков и `occ_prob` для всех `predict_rows`.
    """
    labels = io.load_labels()
    tables = io.load_tables()
    parts = []
    for fit_date, val_date in FOLDS.values():
        train_rows = labels[labels['date_utc'] == fit_date].reset_index(drop=True)
        val_ids = labels.loc[labels['date_utc'] == val_date, 'row_id']
        predict_rows = tables[tables['row_id'].isin(val_ids)].reset_index(drop=True)
        pred = _read_pred(fit_predict(train_rows, predict_rows))
        if set(pred['row_id']) != set(val_ids) or pred['row_id'].duplicated().any():
            raise ValueError(f'fit_predict вернул не те row_id для проверки на {val_date}')
        parts.append(pred)
    return pd.concat(parts, ignore_index=True)


def _views(labels: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Оценки `oof_all`, `f25`, `f26` на заданном подмножестве строк."""
    subsets = {'oof_all': labels}
    for name, (_, val_date) in FOLDS.items():
        subsets[name] = labels[labels['date_utc'] == val_date]
    return {name: (score_frame(part, pred) if len(part) else None) for name, part in subsets.items()}


def _bootstrap(labels: pd.DataFrame, pred: pd.DataFrame, n: int) -> dict:
    """Bootstrap по кадрам: перцентили `oof_all` при выборке кадров с возвращением."""
    merged = labels.merge(pred, on='row_id', suffixes=('', '_p'), validate='one_to_one')
    y = merged[COUNTS].to_numpy(float)
    p = merged[[c + '_p' for c in COUNTS]].to_numpy(float)
    prob = merged['occ_prob'].to_numpy(float)
    groups = list(merged.groupby('frame_id', sort=True).indices.values())
    rng = np.random.default_rng(SEED)
    scores = np.empty(n)
    for i in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        scores[i] = fast_score(y[idx], p[idx], prob[idx])
    return {'n': n, 'frames': len(groups), 'mean': float(scores.mean()), 'std': float(scores.std()),
            'p05': float(np.percentile(scores, 5)), 'p95': float(np.percentile(scores, 95))}


def evaluate(pred_or_fn, bootstrap: int = 1000) -> dict:
    """Отчёт по train: oof_all, f25, f26, вид без выброса, ИК и цвет, bootstrap.

    `pred_or_fn`: путь к csv или DataFrame с предсказаниями минимум на 231
    train-строку (для подгонявшихся предсказателей это должны быть
    out-of-fold значения), либо функция `fit_predict` (см. `oof_predictions`).

    Возвращает словарь:
    - `oof_all`, `f25`, `f26`: результат `ts.metric.score_frame`;
    - `robust`: то же без кадра-выброса `OUTLIER_FRAME`;
    - `ir1`, `ir0`: то же отдельно по строкам с `is_ir = 1` и `is_ir = 0`
      (`None`, если строк нет);
    - `bootstrap`: `p05`, `p95`, `mean`, `std` для `oof_all` (по кадрам);
    - `n_params`: число подгоняемых параметров (атрибут `n_params` функции
      или `DataFrame.attrs['n_params']`, иначе `None`).
    """
    if callable(pred_or_fn):
        pred = oof_predictions(pred_or_fn)
        n_params = getattr(pred_or_fn, 'n_params', None)
    else:
        pred = _read_pred(pred_or_fn)
        n_params = getattr(pred_or_fn, 'attrs', {}).get('n_params') if isinstance(pred_or_fn, pd.DataFrame) else None
    labels = io.load_labels()
    missing = set(labels['row_id']) - set(pred['row_id'])
    if missing:
        raise ValueError(f'В предсказаниях нет {len(missing)} train-строк из 231')
    pred = pred[pred['row_id'].isin(labels['row_id'])].reset_index(drop=True)

    report = _views(labels, pred)
    report['robust'] = _views(labels[labels['frame_id'] != OUTLIER_FRAME], pred)
    rows = load_rows_ext()
    is_ir = rows[rows['dt'] == 0].set_index('row_id')['is_ir']
    flag = labels['row_id'].map(is_ir)
    report['ir1'] = _views(labels[flag == 1], pred)
    report['ir0'] = _views(labels[flag == 0], pred)
    report['bootstrap'] = _bootstrap(labels, pred, bootstrap) if bootstrap else None
    report['n_params'] = n_params
    return report


def format_report(report: dict) -> str:
    """Текстовая таблица отчёта `evaluate` для печати в консоль."""
    header = f"{'вид':<16}{'rows':>5}{'score':>8}{'AP':>8}" + ''.join(f'{c:>8}' for c in COUNTS) + '   abs_error / objects'
    lines = [header, '-' * len(header)]

    def add(title: str, r: dict | None) -> None:
        if r is None:
            lines.append(f'{title:<16}{"нет строк":>13}')
            return
        errors = ' '.join(f"{r['abs_error'][c]:g}/{r['objects'][c]:g}" for c in COUNTS)
        lines.append(f"{title:<16}{r['rows']:>5}{r['score']:>8.4f}{r['occupancy_ap']:>8.4f}"
                     + ''.join(f"{r['quality'][c]:>8.3f}" for c in COUNTS) + '   ' + errors)

    for view in ['oof_all', 'f25', 'f26']:
        add(view, report[view])
    for group in ['robust', 'ir1', 'ir0']:
        for view in ['oof_all', 'f25', 'f26']:
            add(f'{group}/{view}', report[group][view])
    b = report.get('bootstrap')
    if b:
        lines.append(f"bootstrap oof_all по кадрам (n={b['n']}): p05={b['p05']:.4f} p95={b['p95']:.4f} "
                     f"mean={b['mean']:.4f} std={b['std']:.4f}")
    lines.append(f"подгоняемых параметров: {report.get('n_params')}")
    return '\n'.join(lines)


def log_row(report: dict, who: str, name: str, change: str, conclusion: str, when: str = '') -> str:
    """Строка для журнала трека `experiments/LOG_<трек>.md` в формате журнала."""
    a = report['oof_all']
    cells = [when, who, name, change, f"{a['score']:.4f}", f"{report['f25']['score']:.4f}",
             f"{report['f26']['score']:.4f}", f"{a['occupancy_ap']:.3f}"]
    cells += [f"{a['quality'][c]:.3f}" for c in COUNTS] + [conclusion]
    return '| ' + ' | '.join(cells) + ' |'
