"""Расчёты поверх основы COMBO для оставшихся попыток (только цифры, на Kaggle ничего не уходит).

Запуск: python tracks/det/combo_checks.py
  а) cups = max(cups, 1) для занятых столов (people >= 1);
  б) hands: порог confidence ft_yolo v1, подобранный стекером на дне подгонки, умножается на 0.8 и 1.2
     (проверка устойчивости по правилу 12.3).
Для каждого варианта: oof_all, f25, f26, качество колонки на train, число изменённых строк теста против COMBO.
Варианты пишутся в ART/preds/CHK_<имя>.csv с карточкой. Основа: ART/preds/COMBO.csv, собранная по
configs/submissions/COMBO.yaml (должна совпадать с submissions/COMBO.csv).
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import yaml

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # модуль common лежит в tracks/det
import common

sys.path.insert(0, str(common.ROOT / 'tools'))
sys.path.insert(0, str(common.ROOT / 'tracks' / 'core'))
from build_pred import build  # noqa: E402

from ts import PRED_COLUMNS, io, paths, stack  # noqa: E402
from ts.check import check_pred  # noqa: E402
from ts.cv import evaluate  # noqa: E402

BASE = 'COMBO'
CONFIG = common.ROOT / 'configs' / 'submissions' / 'COMBO.yaml'
FT_SOURCE = 'ft_yolo'


def test_changes(pred: pd.DataFrame, base: pd.DataFrame, column: str) -> dict:
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')['row_id']
    a, b = base.set_index('row_id').loc[sample, column], pred.set_index('row_id').loc[sample, column]
    return {'changed': int((a != b).sum()), 'up': int((b > a).sum()), 'down': int((b < a).sum()),
            'sum_base': float(a.sum()), 'sum_new': float(b.sum())}


def row(label: str, name: str, column: str, base: pd.DataFrame, extra: str = '') -> dict:
    path = paths.ART / 'preds' / f'{name}.csv'
    pred = io.read_csv_exact(path)
    report = evaluate(path, bootstrap=0)
    change = test_changes(pred, base, column)
    return {'вариант': label, 'oof_all': round(report['oof_all']['score'], 4), 'f25': round(report['f25']['score'], 4),
            'f26': round(report['f26']['score'], 4), 'колонка': column,
            'качество': round(report['oof_all']['quality'][column], 3),
            'ошибка': f"{report['oof_all']['abs_error'][column]:g}/{report['oof_all']['objects'][column]:g}",
            'тест: изменено': change['changed'], 'вверх': change['up'], 'вниз': change['down'],
            'сумма': f"{change['sum_base']:g} -> {change['sum_new']:g}", 'примечание': extra}


def cups_floor(base: pd.DataFrame) -> str:
    """Вариант (а): занятый стол без чашки получает одну чашку."""
    name = 'CHK_cups_floor'
    out = base.copy()
    occupied = out['people'] >= 1
    out.loc[occupied, 'cups'] = out.loc[occupied, 'cups'].clip(lower=1.0)
    path = paths.art('preds', f'{name}.csv')
    out[PRED_COLUMNS].to_csv(path, index=False)
    card = json.loads((paths.ART / 'preds' / f'{BASE}.json').read_text(encoding='utf-8'))
    card.update({'name': name, 'owner': common.OWNER, 'created_utc': common.utc_now(), 'git_sha': common.git_sha(),
                 'command': common.command_line(), 'notes': f'{BASE} с правилом cups = max(cups, 1) при people >= 1'})
    path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_pred(path)
    return name


def hands_threshold(factor: float) -> tuple[str, list]:
    """Вариант (б): сборка по конфигу COMBO, порог ft_yolo для hands после подбора умножен на `factor`."""
    name = f'CHK_hands_thr_x{factor:g}'.replace('.', '')
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    original = stack.tune_conf
    used: list = []

    def scaled(parts, labels, local, rows):
        min_conf = original(parts, labels, local, rows)
        tuned = min_conf[FT_SOURCE]['hands']
        min_conf[FT_SOURCE]['hands'] = round(tuned * factor, 4)
        used.append((tuned, min_conf[FT_SOURCE]['hands']))
        return min_conf

    stack.tune_conf = scaled
    try:
        build(name, cfg, notes=f'COMBO с порогом ft_yolo hands × {factor:g}', command=common.command_line())
    finally:
        stack.tune_conf = original
    return name, used


def main() -> None:
    common.require_data()
    base = io.read_csv_exact(paths.ART / 'preds' / f'{BASE}.csv')
    rows = [row('COMBO (основа)', BASE, 'cups', base), row('COMBO (основа)', BASE, 'hands', base)]
    rows.append(row('а) cups = max(cups, 1) при people >= 1', cups_floor(base), 'cups', base))
    for factor in (1.0, 0.8, 1.2):
        name, used = hands_threshold(factor)
        note = 'пороги по фолдам f25, f26 и на всём train: ' + ', '.join(f'{a:g} -> {b:g}' for a, b in used)
        label = 'б) hands, порог ft_yolo × 1.0 (контроль)' if factor == 1.0 else f'б) hands, порог ft_yolo × {factor:g}'
        rows.append(row(label, name, 'hands', base, note))
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 250, 'display.max_colwidth', 90)
    print('\n' + table.to_string(index=False))
    table.to_csv(paths.ART / 'preds' / 'CHK_combo_checks.csv', index=False)


if __name__ == '__main__':
    main()
