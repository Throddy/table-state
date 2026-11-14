"""C6: таблица ошибок COMBO по классам за 25.10 и 26.10 (train, out-of-fold) для слайда 6.

Запуск (файл из Kaggle Dataset `victimestsate/ts-artifacts-dets` v5, `preds/COMBO.csv` и карточка `COMBO.json`):
  python tracks/ov/c6_errors.py <путь>/preds/COMBO.csv                       # печать таблицы
  python tracks/ov/c6_errors.py <путь>/preds/COMBO.csv --write tracks/ov/C6_SLIDES.md   # и вставка в слайды

Проверки перед расчётом: в файле все 231 train-строка; в карточке рядом `train_pred: oof_by_date`; тестовые строки
совпадают с итоговым сабмитом `submissions/COMBO.csv` (тот же COMBO, а не другая сборка). Тестовые строки в расчёт
не идут: разметки для них нет.

В ячейке: ошибка = недосчёт + пересчёт (объектов в разметке), ниже качество 1 − WAPE. Все числа считает
`ts.metric.score_frame` (формула организаторов) и разность счётчиков. При `--write` таблица заменяет текст между
маркерами `<!-- c6:errors:start -->` и `<!-- c6:errors:end -->`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ts import COUNTS, PRED_COLUMNS, io, paths  # noqa: E402
from ts.metric import score_frame  # noqa: E402

START, END = '<!-- c6:errors:start -->', '<!-- c6:errors:end -->'
VALUES = [c for c in PRED_COLUMNS if c != 'row_id']   # пять счётчиков и occ_prob
DAYS = [('25.10', '2026-10-25'), ('26.10', '2026-10-26')]


def load_oof(path: Path, submission: Path | None) -> pd.DataFrame:
    """Предсказания с проверками: все train-строки, карточка out-of-fold, тест как в итоговом сабмите."""
    pred = pd.read_csv(path)
    labels = io.load_labels()
    missing = set(labels.row_id) - set(pred.row_id)
    if missing:
        raise SystemExit(f'{path}: нет {len(missing)} train-строк из {len(labels)}. Нужен preds/COMBO.csv с OOF, '
                         'а не файл сабмита')
    card = path.with_suffix('.json')
    if card.is_file():
        train_pred = json.loads(card.read_text(encoding='utf-8')).get('train_pred')
        if train_pred != 'oof_by_date':
            raise SystemExit(f'{card}: train_pred = {train_pred!r}, нужен oof_by_date')
    else:
        print(f'внимание: нет карточки {card.name}, out-of-fold не подтверждён карточкой')
    if submission is not None and submission.is_file():
        test = pd.read_csv(submission).set_index('row_id')
        mine = pred.set_index('row_id').reindex(test.index)
        diff = np.abs(mine[VALUES].to_numpy(float) - test[VALUES].to_numpy(float)).max()
        if not np.isfinite(diff) or diff > 1e-9:
            raise SystemExit(f'Тестовые строки {path.name} не совпадают с {submission} (макс. разница {diff})')
    return pred


def table(pred: pd.DataFrame) -> tuple[str, dict]:
    """Markdown-таблица «класс × день» и словарь чисел."""
    labels = io.load_labels()
    merged = labels.merge(pred[['row_id'] + VALUES], on='row_id', suffixes=('', '_pred'), validate='one_to_one')
    groups = [(title, merged[merged.date_utc == day]) for title, day in DAYS] + [('оба дня', merged)]
    numbers = {}
    for title, rows in groups:
        frame = rows[['row_id'] + [f'{c}_pred' for c in COUNTS] + ['occ_prob']]
        frame = frame.rename(columns={f'{c}_pred': c for c in COUNTS})
        result = score_frame(rows[labels.columns], frame)
        diff = {c: rows[f'{c}_pred'].to_numpy(float) - rows[c].to_numpy(float) for c in COUNTS}
        numbers[title] = {
            'rows': len(rows), 'score': result['score'], 'ap': result['occupancy_ap'],
            **{c: {'objects': int(result['objects'][c]), 'error': float(np.abs(diff[c]).sum()),
                   'under': float(-diff[c][diff[c] < 0].sum()), 'over': float(diff[c][diff[c] > 0].sum()),
                   'quality': result['quality'][c]} for c in COUNTS}}

    def num(v: float) -> str:
        return f'{v:g}'

    head = '| Класс | ' + ' | '.join(f'{t} ({numbers[t]["rows"]} строк)' for t, _ in groups) + ' |'
    lines = [head, '|---|' + '---|' * len(groups)]
    for c in COUNTS:
        cells = []
        for t, _ in groups:
            n = numbers[t][c]
            cells.append(f"{num(n['error'])} = {num(n['under'])} + {num(n['over'])} ({n['objects']}), "
                         f"качество {n['quality']:.3f}")
        lines.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    lines.append('| AP занятости | ' + ' | '.join(f"{numbers[t]['ap']:.3f}" for t, _ in groups) + ' |')
    lines.append('| скор | ' + ' | '.join(f"{numbers[t]['score']:.4f}" for t, _ in groups) + ' |')
    note = ('В ячейке: ошибка = недосчёт + пересчёт (объектов в разметке), качество 1 − WAPE. Предсказания '
            'out-of-fold: строки 25.10 предсказаны с порогами, подобранными на 26.10, и наоборот.')
    return '\n'.join(lines) + '\n\n' + note, numbers


def write(slides: Path, text: str, source: str) -> None:
    content = slides.read_text(encoding='utf-8')
    if content.count(START) != 1 or content.count(END) != 1:
        raise SystemExit(f'В {slides} нужен ровно один блок {START} … {END}')
    head, rest = content.split(START)
    _, tail = rest.split(END)
    block = f'{START}\nИсточник: `{source}`, команда `python tracks/ov/c6_errors.py <файл> --write {slides.as_posix()}`.\n\n{text}\n{END}'
    slides.write_text(head + block + tail, encoding='utf-8', newline='\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pred', type=Path, help='preds/COMBO.csv с out-of-fold train-строками')
    parser.add_argument('--submission', type=Path, default=paths.ROOT / 'submissions' / 'COMBO.csv',
                        help='итоговый сабмит для сверки тестовых строк')
    parser.add_argument('--write', type=Path, help='вставить таблицу в файл слайдов между маркерами')
    args = parser.parse_args()
    text, numbers = table(load_oof(args.pred, args.submission))
    print(text)
    if args.write:
        write(args.write, text, f'{args.pred.parent.name}/{args.pred.name}')
        print(f'таблица вставлена в {args.write}')


if __name__ == '__main__':
    main()
