"""Оценка файла предсказаний по схеме раздела 12.2.

Запуск: python tools/eval_pred.py artifacts/preds/<name>.csv [--bootstrap 1000] [--json out.json]
Файл должен содержать как минимум 231 train-строку. Для предсказателей,
подгонявшихся по счётчикам, train-строки должны быть out-of-fold (раздел 8.5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (добавляет корень репозитория в sys.path)

from ts.cv import evaluate, format_report, log_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pred', type=Path, help='csv с row_id, пятью счётчиками и occ_prob')
    parser.add_argument('--bootstrap', type=int, default=1000, help='повторов bootstrap по кадрам (0 = выключить)')
    parser.add_argument('--json', type=Path, help='сохранить полный отчёт в json')
    parser.add_argument('--who', default='A', help='кто, для строки журнала')
    args = parser.parse_args()

    report = evaluate(args.pred, bootstrap=args.bootstrap)
    print(format_report(report))
    print('\nСтрока для журнала своего трека experiments/LOG_<core|det|ov>.md (время, описание и вывод заполнить):')
    print(log_row(report, args.who, args.pred.stem, '<что изменено>', f'`python tools/eval_pred.py {args.pred}`'))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
