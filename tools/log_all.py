"""Три журнала треков одной таблицей по времени.

Запуск:
  python tools/log_all.py              # все строки
  python tools/log_all.py --tail 30    # последние 30
  python tools/log_all.py --who C      # только трек C
  python tools/log_all.py --tasks      # только строки «TASK <id> done»
"""
from __future__ import annotations

import argparse
import re

from _bootstrap import ROOT

LOGS = ['LOG_core.md', 'LOG_det.md', 'LOG_ov.md']
HEADER = '| время МСК | кто | имя | что изменено (одна фраза) | oof_all | f25(val26) | f26(val25) | AP | people | hands | cups | dishes | items | вывод |'
TIME = re.compile(r'(\d{1,2})\.(\d{1,2})\s+(\d{1,2}):(\d{2})')


def read_rows() -> list[tuple[tuple, int, str]]:
    """Строки всех журналов с ключом сортировки (месяц, день, час, минута) и порядком в файле."""
    rows = []
    for file_index, name in enumerate(LOGS):
        path = ROOT / 'experiments' / name
        if not path.is_file():
            continue
        for line_index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            if not line.startswith('| ') or line.startswith('| время МСК'):
                continue
            match = TIME.search(line.split('|')[1])
            day, month, hour, minute = (int(v) for v in match.groups()) if match else (99, 99, 0, 0)
            rows.append(((month, day, hour, minute), file_index * 100000 + line_index, line))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tail', type=int, help='показать только последние N строк')
    parser.add_argument('--who', help='только строки трека: A, B или C')
    parser.add_argument('--tasks', action='store_true', help='только строки со статусом задач')
    args = parser.parse_args()
    lines = [line for _, _, line in read_rows()]
    if args.who:
        lines = [line for line in lines if line.split('|')[2].strip() == args.who.upper()]
    if args.tasks:
        lines = [line for line in lines if line.split('|')[3].strip().startswith('TASK ')]
    if args.tail:
        lines = lines[-args.tail:]
    print(HEADER)
    print('|---' * 14 + '|')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
