"""Единая точка воспроизводимого запуска (раздел 15.1). Задача A7, пока заглушка.

python run.py --data $TS_DATA --artifacts $TS_ARTIFACTS --config configs/submissions/<name>.yaml --out submission.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    parser.add_argument('--from-cache', action='store_true', help='не пересчитывать артефакты с совпавшим sha')
    args = parser.parse_args()
    os.environ['TS_DATA'] = str(args.data)
    os.environ['TS_ARTIFACTS'] = str(args.artifacts)
    raise NotImplementedError('run.py: задача A7. Этапы: rows_ext -> источники -> assign/temporal/features -> stack -> post')


if __name__ == '__main__':
    main()
