"""Единая точка воспроизводимого запуска (раздел 15.1 контракта).

  python run.py --data $TS_DATA --artifacts $TS_ARTIFACTS --config configs/submissions/COMBO.yaml --out submission.csv

Этапы:
  1. rows_ext: индекс строк (читает кадры для признака is_ir); с --from-cache берётся готовый файл;
  2. источники: детекции и признаки строк треков B и C читаются из каталога артефактов (кэш), каждый файл
     проходит ts.check; команды их пересчёта на GPU описаны в tracks/det/README.md и tracks/ov/README.md;
  3. привязка объектов к столам, контекст, признаки (ts.assign, ts.temporal, ts.features);
  4. подгонка на train-счётчиках (пороги confidence источников, логрег занятости, бейзлайн организаторов
     для колонок compose: baseline) и предсказание 4733 тестовых строк;
  5. постобработка, запись submission.csv, проверка валидатором организаторов.
Сеть не используется. Сид 419.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='каталог пакета организаторов')
    parser.add_argument('--artifacts', type=Path, required=True, help='каталог артефактов (dets/, rowfeats/, rows_ext.parquet)')
    parser.add_argument('--config', type=Path, required=True, help='configs/submissions/<name>.yaml')
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    parser.add_argument('--from-cache', action='store_true', help='не пересчитывать rows_ext.parquet, если он есть')
    args = parser.parse_args()
    # Пути задаются до первого импорта ts: библиотека читает их из окружения.
    os.environ['TS_DATA'] = str(args.data.resolve())
    os.environ['TS_ARTIFACTS'] = str(args.artifacts.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import pandas as pd
    import yaml

    from ts import PRED_COLUMNS, io, paths, stack
    from ts.check import check_rows_ext
    from ts.rows import ROWS_EXT_NAME, build_rows_ext
    from ts.vendor import score as vscore

    started = time.time()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if 'swap' in cfg and 'sources' not in cfg:
        raise SystemExit(f'{args.config} это рецепт замены колонок (tools/swap_cols.py), а не конфиг сборки. '
                         'Воспроизводимые конфиги: configs/submissions/COMBO.yaml и собранные tools/combo.py.')
    if not (args.from_cache and (paths.ART / ROWS_EXT_NAME).is_file()):
        build_rows_ext()
    check_rows_ext()
    print(f'[1/5] rows_ext готов ({time.time() - started:.0f} с)')

    sources = list(cfg['sources'].get('dets', [])) + list(cfg['sources'].get('rowfeats', []))
    print(f'[2/5] источники: {", ".join(sources)} (из {paths.ART}, проверка ts.check при чтении)')
    fit_predict = stack.make_fit_predict(cfg)
    print(f'[3/5] привязка и контекст посчитаны ({time.time() - started:.0f} с)')
    labels = io.load_labels()
    sample = pd.read_csv(paths.DATA / 'sample_submission.csv')
    pred = fit_predict(labels, sample[['row_id']])
    print(f'[4/5] подгонка на {len(labels)} train-строках и предсказание {len(pred)} строк ({time.time() - started:.0f} с)')
    submission = pred.set_index('row_id').loc[sample['row_id']].reset_index()[PRED_COLUMNS]
    vscore.validate_submission(submission, sample['row_id'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f'[5/5] {args.out}: {len(submission)} строк, валидатор организаторов пройден, всего {time.time() - started:.0f} с')


if __name__ == '__main__':
    main()
