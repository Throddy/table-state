"""Склейка двух частей одного источника в артефакт с покрытием `all` (эксперимент с соседними кадрами dt = ±30).

Запуск:
  python tracks/ov/merge_dets.py ov_gdino__v1 ov_gdino__v20 --version 2    # -> ART/dets/ov_gdino__v2.parquet
  python tracks/ov/merge_dets.py ov_owl__v1 ov_owl__v20 --version 2        # -> ART/dets/ov_owl__v2.parquet

Первая часть: готовый прогон `run_ov.py --rows targets` (целевые кадры, dt = 0). Вторая: `--rows context`
(кропы прогона `all`, которых нет в `targets`). Вместе это ровно кропы прогона `--rows all`, поэтому результат
получает `coverage: all`, и `ts.temporal` может строить для источника признаки `prev/next/med`.
Целевые кадры не пересчитываются. Части должны быть посчитаны одной моделью, одним промптом и с одними параметрами.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ov import code_sha, select_rows, utc_now  # noqa: E402

from ts import paths  # noqa: E402
from ts.check import check_dets  # noqa: E402
from ts.io import DET_COLUMNS  # noqa: E402

SAME = ['models', 'prompt_sha1', 'crop', 'min_conf', 'nms_iou', 'max_table_frac', 'same_crop_iou', 'fp16']
PARTS = ['targets', 'context']


def frames_card(rows: pd.DataFrame) -> dict:
    """`frames_processed` для набора кропов: число кадров и sha1 отсортированного списка id."""
    frame_ids = sorted(rows.src_frame_id.unique())
    return {'n': len(frame_ids), 'sha1': hashlib.sha1('\n'.join(frame_ids).encode()).hexdigest()}


def check_parts(cards: list[dict], expected: dict[str, pd.DataFrame]) -> None:
    """Части подходят друг к другу: порядок `targets`, `context`, те же параметры, полные наборы кропов."""
    if [c.get('rows') for c in cards] != PARTS:
        raise SystemExit(f"Нужны прогоны --rows {PARTS[0]} и --rows {PARTS[1]} в этом порядке, "
                         f"в карточках: {[c.get('rows') for c in cards]}")
    different = [key for key in SAME if cards[0].get(key) != cards[1].get(key)]
    if different:
        raise SystemExit(f'Части посчитаны с разными параметрами: {different}')
    for card, part in zip(cards, PARTS):
        want = {'frames_processed': frames_card(expected[part]), 'crops_processed': len(expected[part])}
        have = {key: card.get(key) for key in want}
        if have != want:
            raise SystemExit(f"{card['name']}: это не полный прогон --rows {part}: в карточке {have}, должно быть {want}")


def merge(tables: list[pd.DataFrame], source: str) -> pd.DataFrame:
    """Детекции двух частей одним файлом формата 8.1, `det_id` перенумерованы внутри кадра."""
    crops = [set(zip(t.from_row_id, t.from_dt.astype(int))) for t in tables]
    if crops[0] & crops[1]:
        raise SystemExit(f'Части пересекаются по кропам: {len(crops[0] & crops[1])} общих')
    dets = pd.concat(tables, ignore_index=True)
    dets['source'] = source
    dets['det_id'] = source + '_' + dets.frame_id + '_' + dets.groupby('frame_id', sort=False).cumcount().astype(str)
    dets['from_dt'] = dets['from_dt'].astype('Int64')
    return dets[DET_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dets', nargs=2, help='артефакты ART/dets без расширения: прогон targets и прогон context')
    parser.add_argument('--version', type=int, required=True, help='суффикс результата: <source>__v<N>')
    args = parser.parse_args()

    files = [paths.ART / 'dets' / f'{name}.parquet' for name in args.dets]
    for path in files:
        check_dets(path)
    cards = [json.loads(path.with_suffix('.json').read_text(encoding='utf-8')) for path in files]
    source = args.dets[0].split('__')[0]
    if args.dets[1].split('__')[0] != source:
        raise SystemExit('Части разных источников')
    name = f'{source}__v{args.version}'
    out_path = paths.art('dets', f'{name}.parquet')
    if out_path.is_file():
        print(f'{name}: уже готов ({out_path}), пропуск. Артефакты неизменяемы: укажите новый --version.')
        return

    expected = {part: select_rows(part, None) for part in PARTS + ['all']}
    check_parts(cards, expected)
    dets = merge([pd.read_parquet(path) for path in files], source)
    dets.to_parquet(out_path, index=False)
    first = cards[0]
    card = {
        'name': name, 'kind': 'dets', 'owner': 'C', 'created_utc': utc_now(), 'git_sha': code_sha(),
        'command': 'python tracks/ov/merge_dets.py ' + ' '.join(sys.argv[1:]),
        'models': first['models'],
        'inputs': [f'dets/{n}.parquet' for n in args.dets] + ['rows_ext.parquet'],
        'coverage': 'all', 'rows': 'all', 'dt': [-30, 0, 30], 'context_features': True,
        'frames_processed': frames_card(expected['all']),
        'crops_processed': len(expected['all']),
        'manual_labels_used': False,
        'uses_train_counters': False,
        **{key: first.get(key) for key in ['prompt'] + SAME[1:]},
        'parts': {c['name']: {key: c.get(key) for key in ['git_sha', 'created_utc', 'command', 'device', 'torch',
                                                           'transformers']} for c in cards},
        'notes': 'склейка прогонов run_ov.py --rows targets и --rows context: вместе это кропы прогона --rows all, '
                 'целевые кадры не пересчитывались; один кроп на стол каждого кадра-источника; объект в кропах '
                 'двух столов записан дважды, дедупликация в ts.assign',
    }
    out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_dets(out_path)
    print(f"{name}: {len(dets)} детекций, {card['crops_processed']} кропов, {card['frames_processed']['n']} кадров "
          f'-> {out_path}')


if __name__ == '__main__':
    main()
