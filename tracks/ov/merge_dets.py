"""Склейка частей одного источника в артефакт с покрытием `all` (соседние кадры dt = ±30 для ts.temporal).

Запуск:
  python tracks/ov/merge_dets.py ov_owl__v1 ov_owl__v2 ov_owl__v3 --version 4   # targets + доли neighbours 0/2, 1/2
  python tracks/ov/merge_dets.py ov_gdino__v1 ov_gdino__v20 --version 2         # targets + context

Первая часть: готовый прогон `run_ov.py --rows targets` (839 целевых кадров, dt = 0). Остальные части:
- либо один прогон `--rows context` (кропы прогона `all`, которых нет в `targets`);
- либо прогон `--rows neighbours` (кадры-соседи, которые сами не целевые, 879 кадров) целиком или все его доли
  `--shard K/N` (K = 0 … N−1).
Вместе части покрывают все 1718 кадров, поэтому результат получает `coverage: all`, и `ts.temporal` строит для
источника признаки `prev/next/med`. Целевые кадры не пересчитываются. Части должны быть посчитаны одной моделью,
одним промптом и с одними параметрами; каждая часть должна быть полной (сверка с `select_rows`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ov import code_sha, select_rows, utc_now  # noqa: E402

from ts import io, paths  # noqa: E402
from ts.check import check_dets  # noqa: E402
from ts.io import DET_COLUMNS  # noqa: E402

SAME = ['models', 'prompt_sha1', 'crop', 'min_conf', 'nms_iou', 'max_table_frac', 'same_crop_iou', 'fp16']
PART_RE = re.compile(r'^(targets|context|neighbours)(?:, доля (\d+)/(\d+))?$')


def frames_card(frame_ids) -> dict:
    """`frames_processed`: число кадров и sha1 отсортированного списка id через перевод строки."""
    frame_ids = sorted(set(frame_ids))
    return {'n': len(frame_ids), 'sha1': hashlib.sha1('\n'.join(frame_ids).encode()).hexdigest()}


def parse_part(card: dict) -> tuple[str, str | None]:
    """Поле `rows` карточки -> (режим, доля K/N или None)."""
    match = PART_RE.match(str(card.get('rows')))
    if not match:
        raise SystemExit(f"{card.get('name')}: rows = {card.get('rows')!r}, нужен полный прогон targets, context "
                         'или neighbours (можно долями --shard)')
    mode, k, n = match.groups()
    return mode, (f'{k}/{n}' if k is not None else None)


def check_parts(cards: list[dict]) -> dict[str, pd.DataFrame]:
    """Части подходят друг к другу и вместе дают все кадры. Возвращает ожидаемые кропы каждой части."""
    parts = [parse_part(c) for c in cards]
    if parts[0] != ('targets', None):
        raise SystemExit('Первая часть должна быть полным прогоном --rows targets')
    rest = parts[1:]
    modes = {mode for mode, _ in rest}
    if modes == {'context'}:
        if rest != [('context', None)]:
            raise SystemExit('context склеивается одним полным прогоном, без долей')
    elif modes == {'neighbours'}:
        shards = [s for _, s in rest]
        if shards != [None]:
            counts = {s.split('/')[1] for s in shards if s}
            want = [f'{k}/{n}' for n in map(int, counts) for k in range(n)] if len(counts) == 1 else []
            if not all(shards) or sorted(shards) != sorted(want):
                raise SystemExit(f'Доли neighbours неполные или повторяются: {shards}')
    else:
        raise SystemExit(f'После targets нужны части одного вида (context или neighbours), получено {sorted(modes)}')

    different = [(c['name'], key) for c in cards[1:] for key in SAME if c.get(key) != cards[0].get(key)]
    if different:
        raise SystemExit(f'Части посчитаны с разными параметрами: {different}')
    expected = {}
    for card, (mode, shard) in zip(cards, parts):
        rows = select_rows(mode, None, shard)
        want = {'frames_processed': frames_card(rows.src_frame_id), 'crops_processed': len(rows)}
        have = {key: card.get(key) for key in want}
        if have != want:
            raise SystemExit(f"{card['name']}: часть неполная: в карточке {have}, должно быть {want}")
        expected[card['name']] = rows
    return expected


def merge(tables: list[pd.DataFrame], source: str) -> pd.DataFrame:
    """Детекции частей одним файлом формата 8.1, `det_id` перенумерованы внутри кадра."""
    seen: set = set()
    for table in tables:
        crops = set(zip(table.from_row_id, table.from_dt.astype(int)))
        if crops & seen:
            raise SystemExit(f'Части пересекаются по кропам: {len(crops & seen)} общих')
        seen |= crops
    dets = pd.concat(tables, ignore_index=True)
    dets['source'] = source
    dets['det_id'] = source + '_' + dets.frame_id + '_' + dets.groupby('frame_id', sort=False).cumcount().astype(str)
    dets['from_dt'] = dets['from_dt'].astype('Int64')
    return dets[DET_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dets', nargs='+', help='артефакты ART/dets без расширения: сначала targets, затем остальные')
    parser.add_argument('--version', type=int, required=True, help='суффикс результата: <source>__v<N>')
    args = parser.parse_args()
    if len(args.dets) < 2:
        raise SystemExit('Нужно не меньше двух частей')

    sources = {name.split('__')[0] for name in args.dets}
    if len(sources) != 1:
        raise SystemExit(f'Части разных источников: {sorted(sources)}')
    source = sources.pop()
    name = f'{source}__v{args.version}'
    out_path = paths.art('dets', f'{name}.parquet')
    if out_path.is_file():
        print(f'{name}: уже готов ({out_path}), пропуск. Артефакты неизменяемы: укажите новый --version.')
        return
    files = [paths.ART / 'dets' / f'{n}.parquet' for n in args.dets]
    for path in files:
        check_dets(path)
    cards = [json.loads(path.with_suffix('.json').read_text(encoding='utf-8')) for path in files]

    expected = check_parts(cards)
    frames = set().union(*(rows.src_frame_id for rows in expected.values()))
    all_frames = set(io.load_frames()['frame_id'])
    if frames != all_frames:
        raise SystemExit(f'Части покрывают {len(frames)} кадров из {len(all_frames)}')
    dets = merge([pd.read_parquet(path) for path in files], source)
    dets.to_parquet(out_path, index=False)
    first = cards[0]
    card = {
        'name': name, 'kind': 'dets', 'owner': 'C', 'created_utc': utc_now(), 'git_sha': code_sha(),
        'command': 'python tracks/ov/merge_dets.py ' + ' '.join(sys.argv[1:]),
        'models': first['models'],
        'inputs': [f'dets/{n}.parquet' for n in args.dets] + ['rows_ext.parquet'],
        'coverage': 'all', 'rows': ' + '.join(c['rows'] for c in cards), 'dt': [-30, 0, 30], 'context_features': True,
        'frames_processed': frames_card(frames),
        'crops_processed': int(sum(len(rows) for rows in expected.values())),
        'manual_labels_used': False,
        'uses_train_counters': False,
        **{key: first.get(key) for key in ['prompt'] + SAME[1:]},
        'parts': {c['name']: {key: c.get(key) for key in ['rows', 'git_sha', 'created_utc', 'command', 'device',
                                                           'torch', 'transformers']} for c in cards},
        'notes': 'склейка частей run_ov.py: целевые кадры (--rows targets) и соседние кадры; целевые кадры не '
                 'пересчитывались; один кроп на стол каждого кадра-источника; объект в кропах двух столов записан '
                 'дважды, дедупликация в ts.assign',
    }
    out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_dets(out_path)
    print(f"{name}: {len(dets)} детекций, {card['crops_processed']} кропов, {card['frames_processed']['n']} кадров "
          f'-> {out_path}')


if __name__ == '__main__':
    main()
