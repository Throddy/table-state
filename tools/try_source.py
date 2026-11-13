"""Проверка нового источника одной командой: детекции трека -> привязка -> счётчики -> замена колонок в базовом файле.

Запуск:
  python tools/try_source.py --source coco_person__v1 --cols people --base base_calibrated --name AB_people_coco
  python tools/try_source.py --source ov_gdino__v1 --cols items dishes --base <лучший файл> --name AB_ov_items_dishes
  python tools/try_source.py --source ft_yolo__v1 --cols hands cups --base <лучший файл> --name AB_ft --alone

Шаги:
  1. ts.check.check_dets для ART/dets/<source>.parquet (без карточки или с ошибкой формата дальше не идём);
  2. сборка предсказания <name>__src по configs/default.yaml с источниками [prod, <source>]
     (--alone: только <source>); пороги confidence источника подбираются на дне подгонки;
  3. отчёт 12.2 для этой сборки и для базового файла;
  4. tools/swap_cols.py: колонки --cols из сборки подставляются в ART/preds/<base>.csv,
     результат проходит submit_gate -> submissions/<name>.csv. На Kaggle не отправляет.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT / 'tracks' / 'core'))
from build_pred import build  # noqa: E402
from swap_cols import pred_path, swap  # noqa: E402

from ts import COUNTS, io, paths  # noqa: E402
from ts.check import check_dets  # noqa: E402
from ts.cv import evaluate  # noqa: E402


def report_test_changes(base: str, donor: str, cols: list[str]) -> None:
    """Сколько тестовых строк меняет каждая колонка и на сколько меняется её сумма, по дням (без меток)."""
    tables = io.load_tables().set_index('row_id')
    test_ids = tables.index[tables['split'] == 'test']
    a = io.read_csv_exact(pred_path(base)).set_index('row_id').loc[test_ids]
    b = io.read_csv_exact(pred_path(donor)).set_index('row_id').loc[test_ids]
    date = tables.loc[test_ids, 'date_utc']
    print('\nИзменения в тестовых строках (колонка: день: строк изменено, больше / меньше, сумма база -> новая):')
    total = 0
    for column in cols:
        diff = b[column] - a[column]
        for day in sorted(date.unique()) + ['все']:
            m = (date == day) if day != 'все' else pd.Series(True, index=date.index)
            total += int((diff[m].abs() > 1e-9).sum()) if day == 'все' else 0
            print(f"  {column:<8} {day[-5:] if day != 'все' else 'все  '}: {int((diff[m].abs() > 1e-9).sum()):>5} строк, "
                  f"+{int((diff[m] > 1e-9).sum())} / -{int((diff[m] < -1e-9).sum())}, сумма {a.loc[m, column].sum():.1f} -> {b.loc[m, column].sum():.1f}")
    print(f'  всего изменённых значений по колонкам {", ".join(cols)}: {total}'
          + ('  <-- МЕНЬШЕ 100: сабмит почти не отличается от базы' if total < 100 else ''))


def report_bowls(source: str, cfg: dict) -> None:
    """Сколько мисок источника отброшено правилом двойного счёта (тест, текущий кадр)."""
    from ts.features import assigned_for
    from ts.temporal import merge_sources
    parts = [assigned_for(s, cfg) for s in cfg['sources']['dets']]
    if not any((p['class_name'] == 'bowl').any() for p in parts if len(p)):
        return
    off = {**cfg, 'temporal': {**cfg.get('temporal', {}), 'bowl_overlap_iou': None}}
    tables = io.load_tables().set_index('row_id')

    def bowls(c: dict) -> pd.Series:
        merged = merge_sources(parts, c)
        merged = merged[(merged['class_name'] == 'bowl') & (merged['dt'] == 0)]
        return merged['row_id'].map(tables['split']).value_counts()

    before, after = bowls(off), bowls(cfg)
    print('\nМиски после порога confidence из конфига, текущий кадр (до подбора порога на фолдах):')
    for split in ['train', 'test']:
        print(f"  {split}: всего {int(before.get(split, 0))}, отброшено как перекрытые подносом или чашкой "
              f"(IoU >= {cfg['temporal'].get('bowl_overlap_iou')}): {int(before.get(split, 0) - after.get(split, 0))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='имя файла ART/dets/<source>__v<N> без расширения')
    parser.add_argument('--cols', nargs='+', required=True, help='колонки, которые берём из сборки с источником')
    parser.add_argument('--base', required=True, help='базовое предсказание ART/preds/<base>.csv')
    parser.add_argument('--name', required=True, help='имя итогового сабмита')
    parser.add_argument('--alone', action='store_true', help='собирать только из нового источника, без prod')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    parser.add_argument('--note')
    parser.add_argument('--set', action='append', default=[], metavar='ключ=значение',
                        help='изменить параметр конфига для этой сборки, например post.tray_mult=1.0')
    parser.add_argument('--no-gate', action='store_true')
    args = parser.parse_args()

    check_dets(paths.ART / 'dets' / f'{args.source}.parquet')
    print(f'ts.check: {args.source} OK')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cfg['sources']['dets'] = [args.source] if args.alone else ['prod', args.source]
    for item in args.set:
        node, keys = cfg, item.split('=', 1)[0].split('.')
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(item.split('=', 1)[1])
    cfg['compose'] = {**(cfg.get('compose') or {}), **{column: 'stack' for column in args.cols}}
    for counter in args.cols:
        if counter in COUNTS and not cfg['stack']['counts'].get(counter):
            cfg['stack']['counts'][counter] = 'cur'      # счётчик, который раньше был нулевым (items)
    full = f'{args.name}__src'
    build(full, cfg, notes=f'сборка для проверки источника {args.source}, колонки {args.cols}',
          command='python tools/try_source.py ' + ' '.join(sys.argv[1:]))

    header = f"{'файл':<34}{'oof_all':>8}{'f25':>8}{'f26':>8}{'robust':>8}{'AP':>7}" + ''.join(f'{c:>8}' for c in COUNTS)
    print('\n' + header + '\n' + '-' * len(header))
    for label, name in [('база: ' + args.base, args.base), ('сборка с источником', full)]:
        r = evaluate(pred_path(name), bootstrap=0)
        a = r['oof_all']
        print(f"{label[:33]:<34}{a['score']:>8.4f}{r['f25']['score']:>8.4f}{r['f26']['score']:>8.4f}"
              f"{r['robust']['oof_all']['score']:>8.4f}{a['occupancy_ap']:>7.3f}"
              + ''.join(f"{a['quality'][c]:>8.3f}" for c in COUNTS))
    report_test_changes(args.base, full, args.cols)
    report_bowls(args.source, cfg)
    print()
    swap(args.base, full, args.cols, args.name, args.note, gate=not args.no_gate)


if __name__ == '__main__':
    main()
