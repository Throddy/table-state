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


def report_items(base: str, donor: str) -> None:
    """Вещи на столах без людей (people основы = 0): сколько строк и вещей в тесте."""
    tables = io.load_tables().set_index('row_id')
    test_ids = tables.index[tables['split'] == 'test']
    a = io.read_csv_exact(pred_path(base)).set_index('row_id').loc[test_ids]
    b = io.read_csv_exact(pred_path(donor)).set_index('row_id').loc[test_ids]
    empty = a['people'] < 1
    print('Вещи в тесте по занятости стола (people основы):')
    for label, m in [('people = 0', empty), ('people >= 1', ~empty)]:
        print(f"  {label}: строк {int(m.sum())}; с вещами было {int((a.loc[m, 'items'] >= 0.5).sum())} (сумма {a.loc[m, 'items'].sum():.1f}), "
              f"стало {int((b.loc[m, 'items'] >= 0.5).sum())} (сумма {b.loc[m, 'items'].sum():.1f})")


def report_dishes(base: str, donor: str, cfg: dict) -> None:
    """Посуда: сколько тарелок и мисок отброшено как уже посчитанные и сколько строк меняется без подноса."""
    from ts.features import build_features, source_parts
    from ts.temporal import merge_sources
    parts = source_parts(cfg['sources']['dets'], cfg)
    tables = io.load_tables().set_index('row_id')
    off = {**cfg, 'temporal': {**cfg.get('temporal', {}), 'bowl_overlap_iou': None}}

    def count(c: dict) -> pd.DataFrame:
        merged = merge_sources(parts, c)
        merged = merged[merged['class_name'].isin(['plate', 'bowl']) & (merged['dt'] == 0)]
        return merged.assign(split=merged['row_id'].map(tables['split'])).groupby(['split', 'class_name']).size()

    before, after = count(off), count(cfg)
    print('\nТарелки и миски (текущий кадр, пороги confidence из конфига, до подбора на фолдах):')
    for key in sorted(set(before.index) | set(after.index)):
        n, kept = int(before.get(key, 0)), int(after.get(key, 0))
        print(f'  {key[0]:<5} {key[1]:<5}: всего {n}, отброшено как посуда на подносе или миска-чашка: {n - kept}')
    if before.empty:
        print('  в источниках нет классов plate и bowl')
    test_ids = tables.index[tables['split'] == 'test']
    trays = build_features(cfg['sources']['dets'], cfg).loc[test_ids, 'all__tray__cur']
    a = io.read_csv_exact(pred_path(base)).set_index('row_id').loc[test_ids, 'dishes']
    b = io.read_csv_exact(pred_path(donor)).set_index('row_id').loc[test_ids, 'dishes']
    changed = (b - a).abs() > 1e-9
    date = tables.loc[test_ids, 'date_utc']
    print('Изменения dishes в тесте по наличию подноса (строк изменено / сумма разниц):')
    for label, m in [('столы без подноса', trays == 0), ('столы с подносом', trays > 0)]:
        by_day = {d[-5:]: int((changed & m & (date == d)).sum()) for d in sorted(date.unique())}
        print(f'  {label}: {int((changed & m).sum())} строк из {int(m.sum())}, сумма {float((b - a)[m].sum()):+.1f}, по дням {by_day}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, nargs='+', help='имена файлов ART/dets/<source>__v<N> без расширения (один или несколько)')
    parser.add_argument('--drop', nargs='+', default=[], help='убрать эти источники основы (например ft_yolo__v1 при проверке v2)')
    parser.add_argument('--classes', nargs='+', help='взять от новых источников только эти классы словаря 8.2')
    parser.add_argument('--cols', nargs='+', required=True, help='колонки, которые берём из сборки с источником')
    parser.add_argument('--base', required=True, help='базовое предсказание ART/preds/<base>.csv')
    parser.add_argument('--name', required=True, help='имя итогового сабмита')
    parser.add_argument('--alone', action='store_true', help='собирать только из нового источника, без prod')
    parser.add_argument('--all-counters', action='store_true',
                        help='источник участвует во всех счётчиках сборки (по умолчанию только в --cols, '
                             'чтобы остальные колонки, включая people для постобработки, считались как в основе)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    parser.add_argument('--note')
    parser.add_argument('--set', action='append', default=[], metavar='ключ=значение',
                        help='изменить параметр конфига для этой сборки, например post.tray_mult=1.0')
    parser.add_argument('--no-gate', action='store_true')
    args = parser.parse_args()

    for source in args.source:
        check_dets(paths.ART / 'dets' / f'{source}.parquet')
        print(f'ts.check: {source} OK')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    # Источники основы (например ft_yolo для рук) и их ограничения сохраняются, новые добавляются к ним.
    kept = [s for s in cfg['sources'].get('dets', ['prod']) if s not in args.source and s not in args.drop]
    for dropped in args.drop:
        (cfg['sources'].get('counters') or {}).pop(dropped, None)
    cfg['sources']['dets'] = list(args.source) if args.alone else kept + list(args.source)
    if not args.all_counters and not args.alone:
        cfg['sources']['counters'] = {**(cfg['sources'].get('counters') or {}),
                                      **{source: [c for c in args.cols if c in COUNTS] for source in args.source}}
    if args.classes:
        cfg['sources']['classes'] = {**(cfg['sources'].get('classes') or {}), **{source: list(args.classes) for source in args.source}}
    for column in args.cols:      # колонка считается по детекциям, прежнее голосование с rowfeats для неё выключается
        if (cfg['stack'].get('vote') or {}).get(column):
            cfg['stack']['vote'][column] = None
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
    build(full, cfg, notes=f'сборка для проверки источников {args.source}, колонки {args.cols}, классы {args.classes or "все"}',
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
    if 'items' in args.cols:
        report_items(args.base, full)
    if 'dishes' in args.cols:
        report_dishes(args.base, full, cfg)
    print()
    swap(args.base, full, args.cols, args.name, args.note, gate=not args.no_gate,
         command='python tools/try_source.py ' + ' '.join(sys.argv[1:]))


if __name__ == '__main__':
    main()
