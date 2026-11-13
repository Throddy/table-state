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

import yaml

from _bootstrap import ROOT

sys.path.insert(0, str(ROOT / 'tracks' / 'core'))
from build_pred import build  # noqa: E402
from swap_cols import pred_path, swap  # noqa: E402

from ts import COUNTS, paths  # noqa: E402
from ts.check import check_dets  # noqa: E402
from ts.cv import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='имя файла ART/dets/<source>__v<N> без расширения')
    parser.add_argument('--cols', nargs='+', required=True, help='колонки, которые берём из сборки с источником')
    parser.add_argument('--base', required=True, help='базовое предсказание ART/preds/<base>.csv')
    parser.add_argument('--name', required=True, help='имя итогового сабмита')
    parser.add_argument('--alone', action='store_true', help='собирать только из нового источника, без prod')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    parser.add_argument('--note')
    parser.add_argument('--no-gate', action='store_true')
    args = parser.parse_args()

    check_dets(paths.ART / 'dets' / f'{args.source}.parquet')
    print(f'ts.check: {args.source} OK')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cfg['sources']['dets'] = [args.source] if args.alone else ['prod', args.source]
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
    print()
    swap(args.base, full, args.cols, args.name, args.note, gate=not args.no_gate)


if __name__ == '__main__':
    main()
