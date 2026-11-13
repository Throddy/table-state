"""Абляция «минус один источник» и проверка отдельных флагов одной командой.

Запуск:
  python tools/ablate.py                                  # полная сборка и варианты без каждого источника
  python tools/ablate.py --set post.items_prior=false     # дополнительно вариант с изменённым параметром
  python tools/ablate.py --set temporal.people_max_area=36000 --set stack.tune_people_area=false

Каждый вариант оценивается честно: out-of-fold по дням через ts.cv.evaluate.
Печатает таблицу и готовые строки для experiments/LOG_core.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _bootstrap import ROOT

from ts import COUNTS, stack
from ts.cv import evaluate, log_row


def set_path(cfg: dict, path: str, value) -> None:
    keys = path.split('.')
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def without_source(cfg: dict, source: str) -> dict:
    """Конфиг без источника: убирается из списков и из правил голосования."""
    out = json.loads(json.dumps(cfg))
    for kind in ['dets', 'rowfeats']:
        out['sources'][kind] = [s for s in out['sources'].get(kind, []) if s != source]
    prefix = source.split('__v')[0] + '__'
    vote = out.get('stack', {}).get('vote') or {}
    out.setdefault('stack', {})['vote'] = {c: r for c, r in vote.items() if not r['column'].startswith(prefix)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='ключ=значение',
                        help='вариант с изменённым параметром (значение в синтаксисе yaml), можно несколько')
    parser.add_argument('--together', action='store_true', help='применить все --set одним вариантом')
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding='utf-8'))

    variants = [('full', base)]
    sources = list(base['sources'].get('dets', [])) + list(base['sources'].get('rowfeats', []))
    if len(sources) > 1:
        variants += [(f'minus {s}', without_source(base, s)) for s in sources]
    changes = [(item.split('=', 1)[0], yaml.safe_load(item.split('=', 1)[1])) for item in args.set]
    if args.together and changes:
        cfg = json.loads(json.dumps(base))
        for key, value in changes:
            set_path(cfg, key, value)
        variants.append((', '.join(args.set), cfg))
    else:
        for (key, value), text in zip(changes, args.set):
            cfg = json.loads(json.dumps(base))
            set_path(cfg, key, value)
            variants.append((text, cfg))

    header = f"{'вариант':<44}{'oof_all':>8}{'f25':>8}{'f26':>8}{'robust':>8}{'AP':>7}" + ''.join(f'{c:>8}' for c in COUNTS)
    print(header)
    print('-' * len(header))
    reports = []
    for name, cfg in variants:
        report = evaluate(stack.make_fit_predict(cfg), bootstrap=args.bootstrap)
        reports.append((name, report))
        a = report['oof_all']
        print(f"{name:<44}{a['score']:>8.4f}{report['f25']['score']:>8.4f}{report['f26']['score']:>8.4f}"
              f"{report['robust']['oof_all']['score']:>8.4f}{a['occupancy_ap']:>7.3f}"
              + ''.join(f"{a['quality'][c]:>8.3f}" for c in COUNTS))
    print('\nСтроки для experiments/LOG_core.md:')
    for name, report in reports:
        print(log_row(report, 'A', name, '<что изменено>', f'`python tools/ablate.py` (robust {report["robust"]["oof_all"]["score"]:.4f})'))


if __name__ == '__main__':
    main()
