"""C5: псевдометки для трека B из согласованных детекций Grounding DINO и OWLv2 (раздел 13 контракта).

Запуск:
  python tracks/ov/make_pseudo.py ov_gdino__v1 ov_owl__v1               # -> ART/labels/ov_pseudo__v1.parquet
  python tracks/ov/make_pseudo.py ov_gdino__v0 ov_owl__v0 --version 0   # отладка на train-днях

Кандидат: обе модели нашли на кадре объект одного класса (правило совпадения 9.2.6). Счётные детекции,
перекрытые более уверенным негативом своего источника, не участвуют. Бокс кандидата это среднее двух боксов.

Пороги (по одному на модель) подбираются только по счётчикам train на строках `dt = 0`, отдельно для групп
`GROUPS`: телефоны (счётчик items), тарелки и миски (dishes), подносы (dishes). Оценка precision сверху:
sum(min(найдено, истина)) / sum(найдено). Выбирается пара порогов с наибольшим sum(min) при оценке не ниже
`PRECISION_TARGET` и не меньше `MIN_FOUND` найденных объектов; при равенстве более строгая пара. Если такой пары
нет, группа в псевдометки не попадает. Печатается и проверка между днями: пороги одного дня на другом дне.
Сумки, бумаги и негативы в псевдометки не входят: по счётчикам train их точность отдельно не проверить.

Выход в формате 8.1: `source = ov_pseudo`, `confidence` = минимум из двух моделей. Покрытие наследуется от
входных файлов. Кадр без боксов не означает, что объектов на нём нет. Тестовые кадры обрабатываются только
автоматически; для просмотра глазами пишутся листы только по train-строкам:
ART/labels/ov_pseudo__v<N>_train_p*.jpg (кроп, псевдобоксы, счётчики train).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from contact_sheet import load_font  # noqa: E402  (добавляет корень репозитория в sys.path)
from eval_dets import BOX, GRID, attach, dedup, drop_covered, same_object, train_rows  # noqa: E402
from run_ov import code_sha, utc_now  # noqa: E402

from ts import COUNTS, io, paths  # noqa: E402
from ts.check import CLASS_TO_COUNTER, check_dets  # noqa: E402
from ts.crops import crop_table, to_crop_coords  # noqa: E402
from ts.cv import FOLDS  # noqa: E402
from ts.io import DET_COLUMNS  # noqa: E402

SOURCE = 'ov_pseudo'
# Группа -> (классы, счётчик train для проверки). Приоритет трека B: phone и plate.
GROUPS = {'phone': (['phone'], 'items'), 'plate': (['plate', 'bowl'], 'dishes'), 'tray': (['tray'], 'dishes')}
PRECISION_TARGET = 0.85   # запас над 0.8 из контракта: оценка по счётчикам завышает precision
MIN_FOUND = 5
SHEET_COLS, SHEET_ROWS, GAP, CAPTION_H = 3, 3, 10, 30
COLORS = {'dishes': '#c77dff', 'items': '#ffd60a', None: '#ff4d4d'}


def candidates(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Объекты, найденные обеими моделями: класс, `conf_a`, `conf_b`, средний бокс, строка кропа модели `a`."""
    classes = [c for names, _ in GROUPS.values() for c in names]
    frames_b = dict(tuple(b.groupby('frame_id', sort=False)))
    found = []
    for frame_id, da in a.groupby('frame_id', sort=True):
        if frame_id not in frames_b:
            continue
        da, db = drop_covered(da), drop_covered(frames_b[frame_id])
        for name in classes:
            xa, xb = da[da.class_name == name], db[db.class_name == name]
            if xa.empty or xb.empty:
                continue
            xa = xa.iloc[dedup(xa.confidence.to_numpy(float), xa[BOX].to_numpy(float))]
            ba, bb, cb = xa[BOX].to_numpy(float), xb[BOX].to_numpy(float), xb.confidence.to_numpy(float)
            same = same_object(ba, bb)
            for i in np.flatnonzero(same.any(axis=1)):
                j = np.flatnonzero(same[i])[np.argmax(cb[same[i]])]
                found.append([frame_id, name, float(xa.confidence.iloc[i]), float(cb[j]), *((ba[i] + bb[j]) / 2),
                              xa.from_row_id.iloc[i], int(xa.from_dt.iloc[i])])
    out = pd.DataFrame(found, columns=['frame_id', 'class_name', 'conf_a', 'conf_b'] + BOX + ['from_row_id', 'from_dt'])
    out['confidence'] = out[['conf_a', 'conf_b']].min(axis=1)
    return out


def select(cand: pd.DataFrame, thr: tuple[float, float]) -> pd.DataFrame:
    """Кандидаты выше порогов без повторов одного объекта (по убыванию `confidence`)."""
    cand = cand[(cand.conf_a >= thr[0]) & (cand.conf_b >= thr[1])]
    keep = [g.index[dedup(g.confidence.to_numpy(float), g[BOX].to_numpy(float))]
            for _, g in cand.groupby('frame_id', sort=False)]
    return cand.loc[np.sort(np.concatenate(keep))] if keep else cand


def tune(cand: pd.DataFrame, classes: list[str], counter: str, rows: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Подбор пары порогов группы по счётчикам train. Возвращает пороги и проверку (или `thr = None`)."""
    by_row = attach(cand[cand.class_name.isin(classes)], rows)
    y = labels.loc[rows.row_id, counter].to_numpy()
    combos = list(itertools.product(GRID, GRID))
    found = np.zeros((len(combos), len(rows)), dtype=int)
    for j, row_id in enumerate(rows.row_id):
        d = by_row.get(row_id)
        if d is None or d.empty:
            continue
        ca, cb, conf, boxes = (d.conf_a.to_numpy(), d.conf_b.to_numpy(), d.confidence.to_numpy(float),
                               d[BOX].to_numpy(float))
        for i, (ta, tb) in enumerate(combos):
            mask = (ca >= ta) & (cb >= tb)
            found[i, j] = len(dedup(conf[mask], boxes[mask])) if mask.any() else 0
    hit = np.minimum(found, y[None, :])

    def best(mask: np.ndarray):
        n, h = found[:, mask].sum(axis=1), hit[:, mask].sum(axis=1)
        ok = (n >= MIN_FOUND) & (h >= PRECISION_TARGET * n)
        if not ok.any():
            return None
        return max(np.flatnonzero(ok), key=lambda i: (h[i], sum(combos[i]), combos[i][0]))

    def stats(i: int, mask: np.ndarray) -> dict:
        n, h, total = int(found[i, mask].sum()), int(hit[i, mask].sum()), int(y[mask].sum())
        return {'found': n, 'hit': h, 'truth': total, 'precision_max': round(h / n, 3) if n else None,
                'recall_max': round(h / total, 3) if total else None}

    everything = np.ones(len(rows), bool)
    chosen = best(everything)
    report = {'classes': classes, 'counter': counter, 'thr': None}
    if chosen is None:
        return report
    report.update(thr=[float(t) for t in combos[chosen]], train=stats(chosen, everything))
    for name, (fit_date, val_date) in FOLDS.items():   # пороги дня подгонки на дне проверки
        fit, val = (rows.date_utc == fit_date).to_numpy(), (rows.date_utc == val_date).to_numpy()
        report[f'day_{val_date}'] = stats(chosen, val)
        i = best(fit)
        report[name] = None if i is None else {'thr': [float(t) for t in combos[i]], **stats(i, val)}
    return report


def train_sheets(pseudo: pd.DataFrame, rows: pd.DataFrame, labels: pd.DataFrame, stem: Path) -> int:
    """Листы для просмотра: кропы train-строк `dt = 0`, где есть счётные псевдобоксы или посуда и вещи по разметке."""
    for old in stem.parent.glob(f'{stem.name}_train_p*.jpg'):
        old.unlink()
    by_row = attach(pseudo, rows)
    busy = labels.loc[rows.row_id, ['dishes', 'items']].sum(axis=1).to_numpy() > 0
    cells = []
    for r, has_objects in zip(rows.itertuples(), busy):
        d = by_row.get(r.row_id, pseudo.iloc[:0])
        if not has_objects and d.class_name.map(CLASS_TO_COUNTER).isna().all():
            continue
        crop, meta = crop_table(r.src_frame_id, (r.x1, r.y1, r.x2, r.y2), 'top', 4.0, r.slot)
        draw = ImageDraw.Draw(crop)
        for q, b in zip(d.itertuples(), to_crop_coords(d[BOX].to_numpy(float), meta)):
            color = COLORS[CLASS_TO_COUNTER[q.class_name]]
            draw.rectangle(list(b), outline=color, width=2)
            draw.text((b[0] + 3, b[1] + 2), f'{q.class_name} {q.confidence:.2f}', fill=color, stroke_width=2,
                      stroke_fill='black')
        lab = labels.loc[r.row_id]
        cells.append((crop, f"{r.row_id} {r.date_utc[5:]} s{r.slot}{' IR' if r.is_ir else ''}  "
                            + ' '.join(f'{c[0]}{int(lab[c])}' for c in COUNTS)))
    if not cells:
        return 0
    font = load_font(18)
    cell_w, cell_h = max(c.width for c, _ in cells), max(c.height for c, _ in cells) + CAPTION_H
    per_page = SHEET_COLS * SHEET_ROWS
    for page in range(-(-len(cells) // per_page)):
        canvas = Image.new('RGB', (SHEET_COLS * (cell_w + GAP) + GAP, SHEET_ROWS * (cell_h + GAP) + GAP), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (crop, caption) in enumerate(cells[page * per_page:(page + 1) * per_page]):
            x, y = GAP + (i % SHEET_COLS) * (cell_w + GAP), GAP + (i // SHEET_COLS) * (cell_h + GAP)
            draw.text((x, y), caption, fill='black', font=font)
            canvas.paste(crop, (x, y + CAPTION_H))
        canvas.save(stem.parent / f'{stem.name}_train_p{page + 1}.jpg', quality=90)
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dets', nargs=2, help='два артефакта ART/dets без расширения: Grounding DINO и OWLv2')
    parser.add_argument('--version', type=int, default=1)
    args = parser.parse_args()

    name = f'{SOURCE}__v{args.version}'
    out_path = paths.art('labels', f'{name}.parquet')
    if out_path.is_file():
        print(f'{name}: уже готов ({out_path}), пропуск. Артефакты неизменяемы: для пересчёта укажите новый --version.')
        return
    cards, tables = [], []
    for det_name in args.dets:
        path = paths.ART / 'dets' / f'{det_name}.parquet'
        check_dets(path)
        cards.append(json.loads(path.with_suffix('.json').read_text(encoding='utf-8')))
        tables.append(pd.read_parquet(path))
    if cards[0]['frames_processed'] != cards[1]['frames_processed']:
        raise SystemExit('Входные файлы обработали разные наборы кадров')

    rows, labels = train_rows(), io.load_labels().set_index('row_id')
    cand = candidates(*tables)
    reports = {group: tune(cand, classes, counter, rows, labels) for group, (classes, counter) in GROUPS.items()}
    for group, report in reports.items():
        print(f'{group}: ' + json.dumps({k: v for k, v in report.items() if k != 'classes'}, ensure_ascii=False))
    parts = [select(cand[cand.class_name.isin(r['classes'])], r['thr']) for r in reports.values() if r['thr']]
    if not parts:
        print(f'Ни одна группа не достигла оценки precision {PRECISION_TARGET}: псевдометок нет.')
    pseudo = pd.concat(parts, ignore_index=True) if parts else cand.iloc[:0].copy()
    pseudo = pseudo.sort_values(['frame_id', 'class_name', 'x1', 'y1'], kind='stable').reset_index(drop=True)
    pseudo['source'] = SOURCE
    pseudo['det_id'] = SOURCE + '_' + pseudo.frame_id + '_' + pseudo.groupby('frame_id', sort=False).cumcount().astype(str)
    pseudo['from_dt'] = pseudo['from_dt'].astype('Int64')
    pseudo[DET_COLUMNS].to_parquet(out_path, index=False)

    card = {
        'name': name, 'kind': 'labels', 'owner': 'C', 'created_utc': utc_now(), 'git_sha': code_sha(),
        'command': 'python tracks/ov/make_pseudo.py ' + ' '.join(sys.argv[1:]),
        'models': sorted({m for c in cards for m in c['models']}),
        'inputs': [f'dets/{n}.parquet' for n in args.dets] + ['train_labels.csv'],
        'coverage': 'all' if all(c['coverage'] == 'all' for c in cards) else 'train_only',
        'frames_processed': cards[0]['frames_processed'],
        'manual_labels_used': False,
        'uses_train_counters': True,
        'precision_target': PRECISION_TARGET, 'min_found': MIN_FOUND,
        'groups': reports,
        'boxes': {k: int(v) for k, v in pseudo.class_name.value_counts().items()},
        'frames_with_boxes': int(pseudo.frame_id.nunique()),
        'sha1': hashlib.sha1(out_path.read_bytes()).hexdigest(),
        'notes': 'согласие Grounding DINO и OWLv2 (тот же класс, совпадение по правилу 9.2.6); пороги выбраны '
                 'только по счётчикам train, боксы получены только моделями, ручной разметки и правок нет; '
                 'precision_max это оценка сверху по счётчикам; кадр без боксов не означает отсутствие объектов',
    }
    out_path.with_suffix('.json').write_text(json.dumps(card, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    check_dets(out_path)
    print(f'{name}: {len(pseudo)} боксов на {pseudo.frame_id.nunique()} кадрах -> {out_path}')
    print(pseudo.class_name.value_counts().to_string())
    n = train_sheets(pseudo, rows, labels, out_path.with_suffix(''))
    print(f'листы для просмотра: {n} train-строк -> {out_path.with_suffix("")}_train_p*.jpg')


if __name__ == '__main__':
    main()
