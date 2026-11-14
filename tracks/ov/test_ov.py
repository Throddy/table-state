"""Тесты трека C: замечания ревью PR #3 (фильтр по дням, покрытие в карточках). Запуск: pytest tracks/ov."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'experiments'))
import eval_dets  # noqa: E402
import make_pseudo  # noqa: E402
import run_ov  # noqa: E402

from ts import io, paths  # noqa: E402

needs_data = pytest.mark.skipif(not (paths.DATA / 'tables.csv').is_file(), reason='нет каталога данных TS_DATA')
needs_rows = pytest.mark.skipif(not (paths.ART / 'rows_ext.parquet').is_file(), reason='нет rows_ext.parquet')


def test_keep_dates_default_is_test_days_only():
    frame_dates = pd.Series({'a': '2026-10-25', 'b': '2026-10-26', 'c': '2026-10-27', 'd': '2026-10-28'})
    dets = pd.DataFrame({'frame_id': ['a', 'b', 'c', 'c', 'd', 'unknown'], 'n': range(6)})
    kept = make_pseudo.keep_dates(dets, make_pseudo.TEST_DATES, frame_dates)
    assert kept.frame_id.tolist() == ['c', 'c', 'd']     # train-дни и неизвестный кадр не проходят
    assert kept.n.tolist() == [2, 3, 4]                  # порядок и строки не меняются
    assert make_pseudo.TEST_DATES == ('2026-10-27', '2026-10-28')


def test_keep_dates_other_days_and_empty():
    frame_dates = pd.Series({'a': '2026-10-25', 'c': '2026-10-27'})
    dets = pd.DataFrame({'frame_id': ['a', 'c']})
    assert make_pseudo.keep_dates(dets, ['2026-10-25'], frame_dates).frame_id.tolist() == ['a']
    assert make_pseudo.keep_dates(dets.iloc[:0], make_pseudo.TEST_DATES, frame_dates).empty


def test_weakest_coverage():
    assert make_pseudo.weakest_coverage(['all', 'all']) == 'all'
    assert make_pseudo.weakest_coverage(['all', 'targets']) == 'targets'
    assert make_pseudo.weakest_coverage(['targets', 'targets']) == 'targets'
    assert make_pseudo.weakest_coverage(['targets', 'train_only']) == 'train_only'
    assert make_pseudo.weakest_coverage(['train_only', 'all']) == 'train_only'


def test_run_ov_coverage():
    assert run_ov.coverage_of('all', None) == 'all'
    assert run_ov.coverage_of('targets', None) == 'targets'
    assert run_ov.coverage_of('train', None) == 'train_only'
    assert run_ov.coverage_of('train0', None) == 'train_only'
    assert run_ov.coverage_of('targets', 8) == 'train_only'   # обрезанный прогон это отладка
    assert run_ov.coverage_of('all', 8) == 'train_only'


def test_rowfeats_needs_full_coverage():
    cards = {'a': {'coverage': 'targets'}, 'b': {'coverage': 'all'}, 'c': {'coverage': 'train_only'}, 'd': {}}
    assert eval_dets.partial_coverage(cards) == ['c', 'd']
    assert eval_dets.TRAY_MULT == 2.0


@needs_data
@needs_rows
def test_targets_frames_match_ts_check():
    """Кадры прогона `--rows targets` это ровно набор, с которым `ts.check` сверяет `coverage: targets`."""
    rows = run_ov.select_rows('targets', None)
    frame_ids = sorted(rows.src_frame_id.unique())
    expected = sorted(set(io.load_tables()['frame_id']))
    assert set(rows.dt) == {0} and len(rows) == 4964
    assert frame_ids == expected and len(frame_ids) == 839
    assert (hashlib.sha1('\n'.join(frame_ids).encode()).hexdigest()
            == hashlib.sha1('\n'.join(expected).encode('utf-8')).hexdigest())


@needs_data
@needs_rows
def test_all_rows_cover_every_frame_and_neighbours():
    """Прогон `--rows all` покрывает все 1718 кадров и строки `dt = ±30` (нужно для prev/next/med)."""
    rows = run_ov.select_rows('all', None)
    assert set(rows.src_frame_id) == set(io.load_frames()['frame_id']) and rows.src_frame_id.nunique() == 1718
    assert set(rows.dt) == {-30, 0, 30}


@needs_data
def test_train_and_test_days_do_not_overlap():
    dates = io.load_tables().groupby('split')['date_utc'].agg(set)
    assert dates['test'] == set(make_pseudo.TEST_DATES)
    assert not dates['train'] & set(make_pseudo.TEST_DATES)


# ---------- соседние кадры (dt = ±30): режим --rows context и склейка merge_dets.py

@needs_data
@needs_rows
def test_context_rows_complete_targets_to_all():
    """`targets` и `context` не пересекаются и вместе дают ровно кропы прогона `all`."""
    parts = {which: run_ov.select_rows(which, None) for which in ('targets', 'context', 'all')}
    keys = {which: set(zip(rows.row_id, rows.dt)) for which, rows in parts.items()}
    assert not keys['targets'] & keys['context']
    assert keys['targets'] | keys['context'] == keys['all']
    assert len(parts['context']) == len(parts['all']) - len(parts['targets']) == 5234
    assert set(parts['context'].dt) == {-30, 30}          # целевые кадры не пересчитываются
    assert run_ov.coverage_of('context', None) == 'train_only'   # часть сама по себе это не полное покрытие


def _fake_part(rows: pd.DataFrame, which: str, source: str, version: int, folder: Path, **changes) -> str:
    """Артефакт-заглушка: по одной детекции на кроп (бокс рамки стола), карточка как у `run_ov.py`."""
    import merge_dets
    name = f'{source}__v{version}'
    dets = pd.DataFrame({'frame_id': rows.src_frame_id, 'source': source, 'class_name': 'plate', 'confidence': 0.5,
                         'x1': rows.x1.clip(0, 1270), 'y1': rows.y1.clip(0, 710), 'x2': rows.x2.clip(5, 1280),
                         'y2': rows.y2.clip(5, 720), 'from_row_id': rows.row_id, 'from_dt': rows.dt.astype('Int64')})
    dets.insert(0, 'det_id', source + '_' + dets.frame_id + '_' + dets.groupby('frame_id').cumcount().astype(str))
    dets = dets[(dets.x2 > dets.x1) & (dets.y2 > dets.y1)]
    dets.to_parquet(folder / f'{name}.parquet', index=False)
    card = {'name': name, 'kind': 'dets', 'owner': 'C', 'models': ['m@1'], 'coverage': 'train_only', 'rows': which,
            'frames_processed': merge_dets.frames_card(rows.src_frame_id), 'crops_processed': len(rows),
            'manual_labels_used': False, 'uses_train_counters': False, 'prompt': 'p.txt', 'prompt_sha1': 'abc',
            'crop': {'kind': 'top', 'scale': 4.0}, 'min_conf': 0.1, 'nms_iou': 0.5, 'max_table_frac': 0.6,
            'same_crop_iou': 0.8, 'fp16': False, **changes}
    (folder / f'{name}.json').write_text(__import__('json').dumps(card), encoding='utf-8')
    return name


@needs_data
@needs_rows
def test_merge_targets_and_context_gives_coverage_all(tmp_path, monkeypatch):
    import json
    import shutil

    import merge_dets
    from ts.check import check_dets
    parts = {which: run_ov.select_rows(which, None) for which in ('targets', 'context')}
    shutil.copy(paths.ART / 'rows_ext.parquet', tmp_path / 'rows_ext.parquet')
    monkeypatch.setattr(paths, 'ART', tmp_path)
    (tmp_path / 'dets').mkdir()
    names = [_fake_part(parts[which], which, 'ov_gdino', version, tmp_path / 'dets')
             for which, version in (('targets', 1), ('context', 20))]
    monkeypatch.setattr(sys, 'argv', ['merge_dets.py', *names, '--version', '2'])
    merge_dets.main()

    out = tmp_path / 'dets' / 'ov_gdino__v2.parquet'
    check_dets(out)
    dets, card = pd.read_parquet(out), json.loads(out.with_suffix('.json').read_text(encoding='utf-8'))
    assert card['coverage'] == 'all' and card['frames_processed']['n'] == 1718 and card['crops_processed'] == 10198
    assert card['context_features'] is True and card['manual_labels_used'] is False
    assert dets.det_id.is_unique and dets.frame_id.nunique() == 1718
    assert set(dets.from_dt.astype(int)) == {-30, 0, 30}
    part_rows = sum(len(pd.read_parquet(tmp_path / 'dets' / f'{n}.parquet')) for n in names)
    assert len(dets) == part_rows                              # ничего не потеряно и не задвоено

    # повторный запуск ничего не перезаписывает
    before = out.read_bytes()
    merge_dets.main()
    assert out.read_bytes() == before

    # части с разными параметрами, в другом порядке или неполные не склеиваются
    other = _fake_part(parts['context'], 'context', 'ov_gdino', 21, tmp_path / 'dets', prompt_sha1='other')
    cut = _fake_part(parts['context'].head(100), 'context', 'ov_gdino', 22, tmp_path / 'dets')
    for bad in ([names[0], other], [names[1], names[0]], [names[0], cut]):
        monkeypatch.setattr(sys, 'argv', ['merge_dets.py', *bad, '--version', '3'])
        with pytest.raises(SystemExit):
            merge_dets.main()
    assert not (tmp_path / 'dets' / 'ov_gdino__v3.parquet').exists()


@needs_data
@needs_rows
def test_merge_targets_and_neighbour_shards_gives_coverage_all(tmp_path, monkeypatch):
    """Ноутбук kaggle_owl_neighbours: ov_owl__v1 (targets) + доли neighbours 0/2 и 1/2 -> один источник, coverage all."""
    import json
    import shutil

    import merge_dets
    from ts.check import check_dets
    shutil.copy(paths.ART / 'rows_ext.parquet', tmp_path / 'rows_ext.parquet')
    monkeypatch.setattr(paths, 'ART', tmp_path)
    (tmp_path / 'dets').mkdir()
    targets = _fake_part(run_ov.select_rows('targets', None), 'targets', 'ov_owl', 1, tmp_path / 'dets', fp16=True)
    shards = [_fake_part(run_ov.select_rows('neighbours', None, f'{k}/2'), f'neighbours, доля {k}/2', 'ov_owl', 2 + k,
                         tmp_path / 'dets', fp16=True) for k in (0, 1)]
    assert [len(run_ov.select_rows('neighbours', None, f'{k}/2')) for k in (0, 1)] == [2580, 2593]
    monkeypatch.setattr(sys, 'argv', ['merge_dets.py', targets, *shards, '--version', '4'])
    merge_dets.main()

    out = tmp_path / 'dets' / 'ov_owl__v4.parquet'
    check_dets(out)
    dets, card = pd.read_parquet(out), json.loads(out.with_suffix('.json').read_text(encoding='utf-8'))
    assert card['coverage'] == 'all' and card['frames_processed']['n'] == 1718
    assert card['crops_processed'] == 4964 + 5173 and card['context_features'] is True
    assert dets.det_id.is_unique and set(dets.from_dt.astype(int)) == {-30, 0, 30}
    assert len(dets) == sum(len(pd.read_parquet(tmp_path / 'dets' / f'{n}.parquet')) for n in [targets, *shards])

    # без одной доли, с повтором доли или с другим fp16 не склеивается
    other = _fake_part(run_ov.select_rows('neighbours', None, '1/2'), 'neighbours, доля 1/2', 'ov_owl', 9,
                       tmp_path / 'dets', fp16=False)
    for bad in ([targets, shards[0]], [targets, shards[0], shards[0]], [targets, shards[0], other], [shards[0], targets]):
        monkeypatch.setattr(sys, 'argv', ['merge_dets.py', *bad, '--version', '5'])
        with pytest.raises(SystemExit):
            merge_dets.main()
    assert not (tmp_path / 'dets' / 'ov_owl__v5.parquet').exists()


# ---------- лист с ошибками на train-кадрах (C6)

def test_demo_error_frames_use_all_train_rows():
    import demo_errors
    counts = demo_errors.COUNTS
    labels = pd.DataFrame({'row_id': ['r1', 'r2', 'r3'], 'frame_id': ['a', 'a', 'b'], 'table_id': ['t1', 't2', 't1'],
                           **{c: [1, 0, 2] for c in counts}})
    pred = pd.DataFrame({'row_id': ['r3', 'r1', 'r2', 'test_row'], **{c: [0, 1, 1, 9] for c in counts}})
    frames, rows = demo_errors.pick_error_frames(pred, labels, 1)
    assert frames == ['b']                                   # ошибка 10 против 5 у кадра a
    assert rows.abs_error.tolist() == [10] and rows.row_id.tolist() == ['r3']
    with pytest.raises(SystemExit):                          # без части train-строк лист не строится
        demo_errors.pick_error_frames(pred[pred.row_id != 'r2'], labels, 1)


# ---------- таблица ошибок COMBO по дням (C6)

@needs_data
def test_c6_errors_table_and_checks(tmp_path):
    import c6_errors
    labels = io.load_labels()
    sub = pd.read_csv(paths.ROOT / 'submissions' / 'COMBO.csv')
    with pytest.raises(SystemExit):                       # файл сабмита без train-строк не подходит
        c6_errors.load_oof(paths.ROOT / 'submissions' / 'COMBO.csv', None)

    train = labels[['row_id', 'people', 'hands', 'cups', 'dishes', 'items']].assign(occ_prob=labels.occupied * 0.9)
    train.loc[0, 'items'] += 2                            # один пересчёт на 2 вещи
    oof = pd.concat([train, sub], ignore_index=True)
    path = tmp_path / 'COMBO.csv'
    oof.to_csv(path, index=False)
    (tmp_path / 'COMBO.json').write_text('{"train_pred": "oof_by_date"}', encoding='utf-8')
    text, numbers = c6_errors.table(c6_errors.load_oof(path, paths.ROOT / 'submissions' / 'COMBO.csv'))
    day = '25.10' if labels.loc[0, 'date_utc'] == '2026-10-25' else '26.10'
    assert numbers[day]['items'] == {**numbers[day]['items'], 'error': 2.0, 'under': 0.0, 'over': 2.0}
    assert numbers['оба дня']['people']['error'] == 0 and numbers['оба дня']['people']['quality'] == 1.0
    assert numbers['25.10']['rows'] + numbers['26.10']['rows'] == 231

    (tmp_path / 'COMBO.json').write_text('{"train_pred": "zero_shot"}', encoding='utf-8')
    with pytest.raises(SystemExit):                       # карточка не out-of-fold
        c6_errors.load_oof(path, None)
    sub.loc[0, 'people'] += 1
    sub.to_csv(tmp_path / 'other.csv', index=False)
    (tmp_path / 'COMBO.json').write_text('{"train_pred": "oof_by_date"}', encoding='utf-8')
    with pytest.raises(SystemExit):                       # тест не совпадает с итоговым сабмитом
        c6_errors.load_oof(path, tmp_path / 'other.csv')

    slides = tmp_path / 'slides.md'
    slides.write_text('до\n<!-- c6:errors:start -->\nстарое\n<!-- c6:errors:end -->\nпосле\n', encoding='utf-8')
    c6_errors.write(slides, text, 'preds/COMBO.csv')
    content = slides.read_text(encoding='utf-8')
    assert content.startswith('до\n') and content.endswith('\nпосле\n') and 'старое' not in content
    assert '| items |' in content and content.count('<!-- c6:errors:start -->') == 1
