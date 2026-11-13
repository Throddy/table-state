import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from ts.assign import assign
from ts.temporal import aggregate, context_median

T1 = (100.0, 200.0, 300.0, 300.0)   # w=200, h=100, зона top: 84..316 × 192..308
T2 = (320.0, 200.0, 520.0, 300.0)


def make_rows(dts=(0, -30, 30), tables=(T1, T2)):
    out = []
    for i, box in enumerate(tables, start=1):
        for dt in dts:
            out.append({'row_id': f'q{i}', 'frame_id': 'im_t', 'src_frame_id': 'im_t' if dt == 0 else f'im_n{dt}',
                        'dt': dt, 'table_id': f'table_{i}', 'slot': i + 1, 'split': 'train',
                        'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]})
    return pd.DataFrame(out)


def make_dets(items):
    return pd.DataFrame([{'det_id': f'prod_{f}_{n}', 'frame_id': f, 'source': 'prod', 'class_name': c, 'confidence': conf,
                          'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3], 'from_row_id': None, 'from_dt': None}
                         for n, (f, c, conf, b) in enumerate(items)])


def owners(assigned, dt=0):
    part = assigned[assigned.dt == dt]
    return {(r.class_name, r.row_id) for r in part.itertuples()}


def test_small_objects_rules():
    dets = make_dets([
        ('im_t', 'cup', 0.9, (150, 240, 178, 268)),        # на столе 1
        ('im_t', 'cup', 0.8, (302, 250, 322, 270)),        # между столами: больше площади в рамке стола 2
        ('im_t', 'cup', 0.9, (600, 250, 620, 270)),        # вне всех зон
        ('im_t', 'hand', 0.9, (150, 150, 178, 178)),       # у лица выше стола
        ('im_t', 'hand', 0.9, (150, 190, 178, 206)),       # центр y=198: не выше y1 - 0.05h
        ('im_t', 'napkin', 0.9, (400, 240, 430, 270)),     # негатив
        ('im_t', 'tray', 0.5, (401, 241, 431, 271)),       # перекрыт более уверенным негативом
        ('im_t', 'tray', 0.9, (440, 220, 500, 280)),
    ])
    a = assign(dets, make_rows(dts=(0,)), {})
    assert owners(a) == {('cup', 'q1'), ('cup', 'q2'), ('hand', 'q1'), ('tray', 'q2')}
    assert len(a) == 4 and not a.det_id.duplicated().any()
    assert set(a.counter) == {'cups', 'hands', 'dishes'} and (a.weight == 1.0).all()


def test_object_between_tables_goes_to_one_table():
    # центр x=312 лежит в зонах top обоих столов (84..316 и 304..536); больше площади бокса в рамке стола 2
    dets = make_dets([('im_t', 'cup', 0.9, (298, 240, 326, 268))])
    a = assign(dets, make_rows(dts=(0,)), {})
    assert owners(a) == {('cup', 'q2')}


def test_dedup_and_bag_share():
    dets = make_dets([
        ('im_t', 'cup', 0.9, (150, 240, 178, 268)),
        ('im_t', 'cup', 0.6, (153, 242, 181, 270)),            # тот же объект
        ('im_t', 'bag', 0.9, (60, 150, 140, 250)),             # в основном вне зоны top
        ('im_t', 'bag', 0.9, (350, 210, 430, 290)),
    ])
    a = assign(dets, make_rows(dts=(0,)), {})
    assert sorted(a.class_name) == ['bag', 'cup']
    assert a.loc[a.class_name == 'cup', 'confidence'].iloc[0] == 0.9
    assert a.loc[a.class_name == 'bag', 'row_id'].iloc[0] == 'q2'


def test_people_rules():
    dets = make_dets([
        ('im_t', 'person', 0.9, (120, 120, 250, 260)),     # закрывает рамку стола 1: правило бейзлайна
        ('im_t', 'person', 0.9, (360, 90, 480, 212)),      # сидит за дальней стороной стола 2: добор
        ('im_t', 'person', 0.6, (380, 320, 480, 400)),     # ниже стола 2, рамку не задевает
        ('im_t', 'person', 0.9, (560, 100, 700, 300)),     # далеко от столов
    ])
    a = assign(dets, make_rows(dts=(0,)), {})
    assert {(r.row_id, r.rule) for r in a.itertuples()} == {('q1', 'overlap'), ('q2', 'rescue')}
    assert a.det_id.is_unique
    counts = aggregate(a, {'temporal': {'people_rescue': False}}, make_rows(dts=(0,)))
    assert counts['prod__people__cur'].to_dict() == {'q1': 1.0, 'q2': 0.0}
    counts = aggregate(a, {'temporal': {'people_max_area': 15000}}, make_rows(dts=(0,)))
    assert counts['prod__people__cur'].to_dict() == {'q1': 0.0, 'q2': 1.0}   # 130×140 = 18200 px² отброшен


def test_neighbour_uses_target_box_and_own_detections():
    dets = make_dets([
        ('im_t', 'cup', 0.9, (150, 240, 178, 268)),
        ('im_n-30', 'cup', 0.9, (150, 240, 178, 268)),
        ('im_n-30', 'cup', 0.9, (220, 240, 248, 268)),
    ])
    rows = make_rows()
    a = assign(dets, rows, {})
    f = aggregate(a, {}, rows)
    assert f.loc['q1', ['prod__cups__cur', 'prod__cups__prev', 'prod__cups__next']].tolist() == [1, 2, 0]
    assert f.loc['q1', 'prod__cups__med'] == 1 and f.loc['q1', 'prod__cups__max'] == 2
    assert f.loc['q1', 'prod__cups__conf_max'] == pytest.approx(0.9)
    assert f.loc['q2', 'prod__cups__cur'] == 0 and len(f) == 2


def test_one_neighbour_is_not_filled_with_zeros():
    dets = make_dets([
        ('im_t', 'cup', 0.9, (150, 240, 178, 268)),
        ('im_n30', 'cup', 0.9, (150, 240, 178, 268)),
        ('im_n30', 'cup', 0.9, (220, 240, 248, 268)),
        ('im_n30', 'tray', 0.9, (400, 220, 460, 280)),
    ])
    rows = make_rows(dts=(0, 30))
    f = aggregate(assign(dets, rows, {}), {}, rows)
    assert np.isnan(f.loc['q1', 'prod__cups__prev']) and np.isnan(f.loc['q2', 'prod__dishes__prev'])
    assert f.loc['q1', 'prod__cups__next'] == 2
    assert f.loc['q1', 'prod__cups__med'] == 2        # оба больше 0: максимум из двух
    assert f.loc['q2', 'prod__dishes__med'] == 0      # сейчас 0: остаётся текущий
    assert f.loc['q1', 'prod__cups__min'] == 1 and f.loc['q2', 'prod__dishes__max'] == 1


def test_no_neighbours_and_no_detections():
    rows = make_rows(dts=(0,))
    f = aggregate(assign(make_dets([('im_t', 'cup', 0.9, (150, 240, 178, 268))]), rows, {}), {}, rows)
    assert f.loc['q1', 'prod__cups__med'] == 1 and np.isnan(f.loc['q1', 'prod__cups__next'])
    empty = assign(make_dets([('im_other', 'cup', 0.9, (150, 240, 178, 268))]), rows, {})
    assert len(empty) == 0
    f = aggregate(empty, {'source': 'prod'}, rows)
    assert (f.filter(like='__cur') == 0).all().all() and len(f) == 2


def test_context_median_rule():
    nan = np.nan
    cur = np.array([0., 1., 1., 0., 2., 1.])
    prev = np.array([1., 0., 3., nan, nan, nan])
    nxt = np.array([1., 0., 2., 2., 1., nan])
    assert context_median(cur, prev, nxt).tolist() == [1, 0, 2, 0, 2, 1]


@needs_data
def test_real_rows_with_one_neighbour():
    from ts.features import build_features
    f = build_features(['prod'], {})
    f.columns = [c.replace('all__', 'prod__') for c in f.columns]
    assert len(f) == 4964 and f.index.is_unique
    one = f[f.n_ctx == 1]
    assert len(one) > 0
    assert (one[['prod__cups__prev', 'prod__cups__next']].isna().sum(axis=1) == 1).all()
    assert not f[[c for c in f.columns if c.endswith(('__cur', '__med', '__max', '__min'))]].isna().any().any()
    full = f[f.n_ctx == 2]
    assert not full[['prod__people__prev', 'prod__people__next']].isna().any().any()


@needs_data
def test_stack_guarantees():
    from ts import io, stack
    from ts.features import build_features
    cfg = {'temporal': {'people_max_area': 45000}}
    f = build_features(['prod'], cfg)
    labels = io.load_labels()
    pred = stack.predict(stack.fit(f, labels, cfg), f, cfg).set_index('row_id')
    assert len(pred) == 4964 and pred.occ_prob.between(0, 1).all()
    seated = pred[pred.people >= 1]
    assert (seated.occ_prob >= 0.5 + 0.01 * seated.people.clip(upper=5)).all()
    assert pred.occ_prob.nunique() == len(pred)
    assert (pred.hands == f['all__hands__cur']).all()      # руки только по текущему кадру
    assert (pred['items'] == 0).all()
