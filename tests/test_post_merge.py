import numpy as np
import pandas as pd
import pytest

from test_assign_temporal import make_dets, make_rows
from ts import post, stack
from ts.assign import assign
from ts.temporal import aggregate, apply_filters, merge_sources


def pred_frame():
    return pd.DataFrame({'row_id': ['a', 'b', 'c', 'd'], 'people': [0.0, 1.0, 2.0, 1.0], 'hands': [1.0, 3.0, 0.0, 0.0],
                         'cups': [0.4, 0.0, 60.0, 0.0], 'dishes': 0.0, 'items': [0.0, 0.0, 2.0, 0.0],
                         'occ_prob': [0.1, 0.9, 1.2, 0.7]})


def test_items_prior_without_sources():
    out = post.apply(pred_frame(), {'post': {'items_prior': True}})
    assert out['items'].tolist() == [0, 1, 2, 1]
    assert post.apply(pred_frame(), {'post': {}})['items'].tolist() == [0, 0, 2, 0]


@pytest.mark.parametrize('stack_cfg', [{'counts': {'items': 'cur'}}, {'vote': {'items': {'column': 'vlm_qwen__items'}}}])
def test_items_prior_with_sources_needs_floor_flag(stack_cfg):
    cfg = {'stack': stack_cfg, 'post': {'items_prior': True}}
    assert post.apply(pred_frame(), cfg)['items'].tolist() == [0, 0, 2, 0]
    cfg['post']['items_prior_floor'] = True
    assert post.apply(pred_frame(), cfg)['items'].tolist() == [0, 1, 2, 1]


def test_other_post_flags():
    out = post.apply(pred_frame(), {'post': {'hands_need_people': True, 'hands_cap': True, 'round_counts': True}})
    assert out['hands'].tolist() == [0, 2, 0, 0]
    assert out['cups'].tolist() == [0, 0, 50, 0] and out['occ_prob'].max() == 1.0
    raw = post.apply(pred_frame(), {'post': {'clip': False}})
    assert raw['cups'].max() == 60 and raw['occ_prob'].max() == 1.2
    with pytest.raises(NotImplementedError):
        post.apply(pred_frame(), {'post': {'empty_night': True}})


def other_source(dets, name):
    out = dets.copy()
    out['source'] = name
    out['det_id'] = [f'{name}_{i}' for i in range(len(out))]
    return out


def test_merge_sources_dedups_and_thresholds():
    rows = make_rows(dts=(0,))
    prod = make_dets([('im_t', 'cup', 0.9, (150, 240, 178, 268)), ('im_t', 'person', 0.9, (120, 120, 250, 260))])
    extra = other_source(make_dets([
        ('im_t', 'cup', 0.7, (152, 241, 180, 269)),        # тот же объект, что у prod
        ('im_t', 'cup', 0.3, (220, 240, 248, 268)),        # новый объект
        ('im_t', 'phone', 0.2, (400, 240, 428, 268)),
        ('im_t', 'person', 0.8, (125, 125, 252, 262)),     # тот же человек
    ]), 'ov_x')
    parts = [assign(prod, rows, {}), assign(extra, rows, {})]
    merged = merge_sources(parts, {})
    assert set(merged.source) == {'all'} and set(merged.origin) == {'prod', 'ov_x'}
    f = aggregate(merged, {'source': 'all'}, rows)
    assert f.loc['q1', ['all__cups__cur', 'all__people__cur']].tolist() == [2, 1]
    assert f.loc['q2', 'all__items__cur'] == 1
    assert merged.loc[(merged.counter == 'cups') & (merged.confidence == 0.9), 'n_src'].iloc[0] == 2
    cfg = {'temporal': {'min_conf': {'ov_x': {'cups': 0.5, 'items': 0.5}}}}
    f = aggregate(merge_sources(parts, cfg), {'source': 'all'}, rows)
    assert f.loc['q1', 'all__cups__cur'] == 1 and f.loc['q2', 'all__items__cur'] == 0
    assert len(merge_sources([parts[0].iloc[:0]], {})) == 0


def test_big_person_exceptions():
    rows = make_rows()
    big = (90, 100, 310, 330)      # 220×230 = 50600 px², накрывает стол 1
    dets = make_dets([('im_t', 'person', 0.9, big), ('im_t', 'hand', 0.9, (150, 240, 178, 268)),
                      ('im_n-30', 'person', 0.9, big), ('im_n30', 'person', 0.9, (95, 105, 312, 333))])
    a = assign(dets, rows, {})
    person = a[(a.counter == 'people') & (a.dt == 0)].iloc[0]
    assert person.persist == 2 and person.ctx_n == 2

    def people(**flags):
        kept = apply_filters(a, {'temporal': {'people_max_area': 45000, **flags}})
        return int(((kept.counter == 'people') & (kept.dt == 0)).sum())

    assert people() == 0
    assert people(people_big_keep_with_hand=True) == 1
    assert people(people_big_keep_persistent=True) == 1
    one = make_rows(dts=(0, 30))
    b = assign(dets[dets.frame_id != 'im_n-30'], one, {})
    kept = apply_filters(b, {'temporal': {'people_max_area': 45000, 'people_big_keep_persistent': True}})
    assert int(((kept.counter == 'people') & (kept.dt == 0)).sum()) == 0   # один сосед: «на обоих» не выполнено


def test_vote_with_rowfeats():
    idx = pd.Index(['a', 'b', 'c'], name='row_id')
    f = pd.DataFrame({'is_ir': 0.0}, index=idx)
    for counter in ['people', 'hands', 'cups', 'dishes', 'items']:
        for stat in ['cur', 'prev', 'next', 'med']:
            f[f'all__{counter}__{stat}'] = 0.0
    f['all__cups__cur'], f['all__cups__med'] = [0.0, 2.0, 1.0], [1.0, 2.0, 1.0]
    f['vlm_qwen__cups'], f['vlm_qwen__items'], f['vlm_qwen__ok'] = [1.0, 0.0, 3.0], [2.0, -1.0, 1.0], [1, 0, 1]
    cfg = {'stack': {'vote': {'cups': {'column': 'vlm_qwen__cups', 'mode': 'median'},
                              'items': {'column': 'vlm_qwen__items', 'mode': 'replace'}}}}
    counts = stack.predict_counts(f, cfg)
    assert counts['cups'].tolist() == [1, 2, 1]      # b: ответа VLM нет, остаётся детектор
    assert counts['items'].tolist() == [2, 0, 1]


def test_tray_mult_counts_trays_extra():
    rows = make_rows(dts=(0,))
    dets = make_dets([('im_t', 'tray', 0.9, (150, 220, 210, 280)), ('im_t', 'tray', 0.9, (220, 220, 280, 280)),
                      ('im_t', 'plate', 0.9, (400, 240, 430, 270))])
    f = aggregate(merge_sources([assign(dets, rows, {})], {}), {'source': 'all'}, rows)
    f['is_ir'] = 0.0
    assert f['all__tray__cur'].to_dict() == {'q1': 2.0, 'q2': 0.0}
    assert stack.predict_counts(f, {})['dishes'].to_dict() == {'q1': 2.0, 'q2': 1.0}
    assert stack.predict_counts(f, {'post': {'tray_mult': 2.0}})['dishes'].to_dict() == {'q1': 4.0, 'q2': 1.0}
    assert stack.predict_counts(f, {'post': {'tray_mult': 3.0}})['dishes'].to_dict() == {'q1': 6.0, 'q2': 1.0}


def test_compose_takes_baseline_columns():
    from conftest import HAS_DATA
    if not HAS_DATA:
        pytest.skip('нет каталога данных TS_DATA')
    from ts import io, orgbase
    labels = io.load_labels()
    fit, val = labels[labels.date_utc == '2026-10-25'], labels[labels.date_utc == '2026-10-26']
    cfg = {'sources': {'dets': ['prod'], 'rowfeats': []}, 'post': {'tray_mult': 2.0},
           'compose': {'people': 'stack', 'dishes': 'stack', 'hands': 'baseline', 'cups': 'baseline',
                       'items': 'baseline', 'occ_prob': 'baseline'}}
    tables = io.load_tables()
    pred = stack.make_fit_predict(cfg)(fit, tables[tables.row_id.isin(val.row_id)]).set_index('row_id')
    base = orgbase.calibrated(fit, list(pred.index)).set_index('row_id')
    for column in ['hands', 'cups', 'items']:
        assert (pred[column] == base[column]).all()
    assert np.allclose(pred['occ_prob'], base['occ_prob'], atol=1e-12)
    raw = orgbase.features().loc[pred.index]
    assert (pred['dishes'] == 2 * raw['det_dishes']).all()          # подносы × 2
    assert (pred['people'] >= raw['det_people']).all()              # добор только добавляет людей
    with pytest.raises(ValueError, match='compose'):
        stack.make_fit_predict({**cfg, 'compose': {'people': 'magic'}})(fit, tables[tables.row_id.isin(val.row_id)])
