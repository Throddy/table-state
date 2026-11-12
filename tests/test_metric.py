import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, needs_data
from ts import COUNTS, io, paths
from ts.cv import FOLDS, evaluate
from ts.metric import fast_score, score_frame

pytestmark = needs_data


def test_vendor_copy_is_byte_identical():
    assert (ROOT / 'ts' / 'vendor' / 'score.py').read_bytes() == (paths.DATA / 'score.py').read_bytes()


@pytest.fixture
def labels_pred():
    labels = io.load_labels()
    rng = np.random.default_rng(419)
    pred = labels[['row_id'] + COUNTS].copy()
    pred[COUNTS] = (pred[COUNTS] + rng.integers(-1, 2, (len(pred), 5))).clip(0, 50).astype(float)
    pred['occ_prob'] = rng.random(len(pred)) * 0.5 + 0.5 * (labels.people > 0)
    return labels, pred


def test_perfect_and_zero(labels_pred):
    labels, _ = labels_pred
    perfect = labels[['row_id'] + COUNTS].assign(occ_prob=labels.occupied.astype(float))
    assert score_frame(labels, perfect)['score'] == pytest.approx(1.0)
    zeros = perfect.assign(**{c: 0.0 for c in COUNTS}, occ_prob=0.0)
    result = score_frame(labels, zeros)
    assert result['score'] == pytest.approx(0.2 * 23 / 231)
    assert result['objects'] == {'people': 29, 'hands': 29, 'cups': 21, 'dishes': 19, 'items': 28}


def test_fast_score_equals_vendor(labels_pred):
    labels, pred = labels_pred
    for part in [labels, labels[labels.date_utc == '2026-10-25'], labels[labels.people == 0]]:
        merged = part.merge(pred, on='row_id', suffixes=('', '_p'))
        fast = fast_score(merged[COUNTS].to_numpy(float), merged[[c + '_p' for c in COUNTS]].to_numpy(float),
                          merged.occ_prob.to_numpy(float))
        assert fast == pytest.approx(score_frame(part, pred)['score'], abs=1e-12)


def test_evaluate_report(labels_pred):
    labels, pred = labels_pred
    report = evaluate(pred, bootstrap=50)
    assert report['oof_all']['rows'] == 231 and report['f25']['rows'] == 105 and report['f26']['rows'] == 126
    assert report['robust']['oof_all']['rows'] == 226
    assert report['ir1']['oof_all']['rows'] + report['ir0']['oof_all']['rows'] == 231
    assert report['bootstrap']['p05'] <= report['oof_all']['score'] + 0.1
    assert report['bootstrap']['p05'] < report['bootstrap']['p95']
    assert evaluate(pred, bootstrap=50)['bootstrap'] == report['bootstrap']   # сид 419


def test_evaluate_with_fit_predict(labels_pred):
    labels, pred = labels_pred
    seen = []

    def fit_predict(train_rows, predict_rows):
        seen.append((set(train_rows.date_utc), set(predict_rows.date_utc)))
        assert not set(train_rows.frame_id) & set(predict_rows.frame_id)
        assert not set(COUNTS) & set(predict_rows.columns)
        return pred[pred.row_id.isin(predict_rows.row_id)]

    report = evaluate(fit_predict, bootstrap=0)
    assert seen == [({fit}, {val}) for fit, val in FOLDS.values()]
    assert report['oof_all']['score'] == pytest.approx(score_frame(labels, pred)['score'])
