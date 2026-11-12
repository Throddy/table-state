import json

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from ts import COUNTS, io
from ts.check import CheckError, check_dets, check_pred, check_rowfeats

pytestmark = needs_data

CARD = {'name': 'x', 'kind': 'dets', 'owner': 'A', 'manual_labels_used': False, 'uses_train_counters': False}


def write_card(path, **extra):
    path.with_suffix('.json').write_text(json.dumps({**CARD, **extra}), encoding='utf-8')


# ---------- предсказания ----------

@pytest.fixture
def pred():
    ids = io.load_tables().row_id
    rng = np.random.default_rng(419)
    df = pd.DataFrame({'row_id': ids})
    for c in COUNTS:
        df[c] = rng.integers(0, 3, len(ids)).astype(float)
    df['occ_prob'] = rng.random(len(ids))
    return df


def save_pred(tmp_path, df, **card):
    path = tmp_path / 'p.csv'
    df.to_csv(path, index=False)
    write_card(path, train_pred='zero_shot', **card)
    return path


def test_pred_ok(tmp_path, pred):
    check_pred(save_pred(tmp_path, pred))


@pytest.mark.parametrize('breaker, message', [
    (lambda d: d.iloc[:-10], 'не хватает 10'),
    (lambda d: pd.concat([d, d.iloc[:1]]), 'уникальным'),
    (lambda d: d.assign(occ_prob=d.occ_prob + 0.5), 'occ_prob'),
    (lambda d: d.assign(cups=-1.0), 'счётчики'),
    (lambda d: d.assign(hands=np.where(d.index == 7, np.nan, d.hands)), 'NaN'),
    (lambda d: d.drop(columns='items'), 'колонки'),
    (lambda d: d.assign(extra=1), 'колонки'),
    (lambda d: d.assign(people='много'), 'числовой'),
])
def test_pred_broken(tmp_path, pred, breaker, message):
    with pytest.raises(CheckError, match=message):
        check_pred(save_pred(tmp_path, breaker(pred)))


def test_pred_card_rules(tmp_path, pred):
    path = tmp_path / 'p.csv'
    pred.to_csv(path, index=False)
    with pytest.raises(CheckError, match='карточки'):
        check_pred(path)
    with pytest.raises(CheckError, match='oof_by_date'):
        check_pred(save_pred(tmp_path, pred, uses_train_counters=True))


# ---------- детекции ----------

@pytest.fixture
def dets():
    return io.load_prod_dets()


def save_dets(tmp_path, df, name='prod__v1.parquet', **card):
    path = tmp_path / name
    df.to_parquet(path, index=False)
    write_card(path, **{'coverage': 'all', 'frames_processed': {'n': 1718, 'sha1': 'x'}, **card})
    return path


def test_dets_ok(tmp_path, dets):
    check_dets(save_dets(tmp_path, dets))
    crop = dets.head(50).assign(from_row_id='q_x', from_dt=[-30, 0, 30, 0, 0] * 10)
    check_dets(save_dets(tmp_path, crop, coverage='train_only',
                         frames_processed={'n': crop.frame_id.nunique(), 'sha1': 'x'}))


@pytest.mark.parametrize('breaker, message', [
    (lambda d: d.assign(class_name=d.class_name.replace('hand', 'nude_hands')), 'словаря'),
    (lambda d: d.assign(x2=d.x1), 'x2 > x1'),
    (lambda d: d.assign(x2=d.x2 + 2000), 'за кадр'),
    (lambda d: d.assign(confidence=d.confidence * 100), 'confidence'),
    (lambda d: d.assign(det_id='same'), 'det_id'),
    (lambda d: d.assign(frame_id='im_unknown'), 'frame_id'),
    (lambda d: d.assign(source='other'), 'source'),
    (lambda d: d.assign(from_dt=15), 'from_dt'),
    (lambda d: d.drop(columns='from_row_id'), 'нет колонок'),
])
def test_dets_broken(tmp_path, dets, breaker, message):
    with pytest.raises(CheckError, match=message):
        check_dets(save_dets(tmp_path, breaker(dets)))


def test_dets_card_rules(tmp_path, dets):
    with pytest.raises(CheckError, match='1718'):
        check_dets(save_dets(tmp_path, dets, frames_processed={'n': 39, 'sha1': 'x'}))
    path = tmp_path / 'nocard__v1.parquet'
    dets.assign(source='nocard').to_parquet(path, index=False)
    with pytest.raises(CheckError, match='карточки'):
        check_dets(path)
    bad_name = tmp_path / 'prod.parquet'
    dets.to_parquet(bad_name, index=False)
    with pytest.raises(CheckError, match='имя файла'):
        check_dets(bad_name)


# ---------- признаки строк ----------

@pytest.fixture
def rowfeats():
    ids = io.load_tables().row_id
    df = pd.DataFrame({'row_id': ids, 'dt': 0})
    for c in COUNTS:
        df[f'vlm_qwen__{c}'] = 0.0
    df['vlm_qwen__ok'] = 1
    return df


def save_rowfeats(tmp_path, df):
    path = tmp_path / 'vlm_qwen__v1.parquet'
    df.to_parquet(path, index=False)
    write_card(path)
    return path


def test_rowfeats_ok(tmp_path, rowfeats):
    check_rowfeats(save_rowfeats(tmp_path, rowfeats))
    with_context = pd.concat([rowfeats, rowfeats.head(100).assign(dt=-30)])
    check_rowfeats(save_rowfeats(tmp_path, with_context))


@pytest.mark.parametrize('breaker, message', [
    (lambda d: d.iloc[5:], 'dt=0'),
    (lambda d: d.assign(vlm_qwen__cups=np.nan), 'NaN'),
    (lambda d: d.assign(other=1.0), 'признаков'),
    (lambda d: d.assign(dt=10), 'dt'),
    (lambda d: d.drop(columns='vlm_qwen__ok'), 'нет колонок'),
    (lambda d: d.assign(vlm_qwen__ok=2), '__ok'),
    (lambda d: pd.concat([d, d.head(3)]), 'уникальны'),
])
def test_rowfeats_broken(tmp_path, rowfeats, breaker, message):
    with pytest.raises(CheckError, match=message):
        check_rowfeats(save_rowfeats(tmp_path, breaker(rowfeats)))
