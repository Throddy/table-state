import pandas as pd
import pytest

from conftest import needs_data
from ts import io
from ts.check import CheckError, check_rows_ext
from ts.rows import ROWS_EXT_COLUMNS, load_rows_ext

pytestmark = needs_data


@pytest.fixture(scope='module')
def rows():
    return load_rows_ext()


def test_columns_and_size(rows):
    assert list(rows.columns) == ROWS_EXT_COLUMNS
    assert int((rows.dt == 0).sum()) == 4964
    assert 14700 <= len(rows) <= 14900
    assert not rows.isna().any().any()
    assert not rows.duplicated(['row_id', 'dt']).any()


def test_values(rows):
    assert set(rows.dt.unique()) == {-30, 0, 30}
    assert set(rows.split.unique()) == {'train', 'test'}
    assert rows.slot.between(1, 6).all() and rows.hour_msk.between(0, 23).all()
    assert set(rows.is_ir.unique()) <= {0, 1}
    cur = rows[rows.dt == 0]
    assert (cur.frame_id == cur.src_frame_id).all()
    assert int((cur.split == 'train').sum()) == 231 and int((cur.split == 'test').sum()) == 4733
    assert (cur.groupby('frame_id').size() == cur.groupby('frame_id').n_tables.first()).all()


def test_neighbours_keep_target_box_and_context(rows):
    assert rows.groupby('row_id')[['x1', 'y1', 'x2', 'y2', 'slot']].nunique().max().max() == 1
    targets = io.load_targets().set_index('frame_id')
    per_row = rows.groupby('row_id').agg(frame_id=('frame_id', 'first'), n=('dt', 'size'))
    assert (per_row.n == 1 + per_row.frame_id.map(targets.context.map(len))).all()
    # строки с одним соседом это штатная ситуация
    assert int((per_row.n == 2).sum()) > 0 and int((per_row.n == 3).sum()) > 0
    pairs = rows[rows.dt != 0]
    assert all(src in targets.at[frame, 'context'] for frame, src in zip(pairs.frame_id, pairs.src_frame_id))


def test_matches_tables_csv(rows):
    tables = pd.read_csv(io.paths.DATA / 'tables.csv').set_index('row_id')
    cur = rows[rows.dt == 0].set_index('row_id')
    assert (cur.loc[tables.index, ['x1', 'y1', 'x2', 'y2']].to_numpy() == tables[['x1', 'y1', 'x2', 'y2']].to_numpy()).all()


def test_check_rows_ext(rows, tmp_path):
    good = tmp_path / 'rows_ext.parquet'
    rows.to_parquet(good, index=False)
    check_rows_ext(good)
    bad = tmp_path / 'bad.parquet'
    rows[~((rows.dt == 0) & (rows.index % 100 == 0))].to_parquet(bad, index=False)
    with pytest.raises(CheckError, match='4964'):
        check_rows_ext(bad)
