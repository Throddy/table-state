import pytest

from ts.slots import SLOT_EDGES, slot_of


@pytest.mark.parametrize('cx, slot', [(194, 1), (348, 2), (560, 3), (798, 4), (785, 4), (835, 4), (1068, 5), (1228, 6)])
def test_median_centers(cx, slot):
    assert slot_of(cx) == slot


def test_edges_are_half_open():
    for i, edge in enumerate(SLOT_EDGES[1:-1], start=1):
        assert slot_of(edge - 0.01) == i
        assert slot_of(edge) == i + 1


def test_out_of_range_is_clamped():
    assert slot_of(-5) == 1
    assert slot_of(0) == 1
    assert slot_of(1280) == 6
    assert slot_of(5000) == 6


def test_returns_int():
    assert isinstance(slot_of(560.5), int)
