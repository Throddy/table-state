import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from conftest import ROOT
from ts import crops, geometry, io

TABLES = {1: (137, 244, 251, 319), 3: (482.3, 188.7, 639.1, 289.4), 6: (1176, 242, 1280, 351)}
MARK = (500, 200, 532, 228)   # красный прямоугольник на синтетическом кадре


@pytest.fixture
def fake_frame(monkeypatch):
    image = Image.new('RGB', (1280, 720), (40, 40, 40))
    ImageDraw.Draw(image).rectangle([MARK[0], MARK[1], MARK[2] - 1, MARK[3] - 1], fill=(255, 0, 0))
    monkeypatch.setattr(io, 'read_image', lambda frame_id: image.copy())
    return image


def test_zone_definitions():
    box = (100.0, 200.0, 300.0, 300.0)   # w=200, h=100
    assert geometry.zone(box, 'top', 3) == pytest.approx((84, 192, 316, 308))
    assert geometry.zone(box, 'seat', 3) == pytest.approx((10, 90, 390, 390))
    assert geometry.zone(box, 'wide', 3) == pytest.approx((10, 90, 390, 390))
    # seat может выходить за кадр, wide обрезается
    edge = (1176.0, 242.0, 1280.0, 351.0)
    assert geometry.zone(edge, 'seat', 6)[2] > 1280
    assert geometry.zone(edge, 'wide', 6)[2] == 1280
    assert geometry.zone((137, 60, 251, 135), 'wide', 1)[1] == 0
    with pytest.raises(ValueError):
        geometry.zone(box, 'unknown', 1)


def test_zone_defaults_match_config():
    cfg = yaml.safe_load((ROOT / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
    for kind in ['top', 'seat', 'wide']:
        assert cfg['zones'][kind] == geometry.ZONE_PARAMS[kind]


@pytest.mark.parametrize('slot', sorted(TABLES))
@pytest.mark.parametrize('kind, scale', [('top', 4.0), ('wide', 2.5), ('seat', 3.0)])
def test_roundtrip(fake_frame, slot, kind, scale):
    crop, meta = crops.crop_table('im_fake', TABLES[slot], kind=kind, scale=scale, slot=slot)
    assert crop.size == (meta.width, meta.height)
    assert 0 <= meta.x0 < meta.x1 <= 1280 and 0 <= meta.y0 < meta.y1 <= 720
    assert meta.sx == pytest.approx(scale, rel=0.02) and meta.sy == pytest.approx(scale, rel=0.02)
    rng = np.random.default_rng(419)
    xy = rng.uniform([meta.x0, meta.y0], [meta.x1 - 30, meta.y1 - 30], size=(50, 2))
    boxes = np.hstack([xy, xy + rng.uniform(5, 30, size=(50, 2))])
    in_crop = crops.to_crop_coords(boxes, meta)
    assert in_crop.min() >= -1e-6
    assert (in_crop[:, [0, 2]].max() <= meta.width + 1e-6) and (in_crop[:, [1, 3]].max() <= meta.height + 1e-6)
    np.testing.assert_allclose(crops.to_frame_coords(in_crop, meta), boxes, atol=1e-9)


def test_pixels_map_back_to_frame(fake_frame):
    crop, meta = crops.crop_table('im_fake', TABLES[3], kind='top', scale=4.0, slot=3)
    a = np.asarray(crop).astype(int)
    red = (a[..., 0] > 160) & (a[..., 1] < 90) & (a[..., 2] < 90)
    ys, xs = np.nonzero(red)
    found = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]])
    np.testing.assert_allclose(crops.to_frame_coords(found, meta)[0], MARK, atol=1.0)


def test_single_box_and_empty_input(fake_frame):
    _, meta = crops.crop_table('im_fake', TABLES[1])
    assert crops.to_frame_coords(np.array([0, 0, meta.width, meta.height]), meta).tolist() == \
        [[meta.x0, meta.y0, meta.x1, meta.y1]]
    assert crops.to_frame_coords(np.zeros((0, 4)), meta).shape == (0, 4)
