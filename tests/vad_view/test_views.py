"""views.py geometry: padding, clamp, zoom, view set, and the p1 fit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vad_view"))

import views  # noqa: E402


def test_view_set():
    names = [views.view_name(*k) for k in views.VIEW_KEYS]
    assert len(names) == 8
    assert "p0.30-z1.5" not in names
    assert views.V0 == "p1-z2.0" and views.V0 in names


def test_padding_inside():
    # box 100 x 50 at (200, 100); p 0.1 adds 10 px left/right and 5 px top/bottom
    assert views.crop_box([200, 100, 300, 150], (1000, 800), 0.1) == (190, 95, 310, 155)
    assert views.crop_box([200, 100, 300, 150], (1000, 800), 0.0) == (200, 100, 300, 150)


def test_padding_fraction_rounds_outward():
    # box 10 x 10, p 0.105 -> 1.05 px: floor on the left/top, ceil on the right/bottom
    assert views.crop_box([20, 20, 30, 30], (100, 100), 0.105) == (18, 18, 32, 32)


def test_clamp():
    assert views.crop_box([0, 0, 50, 40], (60, 45), 0.3) == (0, 0, 60, 45)
    # a box outside the image is clamped first
    assert views.crop_box([-5, 10, 70, 20], (60, 45), 0.0) == (0, 10, 60, 20)


def test_zoom():
    assert views.zoom_size((0, 0, 101, 51), 1.5) == (152, 76)
    assert views.zoom_size((0, 0, 101, 51), 3.0) == (303, 153)
    img = Image.new("RGB", (400, 300), (10, 20, 30))
    v, crop = views.make_view(img, [100, 100, 200, 150], 0.3, 2.0)
    assert crop == (70, 85, 230, 165)
    assert v.size == (320, 160)
    assert v.getpixel((5, 5)) == (10, 20, 30)


def test_fit_p1():
    p = 0.105
    samples = []
    for i, (bw, bh) in enumerate([(100, 50), (200, 80), (60, 120), (300, 300)]):
        bbox = [500, 500, 500 + bw, 500 + bh]
        crop = views.crop_box(bbox, (2000, 2000), p)
        samples.append({"bbox": bbox, "size": [2000, 2000], "released": list(views.zoom_size(crop, 2.0))})
    # one row at the edge is left out of the fit
    samples.append({"bbox": [0, 0, 100, 100], "size": [2000, 2000], "released": [210, 210]})
    res = views.fit_p1(samples)
    assert res["rows_fit"] == 4 and res["rows_all"] == 5
    assert res["p1"] == pytest.approx(p, abs=0.01)
    assert res["residual_px_max_abs"] < 1.5
