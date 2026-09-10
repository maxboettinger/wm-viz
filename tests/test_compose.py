"""wmviz/compose.py — PIL/numpy post-processing (no bpy)."""
import numpy as np
import pytest

from wmviz.compose import grid, hstack, hud, label, pip, split, strip, to_rgb, upscale, vstack


def _img(h, w, v):
    return np.full((h, w, 3), v, dtype=np.uint8)


def test_to_rgb_drops_alpha_and_promotes_grey():
    assert to_rgb(np.zeros((4, 5, 4), np.uint8)).shape == (4, 5, 3)
    assert to_rgb(np.zeros((4, 5), np.uint8)).shape == (4, 5, 3)


def test_hstack_pads_and_gaps():
    out = hstack([_img(10, 4, 1), _img(6, 3, 2)], gap=2, gap_color=(9, 9, 9))
    assert out.shape == (10, 9, 3)
    assert (out[:, 4:6] == 9).all() and (out[:6, 6:] == 2).all() and (out[6:, 6:] == 9).all()


def test_vstack_and_grid_shapes():
    assert vstack([_img(3, 4, 1), _img(2, 4, 2)], gap=1).shape == (6, 4, 3)
    g = grid([_img(2, 2, 1)] * 5, cols=2, gap=0)
    assert g.shape == (6, 4, 3) and (g[4:, 2:] == 0).all()     # 3 rows, last tile padded black


def test_label_adds_a_bar_with_text():
    out = label(_img(10, 40, 0), "step 12", height=12)
    assert out.shape == (22, 40, 3)
    assert out[:12].max() > 100 and (out[12:] == 0).all()    # text pixels in the bar, image untouched


def test_hud_draws_in_a_corner_only():
    base = _img(60, 120, 0)
    out = hud(base, ["step 3/10", "return 0.0"], corner="tl")
    assert out.shape == base.shape and out[:30, :60].max() > 0
    assert (out[:, 150:] == 0).all() and (out[62:, :] == 0).all()      # box stays in the top-left


def test_upscale_nearest():
    out = upscale(_img(2, 2, 7), (8, 4))
    assert out.shape == (4, 8, 3) and (out == 7).all()


def test_pip_places_insets_in_the_corner():
    base = _img(100, 200, 0)
    out = pip(base, [(_img(8, 8, 200), "real"), (_img(8, 8, 100), "dream")], corner="tr", frac=0.2, border=(255, 0, 0))
    assert out.shape == base.shape
    assert out[:40, 100:].max() == 255 and (out[60:, :100] == 0).all()
    assert (out[:, :, 0] == 255).sum() > (out[:, :, 1] == 255).sum()        # red border pixels present


def test_split_upscales_right_and_colours_divider():
    out = split(_img(60, 90, 1), _img(6, 6, 2), divider=(0, 0, 255), divider_px=4)
    assert out.shape == (60, 90 + 4 + 60, 3)
    assert (out[:, 90:94] == (0, 0, 255)).all() and (out[:, 94:] == 2).all()


def test_strip_labels_each_image():
    out = strip([_img(10, 10, 1), _img(10, 10, 2)], ["a", "b"], gap=0)
    assert out.shape[1] == 20 and out.shape[0] > 10
