"""wmviz/mpl.py — matplotlib figures (no bpy)."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from wmviz.aggregate import visit_counts  # noqa: E402
from wmviz.mpl import figure_image, heatmap_image, keyframe_steps  # noqa: E402
from wmviz.preview import save_png  # noqa: E402
from wmviz.trace import Episode, Index  # noqa: E402


def _first(run_dir, **match):
    idx = Index.load(run_dir)
    row = next(r for r in idx.rows if all(getattr(r, k) == v for k, v in match.items()))
    return idx, row, Episode.load(idx.path_of(row))


def test_figure_image_shapes_and_content(run_dir):
    _, _, ep = _first(run_dir, phase="eval", seed=1001)
    img = figure_image(ep, cell_px=20)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    assert img.shape[0] >= 7 * 20 and img.shape[1] >= 7 * 20
    partial = figure_image(ep, step=1, trail=True, heatmap=False, cell_px=20)
    assert partial.shape == img.shape and not np.array_equal(partial, img)


def test_heatmap_image(run_dir):
    _, _, ep = _first(run_dir, phase="eval", seed=1001)
    counts = visit_counts(ep.agent_pos, ep.layout.width, ep.layout.height)
    img = heatmap_image(ep.layout, counts, cell_px=16)
    baseline = heatmap_image(ep.layout, np.zeros_like(counts), cell_px=16)
    assert img.shape[2] == 3
    assert not np.array_equal(img, baseline)                           # visited cells actually paint differently


def test_keyframe_steps(run_dir):
    _, row, ep = _first(run_dir, phase="eval", seed=1001)     # has first_door_step=2 and success
    auto = keyframe_steps(row, ep, "auto")
    assert auto[-1][0] == ep.length
    assert any(l.startswith("door") for _, l in auto)                  # door label wins the step it shares with key
    # this fixture's goal landmark sits on the episode's last step, so it keeps its own label over "end"
    assert auto[-1][1] == f"goal · step {ep.length}"
    assert len({s for s, _ in auto}) == len(auto)                      # no duplicate steps
    assert keyframe_steps(row, ep, "0,3") == [(0, "step 0"), (3, "step 3")]
    with pytest.raises(ValueError):
        keyframe_steps(row, ep, "0,abc")
    with pytest.raises(ValueError):
        keyframe_steps(row, ep, ",")


def test_save_png_still_works(run_dir, tmp_path):
    _, _, ep = _first(run_dir, phase="eval", seed=1001)
    out = save_png(ep, tmp_path / "p.png")
    assert out.exists() and out.stat().st_size > 0
