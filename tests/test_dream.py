"""Dream episodes: reader block, panels, compositing (no bpy)."""
import numpy as np

from wmviz.render import RenderConfig, compose_frame, dream_panels
from wmviz.trace import Episode, Index


def _dream(run_dir):
    idx = Index.load(run_dir)
    row = next(r for r in idx.rows if r.phase == "dream")
    return idx, row, Episode.load(idx.path_of(row))


def test_reader_exposes_dream_block(run_dir):
    _, row, ep = _dream(run_dir)
    assert ep.dream is not None and ep.dream["dream_start"] == 2
    assert ep.dream["dream_frames"].shape[0] == ep.length + 2 and ep.dream["recon_frames"].shape == (2, 8, 8, 3)
    assert ep.obs.shape == (ep.length + 1, 8, 8, 3) and row.seed == 42


def test_dream_panels_switch_at_dream_start(run_dir):
    _, _, ep = _dream(run_dir)
    real, model, dreaming = dream_panels(ep, 1)
    assert real.shape == (8, 8, 3) and model[0, 0, 0] == 90 and dreaming is False
    real, model, dreaming = dream_panels(ep, 2)
    assert model[0, 0, 0] == 200 and dreaming is True
    real, model, dreaming = dream_panels(ep, ep.length)
    assert model is not None and dreaming is True


def test_compose_frame_pip_and_split(run_dir, tmp_path):
    _, row, ep = _dream(run_dir)
    # 216x384, not 108x192: two stacked pip insets at frac=0.25 need 106 px + the 8 px margin,
    # which does not fit a 108 px tall base (pip raises "does not fit").
    base = np.zeros((216, 384, 3), np.uint8)
    cfg = RenderConfig(out=tmp_path / "x.mp4", dream_layout="pip")
    tracking = compose_frame([base], ep, row, 1, cfg)
    dreaming = compose_frame([base], ep, row, 3, cfg)
    assert tracking.shape == base.shape and tracking.max() == 255       # white inset border while tracking
    assert (dreaming[:, :, 0] == 255).sum() < (tracking[:, :, 0] == 255).sum()   # black border while dreaming
    cfg = RenderConfig(out=tmp_path / "x.mp4", dream_layout="split")
    out = compose_frame([base], ep, row, 3, cfg)
    assert out.shape[0] == 216 and out.shape[1] > 384 + 100
    assert (out[:, 384:390] == 0).all()                                  # black divider while dreaming
    out = compose_frame([base], ep, row, 1, cfg)
    assert (out[:, 384:390] == 255).all()                                # white divider while tracking


def test_non_dream_episode_is_untouched(run_dir, tmp_path):
    idx = Index.load(run_dir)
    row = idx.rows[0]
    ep = Episode.load(idx.path_of(row))
    base = np.zeros((20, 30, 3), np.uint8)
    assert (compose_frame([base], ep, row, 1, RenderConfig(out=tmp_path / "x.mp4")) == base).all()
