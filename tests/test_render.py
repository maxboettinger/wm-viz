"""wmviz/render.py — end-to-end smoke renders (needs bpy; a few seconds each)."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
pytestmark = pytest.mark.slow

from wmviz.render import RenderConfig, build, render_episode, render_frames, render_still, write_mp4  # noqa: E402
from wmviz.trace import Episode, Index  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _ep():
    idx = Index.load(FIX / "doorkey6x6")
    row = idx.rows[0]
    return Episode.load(idx.path_of(row)), row


def test_render_config_preview_overrides():
    cfg = RenderConfig(out=Path("x.mp4"), preview=True)
    assert cfg.resolution == (960, 540) and cfg.anim.frames_per_step == 3
    assert RenderConfig(out=Path("x.mp4")).resolution == (1920, 1080)
    assert cfg.frames_dir("topdown") == Path("x_frames") / "topdown"


def test_build_creates_cameras_and_frame_range():
    ep, _ = _ep()
    cfg = RenderConfig(out=Path("x.mp4"), cameras=("topdown", "fpv"), preview=True)
    sc, cams, last = build(ep, cfg)
    assert set(cams) == {"topdown", "fpv"} and last == 1 + ep.length * 3
    assert bpy.context.scene.frame_end == last and sc.agent is not None


def test_still_writes_a_png(tmp_path):
    ep, row = _ep()
    cfg = RenderConfig(out=tmp_path / "ep.png", preview=True, still=2, hud=True)
    out = render_episode(ep, row, cfg)
    assert out == tmp_path / "ep.png" and out.exists()
    img = render_still(ep, row, replace(cfg, out=tmp_path / "still.png"), 2)   # own frames dir: no resume
    assert img.shape == (540, 960, 3) and img.max() > 20


def test_no_render_saves_blend_only(tmp_path):
    ep, row = _ep()
    cfg = RenderConfig(out=tmp_path / "ep.mp4", no_render=True, save_blend=tmp_path / "ep.blend")
    out = render_episode(ep, row, cfg)
    assert out == tmp_path / "ep.blend" and out.exists() and not (tmp_path / "ep.mp4").exists()


def test_short_animation_resumes_and_writes_mp4(tmp_path):
    ep, row = _ep()
    # 2 frames per step, only the first 3 steps: keep the smoke test under ~10 s
    cfg = RenderConfig(out=tmp_path / "ep.mp4", res=(160, 90), frames_per_step=2, samples=4)
    frames = list(range(1, 1 + 3 * 2))
    build(ep, cfg)
    cam = bpy.context.scene.camera
    paths = render_frames(cam, frames, cfg.frames_dir("topdown"))
    assert len(paths) == 6 and all(p.exists() for p in paths)
    mtimes = [p.stat().st_mtime_ns for p in paths]
    assert render_frames(cam, frames, cfg.frames_dir("topdown")) == paths
    assert [p.stat().st_mtime_ns for p in paths] == mtimes             # resumed: nothing re-rendered
    out = write_mp4((np.zeros((90, 160, 3), np.uint8) for _ in range(6)), tmp_path / "z.mp4", fps=12)
    assert out.exists() and out.stat().st_size > 0


def test_resume_wipes_frames_of_a_different_config(tmp_path):
    ep, row = _ep()
    cfg = RenderConfig(out=tmp_path / "ep.png", res=(160, 90), frames_per_step=2, samples=4)
    for step in (0, 1):                                              # frames f00001, f00003
        assert render_still(ep, row, cfg, step).shape == (90, 160, 3)
    old = sorted(cfg.frames_dir("topdown").glob("f*.png"))
    assert [p.name for p in old] == ["f00001.png", "f00003.png"]
    stamp = cfg.frames_root() / "render.json"
    assert json.loads(stamp.read_text()) == cfg.fingerprint()
    # same config → resume: nothing re-rendered
    mtimes = [p.stat().st_mtime_ns for p in old]
    render_still(ep, row, cfg, 1)
    assert [p.stat().st_mtime_ns for p in old] == mtimes
    # changed resolution → the whole frames root is replaced, new frames have the new size
    cfg2 = replace(cfg, res=(320, 180))
    assert render_still(ep, row, cfg2, 1).shape == (180, 320, 3)
    assert not (cfg.frames_dir("topdown") / "f00001.png").exists()
    assert [p.name for p in sorted(cfg2.frames_dir("topdown").glob("f*.png"))] == ["f00003.png"]
    assert json.loads(stamp.read_text()) == cfg2.fingerprint() != cfg.fingerprint()


def test_render_frames_ignores_tmp_and_empty_files(tmp_path):
    """A `.tmp.png` (killed mid-write) never becomes the final frame and a 0-byte PNG is re-rendered."""
    ep, _ = _ep()
    cfg = RenderConfig(out=tmp_path / "ep.mp4", res=(160, 90), frames_per_step=2, samples=4)
    build(ep, cfg)
    d = cfg.frames_dir("topdown")
    d.mkdir(parents=True)
    (d / "f00001.tmp.png").write_bytes(b"junk")
    (d / "f00002.png").write_bytes(b"")
    paths = render_frames(bpy.context.scene.camera, [1, 2], d)
    assert [p.name for p in paths] == ["f00001.png", "f00002.png"]
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 100 for p in paths)
    assert sorted(q.name for q in d.iterdir()) == ["f00001.png", "f00002.png"]     # no .tmp.png left behind
