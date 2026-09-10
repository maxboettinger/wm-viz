"""wmviz/overlays.py — trail and heatmap (needs bpy)."""
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
pytestmark = pytest.mark.slow

from mathutils import Vector  # noqa: E402

from wmviz.aggregate import visit_counts  # noqa: E402
from wmviz.animate import AnimConfig, animate, step_frame  # noqa: E402
from wmviz.overlays import add_heatmap, add_trail, heat_color, set_heatmap_static, time_color  # noqa: E402
from wmviz.render import RenderConfig, build  # noqa: E402
from wmviz.scene.base import reset_scene  # noqa: E402
from wmviz.scene.minigrid import build_scene, cell_center  # noqa: E402
from wmviz.trace import Episode, Index  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _ep():
    """First fixture episode whose agent actually moves (episodes 0, 1 and 6 spin in place for 360
    steps, which is a valid zero-segment trail but nothing to assert on)."""
    idx = Index.load(FIX / "doorkey6x6")
    for row in idx.rows:
        ep = Episode.load(idx.path_of(row))
        if (np.diff(ep.agent_pos, axis=0) != 0).any():
            return ep
    raise AssertionError("fixture has no moving episode")


def test_colour_ramps():
    assert time_color(0.0) != time_color(1.0) and len(time_color(0.5)) == 4
    assert heat_color(0.0) == (0.0, 0.0, 0.0, 1.0) and heat_color(1.0)[0] == 1.0 and heat_color(1.0)[1] > 0.8


def test_trail_segments_join_cell_centres_and_appear_in_time():
    reset_scene()
    ep = _ep()
    cfg = AnimConfig(frames_per_step=4)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    segs = add_trail(sc, ep, cfg)
    moves = [t for t in range(1, ep.length + 1) if tuple(ep.agent_pos[t]) != tuple(ep.agent_pos[t - 1])]
    assert len(segs) == len(moves)
    seg, t = segs[0], moves[0]
    a = seg.matrix_world @ Vector((0, 0, -0.5))
    b = seg.matrix_world @ Vector((0, 0, 0.5))
    ends = {tuple(round(v, 2) for v in a), tuple(round(v, 2) for v in b)}
    x0, y0 = (int(v) for v in ep.agent_pos[t - 1]); x1, y1 = (int(v) for v in ep.agent_pos[t])
    assert ends == {tuple(round(v, 2) for v in cell_center(x0, y0, 0.12)), tuple(round(v, 2) for v in cell_center(x1, y1, 0.12))}
    bpy.context.scene.frame_set(step_frame(t, cfg) - 1)
    assert seg.hide_render
    bpy.context.scene.frame_set(step_frame(t, cfg))
    assert not seg.hide_render
    assert tuple(segs[0].color) != tuple(segs[-1].color)


def test_heatmap_keyframes_tile_colours():
    reset_scene()
    ep = _ep()
    cfg = AnimConfig(frames_per_step=4)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    add_heatmap(sc, ep, cfg)
    start = tuple(int(v) for v in ep.agent_pos[0])
    tile = sc.floor[start]
    bpy.context.scene.frame_set(1)
    assert tile.color[0] > 0.0
    never = next(c for c in sc.floor if not (ep.agent_pos == np.array(c)).all(axis=1).any())
    bpy.context.scene.frame_set(step_frame(ep.length, cfg))
    assert tuple(sc.floor[never].color[:3]) == (0.0, 0.0, 0.0)


def test_static_heatmap_and_build_wiring():
    reset_scene()
    ep = _ep()
    sc = build_scene(ep.layout)
    counts = visit_counts(ep.agent_pos, ep.layout.width, ep.layout.height)
    set_heatmap_static(sc, counts)
    hot = max(sc.floor, key=lambda c: counts[c])
    assert sc.floor[hot].color[0] == pytest.approx(1.0)
    sc2, _, _ = build(ep, RenderConfig(out=Path("x.mp4"), preview=True, trail=True, heatmap=True))
    assert "Trail" in bpy.data.collections and len(bpy.data.collections["Trail"].objects) > 0
