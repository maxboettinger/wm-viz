"""wmviz/animate.py — keyframes from trace arrays (needs bpy)."""
import math
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
pytestmark = pytest.mark.slow

from wmviz.animate import AnimConfig, animate, frame_step, step_frame, unwrap_yaws  # noqa: E402
from wmviz.scene.base import fcurves, reset_scene  # noqa: E402
from wmviz.scene.minigrid import build_scene, cell_center  # noqa: E402
from wmviz.trace import Episode, Index  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _episode_with_key_and_door() -> Episode:
    idx = Index.load(FIX / "doorkey6x6")
    row = next(r for r in idx.rows if r.first_door_step is not None and r.first_key_step is not None)
    return Episode.load(idx.path_of(row))


def test_step_frame_roundtrip():
    cfg = AnimConfig(frames_per_step=6)
    assert step_frame(0, cfg) == 1 and step_frame(3, cfg) == 19
    assert frame_step(19, cfg) == 3 and frame_step(24, cfg) == 3 and frame_step(25, cfg) == 4


def test_unwrap_yaws_takes_the_short_way():
    y = unwrap_yaws([0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0])   # four left turns
    assert [round(v, 3) for v in y] == [round(v, 3) for v in (0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi)]
    y = unwrap_yaws([0.0, -math.pi / 2, math.pi / 2])                  # right turn, then a 180 (either way, |delta| == pi)
    assert abs(y[2] - y[1]) == pytest.approx(math.pi)


def test_agent_location_and_yaw_follow_the_trace():
    reset_scene()
    ep = _episode_with_key_and_door()
    cfg = AnimConfig(frames_per_step=4)
    sc = build_scene(ep.layout)
    last = animate(sc, ep, cfg)
    assert last == step_frame(ep.length, cfg) == bpy.context.scene.frame_end
    for t in (0, ep.length // 2, ep.length):
        bpy.context.scene.frame_set(step_frame(t, cfg))
        x, y = (int(v) for v in ep.agent_pos[t])
        assert tuple(round(v, 3) for v in sc.agent.location) == cell_center(x, y, 0.25)
        yaw = sc.agent.rotation_euler.z
        expected = {0: 0.0, 1: -math.pi / 2, 2: math.pi, 3: math.pi / 2}[int(ep.agent_dir[t])]
        assert math.cos(yaw) == pytest.approx(math.cos(expected), abs=1e-3)
        assert math.sin(yaw) == pytest.approx(math.sin(expected), abs=1e-3)


def test_discrete_holds_between_steps():
    reset_scene()
    ep = _episode_with_key_and_door()
    cfg = AnimConfig(frames_per_step=6, discrete=True)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    t = next(i for i in range(1, ep.length + 1) if tuple(ep.agent_pos[i]) != tuple(ep.agent_pos[i - 1]))
    bpy.context.scene.frame_set(step_frame(t, cfg) - 1)
    x, y = (int(v) for v in ep.agent_pos[t - 1])
    assert tuple(round(v, 3) for v in sc.agent.location) == cell_center(x, y, 0.25)
    assert all(kp.interpolation == "CONSTANT" for fc in fcurves(sc.agent) for kp in fc.keyframe_points)


def test_door_opens_at_the_right_frame():
    reset_scene()
    ep = _episode_with_key_and_door()
    cfg = AnimConfig(frames_per_step=6)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    j = 0
    t_open = int(np.flatnonzero(ep.door_open[:, j])[0])
    (_, panel), = sc.doors.values()
    closed = panel.rotation_euler.z
    bpy.context.scene.frame_set(step_frame(t_open - 1, cfg))
    assert panel.rotation_euler.z == pytest.approx(closed, abs=1e-4)
    bpy.context.scene.frame_set(step_frame(t_open, cfg))
    assert abs(panel.rotation_euler.z - closed) == pytest.approx(math.pi / 2, abs=1e-3)


def test_key_pickup_hides_floor_key_and_shows_carried_copy():
    reset_scene()
    ep = _episode_with_key_and_door()
    cfg = AnimConfig(frames_per_step=6)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    t_pick = int(np.flatnonzero(ep.carrying != -1)[0])
    (key,), = [list(sc.keys.values())]
    assert sc.carried is not None and sc.carried.parent is sc.agent
    bpy.context.scene.frame_set(step_frame(t_pick, cfg) - 1)
    assert not key.hide_render and sc.carried.hide_render
    bpy.context.scene.frame_set(step_frame(t_pick, cfg))
    assert key.hide_render and not sc.carried.hide_render


def test_floor_key_stays_home_until_the_first_drop():
    """The rest location must be keyed too: an unkeyed location F-curve extrapolates the first drop backwards."""
    reset_scene()
    ep = _episode_with_key_and_door()
    t_drop = int(np.flatnonzero((ep.carrying[:-1] != -1) & (ep.carrying[1:] == -1))[0]) + 1
    ep.agent_dir[t_drop] = 0                       # first drop lands in the cell east of the agent, not on the key's home cell
    cfg = AnimConfig(frames_per_step=6)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    ((home, key),) = sc.keys.items()
    drop = (int(ep.agent_pos[t_drop][0]) + 1, int(ep.agent_pos[t_drop][1]))
    assert drop != home
    bpy.context.scene.frame_set(1)
    assert tuple(round(v, 3) for v in key.location) == cell_center(*home, 0.25)
    bpy.context.scene.frame_set(step_frame(t_drop, cfg))
    assert tuple(round(v, 3) for v in key.location) == cell_center(*drop, 0.25)
