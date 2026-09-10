"""wmviz/cameras.py — camera presets (needs bpy)."""
import math
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")
pytestmark = pytest.mark.slow

from mathutils import Vector  # noqa: E402

from wmviz.animate import AnimConfig, animate, step_frame  # noqa: E402
from wmviz.cameras import PRESETS, add_camera, look_at  # noqa: E402
from wmviz.scene.base import reset_scene  # noqa: E402
from wmviz.scene.minigrid import build_scene, cell_center  # noqa: E402
from wmviz.trace import Episode, Index  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _ep() -> Episode:
    idx = Index.load(FIX / "doorkey6x6")
    return Episode.load(idx.path_of(idx.rows[0]))


def _view_dir(cam) -> Vector:
    """The camera's -Z axis (view direction) in world space."""
    return cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))


def test_presets_and_unknown():
    assert PRESETS == ("topdown", "follow", "fpv", "orbit", "iso")
    reset_scene()
    ep = _ep()
    with pytest.raises(ValueError):
        add_camera("drone", ep.layout, ep, AnimConfig())


def test_static_presets_accept_no_episode_moving_ones_refuse():
    reset_scene()
    ep = _ep()
    for preset in ("topdown", "iso"):
        cam = add_camera(preset, ep.layout, None, AnimConfig())
        assert cam.name == f"Cam_{preset}" and cam.animation_data is None
    for preset in ("follow", "fpv", "orbit"):
        with pytest.raises(ValueError, match=f"camera '{preset}' needs an episode"):
            add_camera(preset, ep.layout, None, AnimConfig())


def test_look_at_points_minus_z_at_target():
    reset_scene()
    bpy.ops.object.empty_add(location=(1.0, 2.0, 3.0))
    obj = bpy.context.active_object
    look_at(obj, (4.0, -2.0, 0.0))
    bpy.context.scene.frame_set(1)
    to_target = (Vector((4.0, -2.0, 0.0)) - obj.location).normalized()
    assert _view_dir(obj).dot(to_target) == pytest.approx(1.0, abs=1e-3)


def test_topdown_is_orthographic_and_centred():
    reset_scene()
    ep = _ep()
    cam = add_camera("topdown", ep.layout, ep, AnimConfig())
    assert cam.data.type == "ORTHO" and bpy.context.scene.camera is cam
    W, H = ep.layout.width, ep.layout.height
    assert (round(cam.location.x, 3), round(cam.location.y, 3)) == (W / 2, -H / 2)
    assert cam.data.ortho_scale >= max(W, H)


def test_iso_looks_at_the_centre():
    reset_scene()
    ep = _ep()
    cam = add_camera("iso", ep.layout, ep, AnimConfig())
    W, H = ep.layout.width, ep.layout.height
    bpy.context.scene.frame_set(1)
    to_centre = (Vector((W / 2, -H / 2, 0)) - cam.location).normalized()
    assert _view_dir(cam).dot(to_centre) == pytest.approx(1.0, abs=1e-3)


def test_fpv_tracks_agent_pose():
    reset_scene()
    ep = _ep()
    cfg = AnimConfig(frames_per_step=4)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    cam = add_camera("fpv", ep.layout, ep, cfg)
    assert bpy.context.scene.camera is cam
    for t in (0, ep.length):
        bpy.context.scene.frame_set(step_frame(t, cfg))
        x, y = (int(v) for v in ep.agent_pos[t])
        cx, cy, _ = cell_center(x, y)
        assert (round(cam.location.x, 3), round(cam.location.y, 3)) == (cx, cy)
        assert cam.location.z == pytest.approx(0.5)
        # view direction equals the agent heading in the XY plane
        view = _view_dir(cam)
        d = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}[int(ep.agent_dir[t])]     # y already flipped
        assert (round(view.x, 2), round(view.y, 2)) == d


def test_follow_and_orbit_are_animated():
    reset_scene()
    ep = _ep()
    cfg = AnimConfig(frames_per_step=4)
    sc = build_scene(ep.layout)
    animate(sc, ep, cfg)
    for preset in ("follow", "orbit"):
        cam = add_camera(preset, ep.layout, ep, cfg)
        assert cam.animation_data is not None and cam.animation_data.action is not None
        bpy.context.scene.frame_set(1)
        p0 = tuple(cam.location)
        bpy.context.scene.frame_set(step_frame(ep.length // 2, cfg))
        assert cam.location.z > 0.5 and (preset == "follow" or tuple(cam.location) != p0)


def test_orbit_never_spins_the_long_way():
    """look_at keeps successive Eulers continuous: a full sweep must not wrap through ±π
    (a wrapped key makes LINEAR interpolation spin ~360° backwards over one step)."""
    reset_scene()
    ep = _ep()
    cfg = AnimConfig(frames_per_step=4)
    cam = add_camera("orbit", ep.layout, ep, cfg)
    W, H = ep.layout.width, ep.layout.height
    centre = Vector((W / 2, -H / 2, 0))
    for t in range(ep.length):
        bpy.context.scene.frame_set(step_frame(t, cfg) + cfg.frames_per_step // 2)   # between two keys
        to_centre = (centre - cam.location).normalized()
        assert _view_dir(cam).dot(to_centre) > 0.99, f"orbit camera looks away from the centre after step {t}"
