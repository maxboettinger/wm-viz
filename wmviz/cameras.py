"""Camera presets (spec §4): topdown (ortho), follow, fpv, orbit, iso.

Static presets (topdown, iso) frame the layout and need no episode; moving presets
(follow, fpv, orbit) are keyframed at `step_frame(t)` for every state of an Episode.
fpv follows `cfg.discrete`, follow is EMA-smoothed and eased, orbit is linear.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

from .animate import AnimConfig, step_frame, unwrap_yaws
from .scene.base import set_interpolation, set_linear
from .scene.minigrid import AGENT_Z, YAW, cell_center
from .trace.reader import Episode, Layout

PRESETS = ("topdown", "follow", "fpv", "orbit", "iso")
STATIC_PRESETS = ("topdown", "iso")
EYE_Z = 0.5


def look_at(obj, target) -> None:
    """Rotate `obj` so its -Z axis (a camera's view direction) points at `target`.
    The new Euler is made compatible with the current one, so successive calls (keyframes)
    never wrap through ±π and interpolate the short way round."""
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler("XYZ", obj.rotation_euler)


def _new_camera(name: str):
    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = name
    bpy.context.scene.camera = cam
    return cam


def _ema(values, alpha: float):
    out, prev = [], None
    for v in values:
        prev = v if prev is None else tuple(alpha * a + (1 - alpha) * b for a, b in zip(v, prev))
        out.append(prev)
    return out


def _key_pose(cam, frame: int) -> None:
    cam.keyframe_insert("location", frame=frame)
    cam.keyframe_insert("rotation_euler", frame=frame)


def add_camera(preset: str, layout: Layout, ep: Episode | None, cfg: AnimConfig):
    """Create `Cam_<preset>`, keyframe it if the preset moves, make it the scene camera.
    `ep` may be None for the static presets only."""
    if preset not in PRESETS:
        raise ValueError(f"camera must be one of {', '.join(PRESETS)}, got {preset!r}")
    if ep is None and preset not in STATIC_PRESETS:
        raise ValueError(f"camera '{preset}' needs an episode")
    W, H = layout.width, layout.height
    centre = Vector((W / 2, -H / 2, 0.0))
    size = max(W, H)
    cam = _new_camera(f"Cam_{preset}")

    if preset == "topdown":
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = size * 1.05
        cam.location = (W / 2, -H / 2, 20.0)
        cam.rotation_euler = (0.0, 0.0, 0.0)
        return cam

    if preset == "iso":
        cam.data.lens = 35
        cam.location = centre + Vector((0.9 * size, -0.9 * size, 0.9 * size))
        look_at(cam, centre)
        return cam

    if preset == "orbit":
        cam.data.lens = 35
        r = 1.1 * size
        for t in range(ep.length + 1):
            a = 2 * math.pi * t / max(ep.length, 1) - math.pi / 4
            cam.location = centre + Vector((r * math.cos(a), r * math.sin(a), 0.75 * r))
            look_at(cam, centre)
            _key_pose(cam, step_frame(t, cfg))
        set_linear(cam)
        return cam

    yaws = unwrap_yaws([YAW[int(d)] for d in ep.agent_dir])
    if preset == "fpv":
        cam.data.lens = 18
        for t in range(ep.length + 1):
            x, y = (int(v) for v in ep.agent_pos[t])
            cam.location = cell_center(x, y, EYE_Z)
            cam.rotation_euler = (math.pi / 2, 0.0, yaws[t] - math.pi / 2)   # yaw 0 looks along +X
            _key_pose(cam, step_frame(t, cfg))
        set_interpolation(cam, cfg.discrete)
        return cam

    # follow: behind and above the agent, smoothed
    cam.data.lens = 28
    raw = []
    for t in range(ep.length + 1):
        x, y = (int(v) for v in ep.agent_pos[t])
        cx, cy, _ = cell_center(x, y)
        fx, fy = math.cos(yaws[t]), math.sin(yaws[t])
        raw.append((cx - 2.5 * fx, cy - 2.5 * fy, 2.2, cx, cy))
    smooth = _ema(raw, alpha=0.35)
    for t, (px, py, pz, tx, ty) in enumerate(smooth):
        cam.location = (px, py, pz)
        look_at(cam, (tx, ty, AGENT_Z))
        _key_pose(cam, step_frame(t, cfg))
    set_interpolation(cam, False)
    return cam
