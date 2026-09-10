"""Trail and heatmap overlays (spec §4). Trail = one cylinder per moved step (colour = time via
Object.color); heatmap = floor-tile Object.color keyframed from cumulative visit counts."""
from __future__ import annotations

import math

import bpy
import numpy as np

from .aggregate import cumulative_visit_counts
from .animate import AnimConfig, step_frame
from .scene.base import fcurves, keyframe_hidden, link_only, make_object_color_material, new_collection
from .scene.minigrid import SceneObjects, cell_center
from .trace.reader import Episode

_VIRIDIS = [(0.27, 0.0, 0.33), (0.13, 0.57, 0.55), (0.99, 0.9, 0.14)]
_HEAT = [(0.0, 0.0, 0.0), (0.85, 0.1, 0.05), (1.0, 0.9, 0.2)]


def _ramp(stops, u: float):
    u = min(max(float(u), 0.0), 1.0) * (len(stops) - 1)
    i = min(int(u), len(stops) - 2)
    f = u - i
    a, b = stops[i], stops[i + 1]
    return tuple(round(a[k] + (b[k] - a[k]) * f, 4) for k in range(3)) + (1.0,)


def time_color(u: float):
    return _ramp(_VIRIDIS, u)


def heat_color(u: float):
    return _ramp(_HEAT, u)


def add_trail(sc: SceneObjects, ep: Episode, cfg: AnimConfig, radius: float = 0.06, z: float = 0.12) -> list:
    col = new_collection("Trail")
    mat = make_object_color_material("Trail")
    segs = []
    T = max(ep.length, 1)
    for t in range(1, ep.length + 1):
        p0, p1 = ep.agent_pos[t - 1], ep.agent_pos[t]
        if tuple(p0) == tuple(p1):
            continue
        a = cell_center(int(p0[0]), int(p0[1]), z)
        b = cell_center(int(p1[0]), int(p1[1]), z)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z)
        # a cylinder's axis is local Z; rotation (π/2, 0, yaw) maps Z → (sin yaw, −cos yaw, 0)
        yaw = math.atan2(dx, -dy)
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=mid,
                                            rotation=(math.pi / 2, 0.0, yaw), vertices=12)
        seg = bpy.context.active_object
        seg.name = f"Trail_{t:05d}"
        seg.data.materials.append(mat)
        link_only(seg, col)
        seg.color = time_color(t / T)
        keyframe_hidden(seg, 1, True)
        f = step_frame(t, cfg)
        if f > 1:
            keyframe_hidden(seg, f - 1, True)
        keyframe_hidden(seg, f, False)
        segs.append(seg)
    sc.collections["Trail"] = col
    return segs


def add_heatmap(sc: SceneObjects, ep: Episode, cfg: AnimConfig) -> None:
    W, H = ep.layout.width, ep.layout.height
    cum = cumulative_visit_counts(ep.agent_pos, W, H)
    peak = max(int(cum[-1].max()), 1)
    for cell, tile in sc.floor.items():
        series = cum[:, cell[0], cell[1]]
        prev = -1
        for t, c in enumerate(series.tolist()):
            if c == prev:
                continue
            tile.color = heat_color(c / peak)
            tile.keyframe_insert("color", frame=step_frame(t, cfg))
            prev = c
        for fc in _color_fcurves(tile):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def _color_fcurves(obj):
    return [fc for fc in fcurves(obj) if fc.data_path == "color"]


def set_heatmap_static(sc: SceneObjects, counts: np.ndarray) -> None:
    peak = max(int(np.asarray(counts).max()), 1)
    for cell, tile in sc.floor.items():
        tile.color = heat_color(int(counts[cell[0], cell[1]]) / peak)
