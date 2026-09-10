"""Build a Blender scene from a MiniGrid Layout (spec §4 scene builder).

One unit per cell; cell (x, y) → (x+0.5, -(y+0.5)); agent_dir → yaw about Z.
Object types are looked up in an optional asset library first (`Agent`, `Key`,
`DoorPanel`, `DoorFrame`, `Wall`, `Floor`, `Goal`); otherwise procedural primitives.
Either way every object is linked into its type's collection (`Floor`, `Walls`,
`Doors`, `Items`, `Agent`) so later stages can toggle whole collections.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import bpy
from mathutils import Matrix

from ..grid import COLOR_RGB, DIR_VEC
from ..trace.reader import Layout
from .base import (Style, add_lighting, asset_or, link_only, make_color_material,
                   make_floor_material, make_goal_material, make_wall_material, new_box,
                   new_collection, new_plane)

YAW = {0: 0.0, 1: -math.pi / 2, 2: math.pi, 3: math.pi / 2}
AGENT_Z = 0.25


def cell_center(x: int, y: int, z: float = 0.0) -> tuple[float, float, float]:
    return (float(x) + 0.5, -(float(y) + 0.5), float(z))


def front_cell(pos, d: int) -> tuple[int, int]:
    dx, dy = DIR_VEC[int(d)]
    return int(pos[0]) + dx, int(pos[1]) + dy


def door_axis(layout: Layout, cell: tuple[int, int]) -> str:
    x, y = cell
    return "x" if ((x - 1, y) in layout.walls or (x + 1, y) in layout.walls) else "y"


@dataclass
class SceneObjects:
    floor: dict[tuple[int, int], object] = field(default_factory=dict)
    walls: list = field(default_factory=list)
    doors: dict[tuple[int, int], tuple[object, object]] = field(default_factory=dict)
    keys: dict[tuple[int, int], object] = field(default_factory=dict)
    goals: list = field(default_factory=list)
    agent: object = None
    carried: object | None = None
    style: Style = field(default_factory=Style)
    collections: dict[str, object] = field(default_factory=dict)


def _rgba(colour: str):
    r, g, b = COLOR_RGB.get(colour, COLOR_RGB["grey"])
    return (r, g, b, 1.0)


def _asset(assets: Path | None, name: str, col, fallback: Callable[[], object]) -> tuple[object, bool]:
    """`asset_or` + link into `col`. Returns (object, from_library) so a caller can tell whether
    the procedural fallback ran (whose origin convention it controls) or a library object came back."""
    built = []
    obj = asset_or(assets, name, lambda: built.append(fallback()) or built[0])
    link_only(obj, col)
    return obj, not built


def _make_agent(style: Style, col):
    body = new_box("Agent", (0, 0, AGENT_Z), (0.5, 0.5, 0.5), make_color_material("Agent", style.agent_color, emission=0.3), col)
    nose = new_box("AgentNose", (0.3, 0, AGENT_Z), (0.2, 0.2, 0.2), make_color_material("AgentNose", (1, 1, 1, 1)), col)
    nose.parent = body
    nose.matrix_parent_inverse = body.matrix_world.inverted()
    return body


def _make_key(name: str, colour: str, center, col):
    mat = make_color_material(f"Key-{colour}", _rgba(colour), emission=0.4)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.45, location=(center[0], center[1], 0.25))
    shaft = bpy.context.active_object
    shaft.name = name
    shaft.data.materials.append(mat)
    link_only(shaft, col)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.12, minor_radius=0.04, location=(center[0], center[1], 0.5))
    bow = bpy.context.active_object
    bow.name = name + "Bow"
    bow.data.materials.append(mat)
    link_only(bow, col)
    bow.parent = shaft
    bow.matrix_parent_inverse = shaft.matrix_world.inverted()
    return shaft


def _make_door_frame(cell, colour: str, axis: str, style: Style, col):
    """Two posts + lintel parented to one empty at the cell centre."""
    x, y = cell
    H = style.wall_height
    mat = make_color_material(f"DoorFrame-{colour}", (0.3, 0.2, 0.12, 1))
    cx, cy, _ = cell_center(x, y)
    bpy.ops.object.empty_add(location=(cx, cy, 0))
    frame = bpy.context.active_object
    frame.name = f"DoorFrame_{x}_{y}"
    link_only(frame, col)
    if axis == "x":      # wall runs along x, passage is along y: posts at x edges
        posts = [(x + 0.05, cy), (x + 0.95, cy)]
        post_dims, lintel_dims = (0.1, 0.15, H), (1.0, 0.15, 0.1)
    else:                # wall runs along y: posts at y edges
        posts = [(cx, -(y + 0.05)), (cx, -(y + 0.95))]
        post_dims, lintel_dims = (0.15, 0.1, H), (0.15, 1.0, 0.1)
    parts = [new_box(f"DoorPost_{x}_{y}_{i}", (px, py, H / 2), post_dims, mat, col) for i, (px, py) in enumerate(posts)]
    parts.append(new_box(f"DoorLintel_{x}_{y}", (cx, cy, H - 0.05), lintel_dims, mat, col))
    for p in parts:
        p.parent = frame
        p.matrix_parent_inverse = frame.matrix_world.inverted()
    return frame


def _make_door_panel(cell, colour: str, style: Style, col):
    """Box whose mesh spans x∈[0,1], z∈[0,1] in local space, so its origin is the hinge edge."""
    x, y = cell
    panel = new_box(f"DoorPanel_{x}_{y}", (0, 0, 0), (0.8, style.door_thickness, style.wall_height - 0.1),
                    make_color_material(f"DoorPanel-{colour}", _rgba(colour)), col)
    panel.data.transform(Matrix.Translation((0.5, 0.0, 0.5)))
    return panel


def _make_door(cell, colour: str, axis: str, style: Style, col, assets: Path | None):
    """Frame at the cell centre; panel origin at the hinge post with `rotation_euler.z` = closed
    yaw (opening adds +π/2). Library and procedural objects are placed by the same hinge/yaw."""
    x, y = cell
    cx, cy, _ = cell_center(x, y)
    if axis == "x":      # wall runs along x: hinge on the left post
        hinge, yaw = (x + 0.1, cy, 0.05), 0.0
    else:                # wall runs along y: hinge on the top post, panel points along DIR_VEC[1]
        hinge, yaw = (cx, -(y + 0.1), 0.05), -math.pi / 2
    frame, _ = _asset(assets, "DoorFrame", col, lambda: _make_door_frame(cell, colour, axis, style, col))
    frame.location = (cx, cy, 0.0)
    panel, _ = _asset(assets, "DoorPanel", col, lambda: _make_door_panel(cell, colour, style, col))
    panel.location = hinge
    panel.rotation_euler = (0.0, 0.0, yaw)
    return frame, panel


def build_scene(layout: Layout, style: Style | None = None, assets: Path | None = None) -> SceneObjects:
    style = style or Style()
    sc = SceneObjects(style=style)
    for name in ("Floor", "Walls", "Doors", "Items", "Agent"):
        sc.collections[name] = new_collection(name)
    wall_mat, floor_mat, goal_mat = make_wall_material(style), make_floor_material(style), make_goal_material(style)
    lava_mat = make_color_material("Lava", style.lava_color, emission=4.0) if layout.lava else None
    W, H = layout.width, layout.height
    for x in range(W):
        for y in range(H):
            cell = (x, y)
            if cell in layout.walls:
                wall, _ = _asset(assets, "Wall", sc.collections["Walls"], lambda c=cell: new_box(
                    f"Wall_{c[0]}_{c[1]}", cell_center(*c, style.wall_height / 2), (1, 1, style.wall_height), wall_mat, sc.collections["Walls"]))
                wall.location = cell_center(x, y, style.wall_height / 2)
                sc.walls.append(wall)
                continue
            if cell in layout.doors:
                continue          # door cells get a frame + panel, no floor tile
            tile, _ = _asset(assets, "Floor", sc.collections["Floor"], lambda c=cell: new_plane(
                f"Floor_{c[0]}_{c[1]}", cell_center(*c), 1.0, lava_mat if c in layout.lava else floor_mat, sc.collections["Floor"]))
            tile.location = cell_center(x, y)
            tile.color = (0.0, 0.0, 0.0, 1.0)
            sc.floor[cell] = tile
    for cell, colour in layout.doors.items():
        sc.doors[cell] = _make_door(cell, colour, door_axis(layout, cell), style, sc.collections["Doors"], assets)
    for cell, colour in layout.keys.items():
        key, from_library = _asset(assets, "Key", sc.collections["Items"],
                                   lambda c=cell, k=colour: _make_key(f"Key_{c[0]}_{c[1]}", k, cell_center(*c), sc.collections["Items"]))
        if from_library:          # the procedural key already stands on the floor of its cell
            key.location = cell_center(*cell)
        sc.keys[cell] = key
    for cell in layout.goals:
        g, _ = _asset(assets, "Goal", sc.collections["Items"],
                      lambda c=cell: new_plane(f"Goal_{c[0]}_{c[1]}", cell_center(*c, 0.01), 0.9, goal_mat, sc.collections["Items"]))
        g.location = cell_center(*cell, 0.01)
        sc.goals.append(g)
    sc.agent, _ = _asset(assets, "Agent", sc.collections["Agent"], lambda: _make_agent(style, sc.collections["Agent"]))
    add_lighting(style)
    return sc
