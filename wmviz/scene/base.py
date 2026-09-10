"""Blender primitives shared by the MiniGrid scene builder, overlays and renderer.

Carried over from blender-dream's scene.py: world-space brick walls, noise floor,
warm sun + flat sky, AgX. bpy 5.0 gotchas: Action.fcurves is gone (use fcurves()),
noise/ramp sockets are called "Factor".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import bpy

RGBA = tuple[float, float, float, float]

COLOR_RGB = {"red": (0.85, 0.2, 0.2), "green": (0.2, 0.7, 0.3), "blue": (0.25, 0.4, 0.9),
             "purple": (0.6, 0.3, 0.8), "yellow": (0.95, 0.8, 0.2), "grey": (0.5, 0.5, 0.5)}

ENGINES = {"eevee": "BLENDER_EEVEE", "cycles": "CYCLES"}


@dataclass
class Style:
    wall_color1: RGBA = (0.55, 0.35, 0.25, 1)
    wall_color2: RGBA = (0.45, 0.25, 0.18, 1)
    mortar: RGBA = (0.75, 0.72, 0.68, 1)
    brick_scale: float = 1.2               # bricks are ~0.2 units tall at cell size 1
    floor_dark: RGBA = (0.15, 0.15, 0.17, 1)
    floor_light: RGBA = (0.35, 0.34, 0.32, 1)
    goal_color: RGBA = (1.0, 0.45, 0.1, 1)
    goal_strength: float = 8.0
    lava_color: RGBA = (0.95, 0.4, 0.1, 1)
    agent_color: RGBA = (0.9, 0.15, 0.15, 1)
    wall_height: float = 1.0
    door_thickness: float = 0.08
    sun_energy: float = 3.0
    sun_rotation_deg: tuple[float, float, float] = (40.0, 0.0, 30.0)
    sky: RGBA = (0.35, 0.55, 0.9, 1)


# ── Scene / animation plumbing ────────────────────────────────────────────────

def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.view_settings.view_transform = "AgX"
    s.render.fps = 24
    s.frame_start = 1
    s.frame_set(1)
    return s


def fcurves(obj) -> list:
    """All F-curves animating `obj` (bpy 5 slotted actions; Action.fcurves is gone)."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    slot = ad.action_slot or act.slots[0]
    return list(act.layers[0].strips[0].channelbag(slot).fcurves)


def set_interpolation(obj, discrete: bool, data_path: str | None = None) -> None:
    for fc in fcurves(obj):
        if data_path is not None and fc.data_path != data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if discrete else "BEZIER"
            kp.easing = "EASE_IN_OUT"


def set_linear(obj) -> None:
    for fc in fcurves(obj):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def keyframe_hidden(obj, frame: int, hidden: bool) -> None:
    """Keyframe render+viewport visibility of `obj` and all its descendants at `frame`
    (booleans hold until the next key; Blender does not propagate hide_* through parenting)."""
    for o in (obj, *obj.children_recursive):
        o.hide_render = hidden
        o.hide_viewport = hidden
        o.keyframe_insert("hide_render", frame=frame)
        o.keyframe_insert("hide_viewport", frame=frame)


def new_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def link_only(obj, collection) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)


def _finish(obj, name, material, collection):
    obj.name = name
    if material is not None:
        obj.data.materials.append(material)
    if collection is not None:
        link_only(obj, collection)
    return obj


def new_box(name, center, dims, material, collection):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(center))
    obj = bpy.context.active_object
    obj.scale = tuple(dims)
    return _finish(obj, name, material, collection)


def new_plane(name, center, size, material, collection):
    bpy.ops.mesh.primitive_plane_add(size=size, location=tuple(center))
    return _finish(bpy.context.active_object, name, material, collection)


# ── Materials ─────────────────────────────────────────────────────────────────

def _new_material(name: str):
    mat = bpy.data.materials.new(name)  # bpy 5 creates the node tree (use_nodes is deprecated)
    return mat, mat.node_tree


def make_wall_material(style: Style):
    mat, tree = _new_material("Wall")
    bsdf = tree.nodes["Principled BSDF"]
    # World-space position drives the brick pattern so scale is independent of object scale.
    geo = tree.nodes.new("ShaderNodeNewGeometry")
    sep = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    along = tree.nodes.new("ShaderNodeMath")
    along.operation = "ADD"
    tree.links.new(sep.outputs["X"], along.inputs[0])
    tree.links.new(sep.outputs["Y"], along.inputs[1])
    vec = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(along.outputs["Value"], vec.inputs["X"])
    tree.links.new(sep.outputs["Z"], vec.inputs["Y"])
    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = style.wall_color1
    brick.inputs["Color2"].default_value = style.wall_color2
    brick.inputs["Mortar"].default_value = style.mortar
    brick.inputs["Scale"].default_value = style.brick_scale
    brick.inputs["Mortar Size"].default_value = 0.02
    tree.links.new(vec.outputs["Vector"], brick.inputs["Vector"])
    tree.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 12.0
    tree.links.new(geo.outputs["Position"], noise.inputs["Vector"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    tree.links.new(noise.outputs["Factor"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def make_floor_material(style: Style):
    """Noise/ramp base; emission = Object Info Color so a heatmap can keyframe obj.color.
    Tiles are created with color (0,0,0,1), i.e. no emission until the overlay sets it."""
    mat, tree = _new_material("Floor")
    bsdf = tree.nodes["Principled BSDF"]
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = style.floor_dark
    ramp.color_ramp.elements[1].color = style.floor_light
    tree.links.new(noise.outputs["Factor"], ramp.inputs["Factor"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    info = tree.nodes.new("ShaderNodeObjectInfo")
    tree.links.new(info.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.5
    return mat


def make_goal_material(style: Style):
    mat, tree = _new_material("Goal")
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = style.goal_color
    emit.inputs["Strength"].default_value = style.goal_strength
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_color_material(name: str, rgba: RGBA, emission: float = 0.0):
    """Flat-colour material, idempotent by name: a repeat call returns the existing one unchanged."""
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    mat, tree = _new_material(name)
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.5
    if emission > 0:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def make_object_color_material(name: str):
    """Base colour and emission both read Object Info → Color (trail segments, markers)."""
    mat, tree = _new_material(name)
    bsdf = tree.nodes["Principled BSDF"]
    info = tree.nodes.new("ShaderNodeObjectInfo")
    tree.links.new(info.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(info.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat


# ── Lighting, render settings, files ─────────────────────────────────────────

def add_lighting(style: Style) -> None:
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = style.sun_energy
    sun.data.color = (1.0, 0.9, 0.75)
    sun.rotation_euler = tuple(math.radians(d) for d in style.sun_rotation_deg)
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = style.sky
    bg.inputs["Strength"].default_value = 1.0


def configure_render(engine: str, samples: int, res: tuple[int, int], fps: int) -> None:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, got {engine!r}")
    s = bpy.context.scene
    s.render.engine = ENGINES[engine]
    if engine == "cycles":
        s.cycles.samples = samples
        s.cycles.use_denoising = True
    else:
        s.eevee.taa_render_samples = samples
    s.render.resolution_x, s.render.resolution_y = int(res[0]), int(res[1])
    s.render.resolution_percentage = 100
    s.render.fps = fps
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    s.render.film_transparent = False


def save_blend(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
    return path


def asset_or(library: Path | None, name: str, fallback: Callable[[], object]):
    """Object `name` appended from `library` (.blend) if present there, else `fallback()`.
    The appended object is linked into the scene collection and renamed uniquely by Blender."""
    if library is not None and Path(library).exists():
        with bpy.data.libraries.load(str(library), link=False) as (src, dst):
            if name in src.objects:
                dst.objects = [name]
            else:
                dst.objects = []
        if dst.objects:
            obj = dst.objects[0]
            bpy.context.scene.collection.objects.link(obj)
            bpy.context.view_layer.update()  # evaluate so obj.dimensions etc. are current
            return obj
    return fallback()
