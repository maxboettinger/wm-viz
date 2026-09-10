"""wmviz/scene — Blender scene primitives (needs bpy; skipped otherwise)."""
import math

import pytest

bpy = pytest.importorskip("bpy")
pytestmark = pytest.mark.slow

from wmviz.scene.base import (Style, add_lighting, asset_or, configure_render, fcurves,  # noqa: E402
                              keyframe_hidden, link_only, make_color_material, make_floor_material,
                              make_goal_material, make_object_color_material,
                              make_wall_material, new_box, new_collection, new_plane,
                              reset_scene, save_blend, set_interpolation, set_linear)


def test_reset_scene_is_empty_and_agx():
    s = reset_scene()
    assert len(bpy.data.objects) == 0
    assert s.view_settings.view_transform == "AgX" and s.render.fps == 24 and s.frame_current == 1


def test_materials_build_without_error():
    reset_scene()
    st = Style()
    for mat in (make_wall_material(st), make_floor_material(st), make_goal_material(st),
                make_object_color_material("Trail"), make_color_material("Lava", st.lava_color, 2.0)):
        assert mat.node_tree is not None and mat.node_tree.nodes
    # the floor material reads the per-object colour so heatmaps can keyframe Object.color
    floor = bpy.data.materials["Floor"]
    assert any(n.bl_idname == "ShaderNodeObjectInfo" for n in floor.node_tree.nodes)


def test_new_box_and_collections():
    reset_scene()
    col = new_collection("Walls")
    b = new_box("w", (1.5, -0.5, 0.5), (1, 1, 1), None, col)
    assert tuple(round(v, 3) for v in b.location) == (1.5, -0.5, 0.5)
    assert [c.name for c in b.users_collection] == ["Walls"]
    p = new_plane("f", (0.5, -0.5, 0.0), 1.0, None, col)
    assert p.dimensions.x == pytest.approx(1.0) and len(col.objects) == 2


def test_link_only_moves_between_collections():
    reset_scene()
    a, b = new_collection("A"), new_collection("B")
    o = new_box("o", (0, 0, 0), (1, 1, 1), None, a)
    link_only(o, b)
    assert [c.name for c in o.users_collection] == ["B"] and len(a.objects) == 0


def test_fcurves_and_interpolation():
    reset_scene()
    o = new_box("a", (0, 0, 0), (1, 1, 1), None, bpy.context.scene.collection)
    o.keyframe_insert("location", frame=1)
    o.location = (2, 0, 0)
    o.keyframe_insert("location", frame=7)
    assert len(fcurves(o)) == 3
    set_interpolation(o, discrete=True)
    assert all(kp.interpolation == "CONSTANT" for fc in fcurves(o) for kp in fc.keyframe_points)
    bpy.context.scene.frame_set(4)
    assert o.location.x == pytest.approx(0.0)          # constant: holds until frame 7
    set_interpolation(o, discrete=False)
    bpy.context.scene.frame_set(4)
    assert 0.0 < o.location.x < 2.0
    assert fcurves(o)[0].keyframe_points[0].easing == "EASE_IN_OUT"
    set_linear(o)
    bpy.context.scene.frame_set(4)
    assert o.location.x == pytest.approx(1.0)          # linear: halfway between frames 1 and 7


def test_keyframe_hidden_holds_until_next_key():
    reset_scene()
    o = new_box("h", (0, 0, 0), (1, 1, 1), None, bpy.context.scene.collection)
    keyframe_hidden(o, 1, True)
    keyframe_hidden(o, 5, False)
    bpy.context.scene.frame_set(3)
    assert o.hide_render is True and o.hide_viewport is True
    bpy.context.scene.frame_set(5)
    assert o.hide_render is False and o.hide_viewport is False


def test_lighting_render_config_and_save(tmp_path):
    s = reset_scene()
    add_lighting(Style())
    assert any(o.type == "LIGHT" for o in bpy.data.objects) and s.world is not None
    configure_render("eevee", 32, (320, 180), 12)
    assert s.render.engine == "BLENDER_EEVEE" and (s.render.resolution_x, s.render.fps) == (320, 12)
    assert s.render.image_settings.color_mode == "RGB"
    configure_render("cycles", 8, (64, 64), 24)
    assert s.render.engine == "CYCLES" and s.cycles.samples == 8
    with pytest.raises(ValueError):
        configure_render("workbench", 1, (8, 8), 1)
    out = save_blend(tmp_path / "s.blend")
    assert out.exists() and out.stat().st_size > 0


def test_asset_or_falls_back_and_loads_from_library(tmp_path):
    # build a library with one object called "Agent", save it, then load it into a fresh scene
    reset_scene()
    new_box("Agent", (0, 0, 0), (0.3, 0.3, 0.3), None, bpy.context.scene.collection)
    lib = save_blend(tmp_path / "lib.blend")
    reset_scene()
    made = []
    a = asset_or(None, "Agent", lambda: made.append(1) or new_box("Agent", (0, 0, 0), (1, 1, 1), None, bpy.context.scene.collection))
    assert made == [1] and a.dimensions.x == pytest.approx(1.0)
    reset_scene()
    b = asset_or(lib, "Agent", lambda: pytest.fail("fallback must not run when the asset exists"))
    assert b.name.startswith("Agent") and b.dimensions.x == pytest.approx(0.3)
    assert b.name in bpy.context.scene.objects
    c = asset_or(lib, "Missing", lambda: new_box("Missing", (0, 0, 0), (1, 1, 1), None, bpy.context.scene.collection))
    assert c.dimensions.x == pytest.approx(1.0)


# ── MiniGrid scene builder ────────────────────────────────────────────────────

from pathlib import Path  # noqa: E402

from wmviz.scene.minigrid import (SceneObjects, YAW, build_scene, cell_center, door_axis,  # noqa: E402
                                  front_cell)
from wmviz.trace import Episode, Index  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _first_episode(run: str) -> Episode:
    idx = Index.load(FIX / run)
    return Episode.load(idx.path_of(idx.rows[0]))


def test_cell_center_and_yaw():
    assert cell_center(0, 0) == (0.5, -0.5, 0.0)
    assert cell_center(3, 2, 0.25) == (3.5, -2.5, 0.25)
    assert YAW[0] == 0 and YAW[1] == pytest.approx(-math.pi / 2) and YAW[3] == pytest.approx(math.pi / 2)
    assert front_cell((2, 3), 0) == (3, 3) and front_cell((2, 3), 1) == (2, 4) and front_cell((2, 3), 3) == (2, 2)


def test_door_axis_on_doorkey_fixture():
    ep = _first_episode("doorkey6x6")
    (cell, _), = ep.layout.doors.items()
    x, y = cell
    lay = ep.layout
    expected = "x" if ((x - 1, y) in lay.walls or (x + 1, y) in lay.walls) else "y"
    assert door_axis(lay, cell) == expected


def test_build_scene_object_counts_doorkey():
    reset_scene()
    ep = _first_episode("doorkey6x6")
    sc = build_scene(ep.layout)
    lay = ep.layout
    assert isinstance(sc, SceneObjects)
    assert len(sc.walls) == len(lay.walls)
    assert set(sc.doors) == set(lay.doors) and set(sc.keys) == set(lay.keys)
    assert len(sc.goals) == len(lay.goals) == 1
    n_floor = lay.width * lay.height - len(lay.walls) - len(lay.doors)
    assert len(sc.floor) == n_floor and all(tuple(o.color[:3]) == (0.0, 0.0, 0.0) for o in sc.floor.values())
    assert sc.agent.name == "Agent" and sc.agent.children and sc.carried is None
    assert set(sc.collections) == {"Floor", "Walls", "Doors", "Items", "Agent"}
    assert any(o.type == "LIGHT" for o in bpy.data.objects) and bpy.context.scene.world is not None


def test_door_panel_origin_is_hinge_and_key_has_colour():
    reset_scene()
    ep = _first_episode("doorkey6x6")
    sc = build_scene(ep.layout)
    (cell, (frame, panel)), = sc.doors.items()
    cx, cy, _ = cell_center(*cell)
    # hinge sits on the cell edge, not the cell centre
    assert abs(panel.location.x - cx) == pytest.approx(0.4, abs=0.11) or abs(panel.location.y - cy) == pytest.approx(0.4, abs=0.11)
    assert abs(panel.rotation_euler.z) in (0.0, pytest.approx(math.pi / 2))   # ±π/2: closed yaw follows YAW[1]
    # the panel mesh extends only to one side of its origin (so rotating swings it open)
    xs = [v.co.x for v in panel.data.vertices]
    assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(1.0)
    (kcell, key), = sc.keys.items()
    colour = ep.layout.keys[kcell]
    assert key.data.materials[0].name.startswith(f"Key-{colour}")


def test_build_scene_multiroom_has_many_doors():
    reset_scene()
    ep = _first_episode("multiroom-n4s5")
    sc = build_scene(ep.layout)
    assert len(sc.doors) == len(ep.layout.doors) >= 5
    assert len(sc.walls) == len(ep.layout.walls)
