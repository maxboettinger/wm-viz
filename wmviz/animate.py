"""Keyframe a built scene from an Episode's arrays (spec §4 animation). No env is instantiated."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import bpy

from .scene.base import keyframe_hidden, link_only, set_interpolation
from .scene.minigrid import AGENT_Z, YAW, SceneObjects, cell_center, front_cell
from .trace.reader import Episode


@dataclass
class AnimConfig:
    frames_per_step: int = 6
    discrete: bool = False


def step_frame(t: int, cfg: AnimConfig) -> int:
    return 1 + int(t) * cfg.frames_per_step


def frame_step(f: int, cfg: AnimConfig) -> int:
    return (int(f) - 1) // cfg.frames_per_step


def unwrap_yaws(yaws: Sequence[float]) -> list[float]:
    out = [float(yaws[0])] if len(yaws) else []
    for a, b in zip(yaws[:-1], yaws[1:]):
        d = (float(b) - float(a) + math.pi) % (2 * math.pi) - math.pi   # wrap into [-pi, pi)
        if d == -math.pi:
            d = math.pi
        out.append(out[-1] + d)
    return out


def _animate_agent(sc: SceneObjects, ep: Episode, cfg: AnimConfig) -> None:
    yaws = unwrap_yaws([YAW[int(d)] for d in ep.agent_dir])
    for t in range(ep.length + 1):
        x, y = (int(v) for v in ep.agent_pos[t])
        sc.agent.location = cell_center(x, y, AGENT_Z)
        sc.agent.rotation_euler = (0.0, 0.0, yaws[t])
        f = step_frame(t, cfg)
        sc.agent.keyframe_insert("location", frame=f)
        sc.agent.keyframe_insert("rotation_euler", frame=f)
    set_interpolation(sc.agent, cfg.discrete)


def _animate_doors(sc: SceneObjects, ep: Episode, cfg: AnimConfig) -> None:
    for j, (dx, dy) in enumerate(ep.door_pos.tolist()):
        pair = sc.doors.get((int(dx), int(dy)))
        if pair is None:
            continue
        _, panel = pair
        closed = float(panel.rotation_euler.z)
        opened = closed + math.pi / 2
        flags = ep.door_open[:, j].astype(bool)
        panel.rotation_euler.z = opened if flags[0] else closed
        panel.keyframe_insert("rotation_euler", frame=step_frame(0, cfg))
        for t in range(1, ep.length + 1):
            if flags[t] == flags[t - 1]:
                continue
            panel.rotation_euler.z = opened if flags[t - 1] else closed
            panel.keyframe_insert("rotation_euler", frame=step_frame(t - 1, cfg))
            panel.rotation_euler.z = opened if flags[t] else closed
            panel.keyframe_insert("rotation_euler", frame=step_frame(t, cfg))
        set_interpolation(panel, cfg.discrete)


def _nearest_key(sc: SceneObjects, cell: tuple[int, int], on_floor: set):
    cands = [c for c in sc.keys if c in on_floor]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c[0] - cell[0]) + abs(c[1] - cell[1]))


def _make_carried(sc: SceneObjects):
    src = next(iter(sc.keys.values()))
    copy = src.copy()
    copy.animation_data_clear()      # Object.copy() shares the action+slot: our keys would overwrite the floor key's
    copy.data = src.data
    copy.name = "CarriedKey"
    link_only(copy, sc.collections["Agent"])
    for child in src.children:
        cc = child.copy()
        cc.data = child.data
        link_only(cc, sc.collections["Agent"])
        cc.parent = copy
        cc.matrix_parent_inverse = child.matrix_parent_inverse.copy()
    copy.parent = sc.agent
    copy.matrix_parent_inverse = sc.agent.matrix_world.inverted()
    copy.location = (0.0, 0.0, 0.55)
    copy.scale = (0.6, 0.6, 0.6)
    return copy


def _animate_keys(sc: SceneObjects, ep: Episode, cfg: AnimConfig) -> None:
    if not sc.keys:
        return
    on_floor = set(sc.keys)
    carried = None
    for key in sc.keys.values():
        keyframe_hidden(key, step_frame(0, cfg), hidden=False)
        key.keyframe_insert("location", frame=step_frame(0, cfg))   # else the F-curve extrapolates the first drop cell backwards
    for t in range(1, ep.length + 1):
        was, now = int(ep.carrying[t - 1]), int(ep.carrying[t])
        f = step_frame(t, cfg)
        if was == -1 and now != -1:                              # pickup
            front = front_cell(ep.agent_pos[t - 1], ep.agent_dir[t - 1])
            cell = _nearest_key(sc, front, on_floor)
            if cell is None:
                continue
            on_floor.discard(cell)
            keyframe_hidden(sc.keys[cell], f - 1, hidden=False)
            keyframe_hidden(sc.keys[cell], f, hidden=True)
            if carried is None:
                carried = _make_carried(sc)
                sc.carried = carried
                keyframe_hidden(carried, step_frame(0, cfg), hidden=True)
            keyframe_hidden(carried, f - 1, hidden=True)
            keyframe_hidden(carried, f, hidden=False)
        elif was != -1 and now == -1 and carried is not None:    # drop in front of the agent
            drop = front_cell(ep.agent_pos[t], ep.agent_dir[t])
            cell = next(iter(set(sc.keys) - on_floor), None)
            if cell is None:
                continue
            key = sc.keys[cell]
            key.location = cell_center(*drop, key.location.z)
            key.keyframe_insert("location", frame=f)
            set_interpolation(key, True, data_path="location")
            on_floor.add(cell)
            keyframe_hidden(carried, f - 1, hidden=False)
            keyframe_hidden(carried, f, hidden=True)
            keyframe_hidden(key, f - 1, hidden=True)
            keyframe_hidden(key, f, hidden=False)


def animate(sc: SceneObjects, ep: Episode, cfg: AnimConfig) -> int:
    _animate_agent(sc, ep, cfg)
    _animate_doors(sc, ep, cfg)
    _animate_keys(sc, ep, cfg)
    last = step_frame(ep.length, cfg)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, last
    scene.frame_set(1)
    return last
