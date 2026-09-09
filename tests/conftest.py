"""Synthetic trace writer mirroring the wm trace format (spec §1, format_version 1)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

COLUMNS = ["episode_id", "phase", "actor", "env_slot", "seed", "start_step", "end_step",
           "length", "return", "success", "unique_cells", "coverage_pct",
           "first_key_step", "first_door_step", "first_goal_step", "rooms_visited",
           "layout_hash", "file"]


def doorkey_layout():
    """7x7: walls around, key (1,3), locked yellow door (3,3) in a wall column x=3, goal (5,5)."""
    g = np.zeros((7, 7, 3), dtype=np.uint8)
    g[:, :, 0] = 1
    g[0, :, 0] = g[6, :, 0] = g[:, 0, 0] = g[:, 6, 0] = 2
    g[3, :, 0] = 2
    g[3, 3, 0], g[3, 3, 1], g[3, 3, 2] = 4, 4, 2
    g[1, 3, 0], g[1, 3, 1] = 5, 4
    g[5, 5, 0], g[5, 5, 1] = 8, 1
    return g


def make_episode(layout=None, T=6, phase="train_task", actor="task", seed=None,
                 start_step=0, end_step=None, success=False, with_obs=False, first_door=None):
    g = doorkey_layout() if layout is None else layout
    pos = np.array([[1, 1]] * (T + 1), dtype=np.int16)
    for t in range(1, T + 1):
        pos[t] = [1 + min(t, 4), 1 + max(0, t - 4)]
    dirs = np.zeros(T + 1, dtype=np.int8)
    carrying = np.full(T + 1, -1, dtype=np.int8)
    door_open = np.zeros((T + 1, 1), dtype=np.uint8)
    if first_door is not None:
        carrying[first_door:] = 5
        door_open[first_door:, 0] = 1
    rewards = np.zeros(T, dtype=np.float32)
    if success:
        rewards[-1] = 0.9
    arrays = dict(actions=np.full(T, 2, dtype=np.int16), rewards=rewards,
                  terminated=np.array(success), truncated=np.array(not success),
                  layout=g, agent_pos=pos, agent_dir=dirs, carrying=carrying,
                  door_pos=np.array([[3, 3]], dtype=np.int16), door_open=door_open)
    if with_obs:
        arrays["obs"] = np.zeros((T + 1, 8, 8, 3), dtype=np.uint8)
    meta = dict(format_version=1, env_family="minigrid", env_id="MiniGrid-DoorKey-6x6-v0",
                run_name="run", phase=phase, actor=actor, seed=seed, exploration="none",
                obs_mode="pixel", start_step=start_step, end_step=end_step or start_step + T,
                env_slot=0)
    return arrays, meta


class TraceBuilder:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "trace" / "episodes").mkdir(parents=True)
        self.index = self.run_dir / "trace" / "index.csv"
        with open(self.index, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()
        self.n = 0

    def add(self, **kw):
        arrays, meta = make_episode(**kw)
        ep = self.n; self.n += 1
        meta["episode_id"] = ep
        rel = f"episodes/ep_{ep:06d}.npz"
        with open(self.run_dir / "trace" / rel, "wb") as fh:
            np.savez_compressed(fh, meta=json.dumps(meta), **arrays)
        pos = arrays["agent_pos"]
        cells = {(int(x), int(y)) for x, y in pos}
        fk = next((i for i, c in enumerate(arrays["carrying"]) if c == 5), None)
        fd = next((i for i, d in enumerate(arrays["door_open"][:, 0]) if d), None)
        goal = arrays["layout"][pos[:, 0], pos[:, 1], 0] == 8
        fg = int(np.flatnonzero(goal)[0]) if goal.any() else None
        row = dict(episode_id=ep, phase=meta["phase"], actor=meta["actor"], env_slot=0,
                   seed="" if meta["seed"] is None else meta["seed"],
                   start_step=meta["start_step"], end_step=meta["end_step"],
                   length=len(arrays["actions"]), **{"return": float(arrays["rewards"].sum())},
                   success=int(bool(arrays["terminated"]) and arrays["rewards"].sum() > 0),
                   unique_cells=len(cells), coverage_pct=f"{len(cells) / 25:.6g}",
                   first_key_step="" if fk is None else fk, first_door_step="" if fd is None else fd,
                   first_goal_step="" if fg is None else fg, rooms_visited="",
                   layout_hash=hashlib.sha1(arrays["layout"].tobytes()).hexdigest()[:12], file=rel)
        with open(self.index, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
        return ep


@pytest.fixture
def run_dir(tmp_path):
    """A synthetic run with 6 episodes covering the selector cases."""
    b = TraceBuilder(tmp_path / "logs" / "p2e-doorkey6x6-s0")
    b.add(phase="train_explorer", actor="explorer", start_step=0, T=6)
    b.add(phase="train_task", actor="task", start_step=24, T=6, first_door=3)
    b.add(phase="eval", actor="task", seed=1000, start_step=100, end_step=100, T=6)
    b.add(phase="eval", actor="task", seed=1001, start_step=100, end_step=100, T=8, success=True, first_door=2)
    b.add(phase="coverage_eval", actor="explorer", seed=2000, start_step=200, end_step=200, T=5)
    b.add(phase="eval", actor="task", seed=1000, start_step=300, end_step=300, T=6, with_obs=True)
    return b.run_dir


@pytest.fixture
def logs_dir(run_dir):
    return run_dir.parent
