"""Read wm episode traces (format_version 1).

Contract: wm/docs/superpowers/specs/2026-09-09-run-visualization-design.md §1.
logs/<run>/trace/index.csv + logs/<run>/trace/episodes/ep_NNNNNN.npz.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

# minigrid.core.constants (copied: wmviz does not depend on minigrid)
IDX_TO_OBJECT = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key",
                 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
IDX_TO_COLOR = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}

_REF = re.compile(r"^(?P<run>[^/]+)/(?:ep)?(?P<id>\d+)$")


def parse_ref(ref: str) -> tuple[str, int]:
    m = _REF.match(ref.strip())
    if not m:
        raise ValueError(f"episode reference must look like <run>/ep000123 or <run>/123, got {ref!r}")
    return m.group("run"), int(m.group("id"))


def _opt_int(s: str) -> int | None:
    return int(s) if s not in ("", None) else None


def _opt_float(s: str) -> float | None:
    return float(s) if s not in ("", None) else None


@dataclass(frozen=True)
class IndexRow:
    episode_id: int
    phase: str
    actor: str
    env_slot: int
    seed: int | None
    start_step: int
    end_step: int
    length: int
    ret: float
    success: bool
    unique_cells: int
    coverage_pct: float | None
    first_key_step: int | None
    first_door_step: int | None
    first_goal_step: int | None
    rooms_visited: int | None
    layout_hash: str
    file: str

    @classmethod
    def from_csv(cls, d: dict) -> "IndexRow":
        return cls(
            episode_id=int(d["episode_id"]), phase=d["phase"], actor=d["actor"],
            env_slot=int(d["env_slot"]), seed=_opt_int(d["seed"]),
            start_step=int(d["start_step"]), end_step=int(d["end_step"]),
            length=int(d["length"]), ret=float(d["return"]),
            success=d["success"] in ("1", "True", "true"),
            unique_cells=int(d["unique_cells"]), coverage_pct=_opt_float(d["coverage_pct"]),
            first_key_step=_opt_int(d["first_key_step"]), first_door_step=_opt_int(d["first_door_step"]),
            first_goal_step=_opt_int(d["first_goal_step"]), rooms_visited=_opt_int(d["rooms_visited"]),
            layout_hash=d["layout_hash"], file=d["file"],
        )

    def ref(self, run_name: str) -> str:
        return f"{run_name}/ep{self.episode_id:06d}"


@dataclass
class Index:
    run_dir: Path
    rows: list[IndexRow]

    @property
    def run_name(self) -> str:
        return self.run_dir.name

    @classmethod
    def load(cls, run_dir: Path | str) -> "Index":
        run_dir = Path(run_dir)
        path = run_dir / "trace" / "index.csv"
        if not path.exists():
            raise FileNotFoundError(f"no trace/index.csv under {run_dir} — was the run recorded with --trace auto?")
        with open(path, newline="") as fh:
            rows = [IndexRow.from_csv(d) for d in csv.DictReader(fh)]
        return cls(run_dir, rows)

    def by_id(self, episode_id: int) -> IndexRow:
        for r in self.rows:
            if r.episode_id == episode_id:
                return r
        raise KeyError(f"{self.run_name} has no episode {episode_id} (ids 0..{len(self.rows) - 1})")

    def path_of(self, row: IndexRow) -> Path:
        return self.run_dir / "trace" / row.file


def find_runs(logs_dir: Path | str) -> list[Path]:
    logs_dir = Path(logs_dir)
    return sorted(p for p in logs_dir.iterdir() if (p / "trace" / "index.csv").exists())


@dataclass
class Layout:
    grid: np.ndarray                      # (W, H, 3) uint8: object, colour, state
    rooms: list[tuple[int, int, int, int]] = field(default_factory=list)
    walls: set[tuple[int, int]] = field(default_factory=set)
    doors: dict[tuple[int, int], str] = field(default_factory=dict)
    keys: dict[tuple[int, int], str] = field(default_factory=dict)
    goals: set[tuple[int, int]] = field(default_factory=set)
    lava: set[tuple[int, int]] = field(default_factory=set)

    @property
    def width(self) -> int:
        return int(self.grid.shape[0])

    @property
    def height(self) -> int:
        return int(self.grid.shape[1])

    @classmethod
    def from_grid(cls, grid: np.ndarray, rooms: np.ndarray | None = None) -> "Layout":
        lay = cls(grid=np.asarray(grid, dtype=np.uint8))
        W, H = lay.width, lay.height
        for x in range(W):
            for y in range(H):
                obj, col = int(grid[x, y, 0]), int(grid[x, y, 1])
                name = IDX_TO_OBJECT.get(obj, "unseen")
                if name == "wall":
                    lay.walls.add((x, y))
                elif name == "door":
                    lay.doors[(x, y)] = IDX_TO_COLOR.get(col, "grey")
                elif name == "key":
                    lay.keys[(x, y)] = IDX_TO_COLOR.get(col, "grey")
                elif name == "goal":
                    lay.goals.add((x, y))
                elif name == "lava":
                    lay.lava.add((x, y))
        if rooms is not None:
            lay.rooms = [tuple(int(v) for v in r) for r in np.asarray(rooms).reshape(-1, 4)]
        return lay


@dataclass
class Episode:
    path: Path
    meta: dict
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    agent_pos: np.ndarray
    agent_dir: np.ndarray
    carrying: np.ndarray
    door_pos: np.ndarray
    door_open: np.ndarray
    layout: Layout
    obs: np.ndarray | None = None
    dream: dict | None = None            # {"dream_start", "dream_frames", "recon_frames"?}

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def ret(self) -> float:
        return float(self.rewards.sum())

    @classmethod
    def load(cls, path: Path | str) -> "Episode":
        path = Path(path)
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            v = meta.get("format_version")
            if v != FORMAT_VERSION:
                raise ValueError(f"trace format {v} not supported by wmviz (expects {FORMAT_VERSION}): {path}")
            if meta.get("env_family") != "minigrid":
                raise ValueError(f"env_family {meta.get('env_family')!r} not supported yet: {path}")
            files = set(z.files)
            layout = Layout.from_grid(z["layout"], z["rooms"] if "rooms" in files else None)
            dream = None
            if "dream_start" in files:
                dream = {"dream_start": int(z["dream_start"]), "dream_frames": z["dream_frames"]}
                if "recon_frames" in files:
                    dream["recon_frames"] = z["recon_frames"]
            return cls(
                path=path, meta=meta,
                actions=z["actions"], rewards=z["rewards"],
                terminated=bool(z["terminated"]), truncated=bool(z["truncated"]),
                agent_pos=z["agent_pos"].astype(np.int64), agent_dir=z["agent_dir"].astype(np.int64),
                carrying=z["carrying"].astype(np.int64), door_pos=z["door_pos"].astype(np.int64),
                door_open=z["door_open"].astype(bool), layout=layout,
                obs=z["obs"] if "obs" in files else None, dream=dream,
            )
