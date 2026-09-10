"""Visit-count aggregation and episode grouping for figures (spec §6). No bpy."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .trace.reader import Episode, IndexRow


class LayoutMismatch(Exception):
    pass


def visit_counts(pos: np.ndarray, W: int, H: int) -> np.ndarray:
    counts = np.zeros((W, H), dtype=np.int64)
    p = np.asarray(pos, dtype=np.int64).reshape(-1, 2)
    np.add.at(counts, (p[:, 0], p[:, 1]), 1)
    return counts


def cumulative_visit_counts(pos: np.ndarray, W: int, H: int) -> np.ndarray:
    p = np.asarray(pos, dtype=np.int64).reshape(-1, 2)
    out = np.zeros((len(p), W, H), dtype=np.int64)
    cur = np.zeros((W, H), dtype=np.int64)
    for t, (x, y) in enumerate(p):
        cur[x, y] += 1
        out[t] = cur
    return out


def same_layout(rows: Sequence[IndexRow], force: bool = False) -> str:
    if not rows:
        raise ValueError("no episodes selected")
    hashes = sorted({r.layout_hash for r in rows})
    if len(hashes) > 1 and not force:
        raise LayoutMismatch(f"{len(rows)} episodes span {len(hashes)} layouts ({', '.join(hashes)}); "
                             "aggregating across layouts is refused — narrow with --layout <hash|seed:N> or pass --force")
    return rows[0].layout_hash


def nearest_to_milestones(rows: Sequence[IndexRow], milestones: Sequence[int]) -> list[IndexRow]:
    if not rows:
        raise ValueError("no episodes to pick milestones from")
    return [min(rows, key=lambda r: (abs(r.start_step - int(m)), r.episode_id)) for m in milestones]


def accumulate(episodes: Iterable[Episode], W: int, H: int) -> np.ndarray:
    total = np.zeros((W, H), dtype=np.int64)
    for ep in episodes:
        total += visit_counts(ep.agent_pos, W, H)
    return total
