"""Filter / sort / pick over Index rows. Pure functions; no I/O."""
from __future__ import annotations

from dataclasses import dataclass

from .reader import IndexRow

SORT_KEYS = {
    "return": lambda r: r.ret,
    "unique_cells": lambda r: r.unique_cells,
    "length": lambda r: r.length,
    "start_step": lambda r: r.start_step,
    "episode_id": lambda r: r.episode_id,
    "coverage_pct": lambda r: -1.0 if r.coverage_pct is None else r.coverage_pct,
}
SELECTORS = ("best-return", "most-cells", "first-success", "first-door", "first-key", "at-step", "latest")


class NoMatch(Exception):
    """No episode satisfies the selector, or the filters left no rows to choose from.
    This exception's own message does not name candidates; the CLI (pick_cmd) appends
    nearest-candidate hints to str(e) before printing it."""


@dataclass
class Filters:
    phase: str | None = None
    actor: str | None = None
    after_step: int | None = None
    before_step: int | None = None
    success: bool | None = None
    min_cells: int | None = None
    layout: str | None = None          # layout-hash prefix, or "seed:<n>"


def apply_filters(rows: list[IndexRow], f: Filters) -> list[IndexRow]:
    seed_filter: int | None = None
    if f.layout is not None and f.layout.startswith("seed:"):
        raw = f.layout[5:]
        try:
            seed_filter = int(raw)
        except ValueError:
            raise ValueError(f"--layout seed:<n> needs an integer, got {raw!r}") from None
    out = []
    for r in rows:
        if f.phase is not None and r.phase != f.phase:
            continue
        if f.actor is not None and r.actor != f.actor:
            continue
        if f.after_step is not None and r.start_step < f.after_step:
            continue
        if f.before_step is not None and r.start_step > f.before_step:
            continue
        if f.success is not None and r.success != f.success:
            continue
        if f.min_cells is not None and r.unique_cells < f.min_cells:
            continue
        if f.layout is not None:
            if f.layout.startswith("seed:"):
                if r.seed != seed_filter:
                    continue
            elif not r.layout_hash.startswith(f.layout):
                continue
        out.append(r)
    return out


def sort_rows(rows: list[IndexRow], key: str, descending: bool = True) -> list[IndexRow]:
    if key not in SORT_KEYS:
        raise ValueError(f"unknown sort key {key!r}; choose from {', '.join(SORT_KEYS)}")
    return sorted(rows, key=SORT_KEYS[key], reverse=descending)


def _first(rows, pred, what: str) -> IndexRow:
    for r in sorted(rows, key=lambda r: (r.end_step, r.episode_id)):
        if pred(r):
            return r
    raise NoMatch(f"no {what} among {len(rows)} episodes")


def pick(rows: list[IndexRow], selector: str, at_step: int | None = None) -> IndexRow:
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector {selector!r}; choose from {', '.join(SELECTORS)}")
    if not rows:
        raise NoMatch("no episodes match the filters")
    if selector == "best-return":
        return sort_rows(rows, "return")[0]
    if selector == "most-cells":
        return sort_rows(rows, "unique_cells")[0]
    if selector == "first-success":
        return _first(rows, lambda r: r.success, "successful episode")
    if selector == "first-door":
        return _first(rows, lambda r: r.first_door_step is not None, "episode that opened a door")
    if selector == "first-key":
        return _first(rows, lambda r: r.first_key_step is not None, "episode that picked up a key")
    if selector == "latest":
        return max(rows, key=lambda r: (r.end_step, r.episode_id))
    if at_step is None:
        raise ValueError("selector 'at-step' needs at_step")
    return min(rows, key=lambda r: (abs(r.start_step - at_step), r.episode_id))
