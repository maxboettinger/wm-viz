"""wmviz — list, pick, preview (and later render) recorded wm episodes."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .trace import Episode, Index, IndexRow, find_runs, parse_ref
from .trace.selectors import SORT_KEYS, Filters, NoMatch, apply_filters, pick as pick_row, sort_rows

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
# Rich defaults to an 80-column width when stdout isn't a tty (e.g. under
# CliRunner, or piped output) which truncates our wide tables with ellipses.
# Force a generous fixed width so `list`/`runs` render in full everywhere;
# real terminals still get full-width rendering since rich just word-wraps
# any row content that doesn't fit the actual window.
console = Console(highlight=False, width=200)


class State:
    logs: Path = Path("logs")


state = State()


@app.callback()
def _main(logs: Path = typer.Option(None, "--logs", help="Directory with logs/<run>/ (env WMVIZ_LOGS)")):
    state.logs = Path(logs or os.environ.get("WMVIZ_LOGS", "logs"))


def _fail(msg: str, code: int = 1):
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code)


def _index(run: str) -> Index:
    try:
        return Index.load(state.logs / run)
    except FileNotFoundError as e:
        _fail(str(e))


def _fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _filters(phase, actor, after_step, before_step, success, min_cells, layout) -> Filters:
    return Filters(phase=phase, actor=actor, after_step=after_step, before_step=before_step,
                   success=success, min_cells=min_cells, layout=layout)


# Shared filter options (Typer has no option groups; repeat explicitly).
_PHASE = typer.Option(None, "--phase")
_ACTOR = typer.Option(None, "--actor")
_AFTER = typer.Option(None, "--after-step")
_BEFORE = typer.Option(None, "--before-step")
_SUCCESS = typer.Option(None, "--success/--no-success")
_MIN_CELLS = typer.Option(None, "--min-cells")
_LAYOUT = typer.Option(None, "--layout", help="layout-hash prefix or seed:<n>")


@app.command()
def runs():
    """Runs under --logs that contain traces."""
    found = find_runs(state.logs) if state.logs.exists() else []
    if not found:
        _fail(f"no runs with trace/index.csv under {state.logs}")
    t = Table("run", "env", "exploration", "episodes", "last step")
    for p in found:
        idx = Index.load(p)
        meta = {}
        if idx.rows:
            with np.load(idx.path_of(idx.rows[-1])) as z:
                meta = json.loads(str(z["meta"]))
        last = max((r.end_step for r in idx.rows), default=0)
        t.add_row(p.name, meta.get("env_id", ""), meta.get("exploration", ""), str(len(idx.rows)), str(last))
    console.print(t)


@app.command("list")
def list_cmd(run: str, phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR,
             after_step: Optional[int] = _AFTER, before_step: Optional[int] = _BEFORE,
             success: Optional[bool] = _SUCCESS, min_cells: Optional[int] = _MIN_CELLS,
             layout: Optional[str] = _LAYOUT,
             sort: str = typer.Option("episode_id", "--sort", help=", ".join(SORT_KEYS)),
             asc: bool = typer.Option(False, "--asc"), limit: int = typer.Option(50, "--limit")):
    """Episodes of RUN as a table."""
    idx = _index(run)
    rows = apply_filters(idx.rows, _filters(phase, actor, after_step, before_step, success, min_cells, layout))
    try:
        rows = sort_rows(rows, sort, descending=not asc)
    except ValueError as e:
        _fail(str(e), 2)
    t = Table("ref", "phase", "actor", "seed", "start", "len", "return", "ok", "cells", "cov%",
              "key@", "door@", "goal@", "rooms", "layout")
    for r in rows[:limit]:
        t.add_row(r.ref(idx.run_name), r.phase, r.actor, _fmt(r.seed), str(r.start_step), str(r.length),
                  _fmt(r.ret), "✓" if r.success else "", str(r.unique_cells),
                  "" if r.coverage_pct is None else f"{100 * r.coverage_pct:.0f}",
                  _fmt(r.first_key_step), _fmt(r.first_door_step), _fmt(r.first_goal_step),
                  _fmt(r.rooms_visited), r.layout_hash[:6])
    console.print(t)
    if len(rows) > limit:
        console.print(f"… {len(rows) - limit} more (use --limit)")


@app.command("pick")
def pick_cmd(run: str, phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR,
             after_step: Optional[int] = _AFTER, before_step: Optional[int] = _BEFORE,
             success: Optional[bool] = _SUCCESS, min_cells: Optional[int] = _MIN_CELLS,
             layout: Optional[str] = _LAYOUT,
             best_return: bool = typer.Option(False, "--best-return"),
             most_cells: bool = typer.Option(False, "--most-cells"),
             first_success: bool = typer.Option(False, "--first-success"),
             first_door: bool = typer.Option(False, "--first-door"),
             first_key: bool = typer.Option(False, "--first-key"),
             at_step: Optional[int] = typer.Option(None, "--at-step"),
             latest: bool = typer.Option(False, "--latest")):
    """Resolve one episode and print its reference (for $(...) in shells)."""
    chosen = [name for name, on in (("best-return", best_return), ("most-cells", most_cells),
                                    ("first-success", first_success), ("first-door", first_door),
                                    ("first-key", first_key), ("at-step", at_step is not None),
                                    ("latest", latest)) if on]
    if len(chosen) != 1:
        _fail("give exactly one selector: --best-return, --most-cells, --first-success, "
              "--first-door, --first-key, --at-step N, --latest", 2)
    idx = _index(run)
    rows = apply_filters(idx.rows, _filters(phase, actor, after_step, before_step, success, min_cells, layout))
    try:
        row = pick_row(rows, chosen[0], at_step=at_step)
    except NoMatch as e:
        _fail(str(e))
    print(row.ref(idx.run_name))


def resolve_episode(logs: Path, ref: str) -> tuple[Index, IndexRow, Episode]:
    run, ep_id = parse_ref(ref)
    idx = Index.load(logs / run)
    row = idx.by_id(ep_id)
    return idx, row, Episode.load(idx.path_of(row))


@app.command()
def show(ref: str, png: Optional[Path] = typer.Option(None, "--png", help="also write a PNG preview")):
    """Stats and an ASCII top-down map of one episode (<run>/ep000123)."""
    from .preview import ascii_map, save_png
    try:
        idx, row, ep = resolve_episode(state.logs, ref)
    except (ValueError, FileNotFoundError, KeyError) as e:
        _fail(str(e).strip("'"))
    m = ep.meta
    console.print(f"[bold]{row.ref(idx.run_name)}[/bold]  {m.get('env_id')}  phase={row.phase} actor={row.actor}"
                  f" seed={_fmt(row.seed)}  steps {row.start_step}→{row.end_step}")
    console.print(f"length={row.length} return={_fmt(row.ret)} success={row.success} "
                  f"cells={row.unique_cells} coverage={'' if row.coverage_pct is None else f'{100 * row.coverage_pct:.0f}%'} "
                  f"key@{_fmt(row.first_key_step)} door@{_fmt(row.first_door_step)} goal@{_fmt(row.first_goal_step)} "
                  f"rooms={_fmt(row.rooms_visited)} layout={row.layout_hash}")
    console.print(ascii_map(ep))
    console.print("[dim]# wall  D door  K key  G goal  S start  E end  digits = visits[/dim]")
    if png is not None:
        out = save_png(ep, png)
        console.print(f"wrote {out}")
