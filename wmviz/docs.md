# Noridoc: wmviz

Path: @/wmviz

### Overview

`wmviz` is the top-level package: a Typer CLI ([wmviz/cli.py](wmviz/cli.py)) plus a no-Blender-required preview module ([wmviz/preview.py](wmviz/preview.py)) that together let a user list, filter, pick, and eyeball episodes recorded by the `wm` training repo, without opening Blender. The actual trace parsing and selection logic lives one level down in [wmviz/trace/](wmviz/trace/docs.md); this folder is the user-facing layer built on top of it.

### How it fits into the larger codebase

`wmviz` is a standalone companion package to `wm` (see [@/docs.md](docs.md) at the repo root for the two-repo relationship and rationale). Within this repo, `wmviz/cli.py` is the single entry point registered as the `wmviz` console script (`[project.scripts]` in [pyproject.toml](pyproject.toml)); `wmviz/preview.py` is only ever imported by the CLI's `show` command, not used standalone. Both depend on [wmviz/trace/](wmviz/trace/docs.md) for `Index`, `Episode`, `Layout`, `find_runs`, `parse_ref`, and the filter/sort/pick functions. [tests/](tests/docs.md) exercises this layer end-to-end via Typer's `CliRunner`.

Blender-based rendering (`scene/`, `animate`, `cameras`, `overlays`, `render`, and CLI commands `render`/`figure`/`timeline`/`heatmap`) is planned for a future iteration and does not exist in this package yet — the `blender` extra (`bpy`) declared in [pyproject.toml](pyproject.toml) is not imported anywhere in `wmviz/` today.

### Core Implementation

**CLI ([wmviz/cli.py](wmviz/cli.py))** is a Typer `app` with a module-level `State` singleton (`state.logs`, default `Path("logs")`) set by the `--logs` option or `WMVIZ_LOGS` env var in the `@app.callback()`. Four commands:
- `runs` — lists run directories under `--logs` that contain `trace/index.csv` (via `find_runs`), with env id and exploration method read from each run's *last* episode's `meta`.
- `list` — loads one run's `Index`, applies the shared `Filters` (phase/actor/after-step/before-step/success/min-cells/layout), sorts via `sort_rows`, and renders a Rich table capped at `--limit` (default 50), printing a "… N more" hint if truncated.
- `pick` — same filters, but resolves exactly one episode via one selector flag (`--best-return`, `--most-cells`, `--first-success`, `--first-door`, `--first-key`, `--at-step N`, `--latest`) and prints only its `<run>/ep000NNN` ref to stdout, so it composes in shell substitution (e.g. `wmviz show $(wmviz pick run --best-return)`). Passing zero or more than one selector flag fails with exit code 2.
- `show` — resolves one episode ref via `resolve_episode` (which calls `parse_ref` → `Index.load` → `Index.by_id` → `Episode.load`), prints its metadata/stats line and an ASCII map (`preview.ascii_map`), and optionally writes a PNG (`preview.save_png`) if `--png PATH` is given.

Errors funnel through `_fail(msg, code=1)`, which prints `error: <msg>` in red and raises `typer.Exit(code)`; validation errors from `pick`/`sort_rows` use exit code 2, everything else exit code 1.

**Preview ([wmviz/preview.py](wmviz/preview.py))** has two independent functions:
- `ascii_map(ep)` — builds a `Counter` of agent-visit counts per cell from `ep.agent_pos`, then renders the layout as a grid of characters: `#` wall, `D` door, `K` key, `G` goal, `~` lava, `.`/digit/`*` for unvisited/visited-N-times/visited-10+-times floor, with `S` (start) and `E` (end, overrides any other marker including `G`) overlaid last.
- `save_png(ep, out, cell_px=32)` — renders the same layout plus the agent's path as a matplotlib figure (walls/doors/goals/lava as colored rectangles, keys as markers, the path as a `viridis`-colored `LineCollection`, start as a white circle, end as a black square) and saves it to `out`. Matplotlib is imported lazily inside the function; if unavailable it raises `RuntimeError` telling the user to `uv sync --extra figures` rather than failing at module import time.

### Things to Know

- `Filters`/selector construction is duplicated as explicit Typer `Option` objects (`_PHASE`, `_ACTOR`, etc.) at module level in `cli.py` and reused across `list_cmd` and `pick_cmd`, because Typer has no built-in option-group mechanism — the comment in the code calls this out directly.
- `show`'s stats line always prints `key@`/`door@`/`goal@`/`rooms=` fields even when they're `None` (rendered as empty string by `_fmt`), since not every phase records those milestones.
- `COLOR_RGB` in `preview.py` is a small fixed palette (`red`/`green`/`blue`/`purple`/`yellow`/`grey`) matching the `IDX_TO_COLOR` values decoded in [wmviz/trace/reader.py](wmviz/trace/reader.py) — the two must stay in sync for door/key colors to render correctly.

Created and maintained by Nori.
