# wm-viz

Record → select → render for the `wm` exploration benchmark. `wm` records
every episode as a compact ground-truth trace; `wmviz` lists, filters and
previews them here, and (later, planned — not implemented yet) will render
chosen episodes in Blender.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned because
the Blender `bpy` wheel only exists for 3.11.

    uv sync --group dev              # reader, CLI, previews, tests
    uv sync --extra figures          # + matplotlib PNG previews
    uv sync --extra blender          # + bpy (planned: rendering, not implemented yet)

## Usage

    export WMVIZ_LOGS=/path/to/wm/logs      # or pass --logs
    wmviz runs
    wmviz list p2e-multiroom-n6-s0 --phase eval --sort return
    wmviz pick p2e-multiroom-n6-s0 --phase coverage_eval --at-step 50000
    wmviz show p2e-multiroom-n6-s0/ep000123 --png ep.png
    wmviz show $(wmviz pick p2e-multiroom-n6-s0 --first-success)

Try it right away against the fixtures checked into this repo (after
`uv sync --group dev`, from the repo root):

    uv run wmviz --logs tests/fixtures runs
    uv run wmviz --logs tests/fixtures list doorkey6x6 --phase eval
    uv run wmviz --logs tests/fixtures show $(uv run wmviz --logs tests/fixtures pick doorkey6x6 --first-success)

Episode reference: `<run>/ep000123` (or `<run>/123`). `show` also accepts
a bare `<run>/<id>` and prints an ASCII top-down map plus episode stats;
`--png <path>` additionally writes a matplotlib preview (needs
`uv sync --extra figures`).

Commands: `runs` (lists runs under `--logs` that contain traces), `list RUN`
(episodes of a run as a table, filterable and sortable), `pick RUN` (resolve
one episode to a `<run>/ep000123` reference — for `$(...)` in shells; exactly
one selector required), `show REF` (stats + ASCII map for one episode).

`list --sort` keys: `return`, `unique_cells`, `length`, `start_step`,
`episode_id`, `coverage_pct` (default `episode_id`; `--asc` to reverse the
default descending order; `--limit` caps rows, default 50).

`pick` selectors (choose exactly one): `--best-return`, `--most-cells`,
`--first-success`, `--first-door`, `--first-key`, `--at-step N`, `--latest`.

Filters shared by `list` and `pick`: `--phase`, `--actor`, `--after-step`,
`--before-step`, `--success/--no-success`, `--min-cells`,
`--layout <hash-prefix|seed:N>`.

## Trace format

Read by `wmviz/trace/reader.py`; written by `wm/trace.py`. Spec:
`wm/docs/superpowers/specs/2026-09-09-run-visualization-design.md` §1.
`format_version` 1. One `trace/index.csv` per run, one
`trace/episodes/ep_NNNNNN.npz` per episode with `meta` (JSON), `actions`,
`rewards`, `terminated`, `truncated`, `layout` (MiniGrid `grid.encode()`),
`agent_pos`, `agent_dir`, `carrying`, `door_pos`, `door_open`, optional
`rooms`, `obs`, dream block.

## Layout

    wmviz/trace/reader.py     Index, IndexRow, Episode, Layout, parse_ref
    wmviz/trace/selectors.py  Filters, apply_filters, sort_rows, pick
    wmviz/preview.py          ascii_map, save_png
    wmviz/cli.py              runs, list, pick, show
    scripts/trim_trace.py     trims a full wm trace down to a small fixture
    tests/fixtures/           real traces recorded by wm (contract tests)

Rendering (Blender: `render`, `figure`, `timeline`, `heatmap`) is planned
but not implemented yet — see the design spec above.

## Tests

    uv run pytest -q
