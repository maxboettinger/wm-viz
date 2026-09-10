# Noridoc: wmviz.trace

Path: @/wmviz/trace

### Overview

The trace subpackage is the data-access layer of `wmviz`: it reads the `wm` training repo's on-disk episode traces into typed Python objects (`Index`, `IndexRow`, `Episode`, `Layout`) and provides pure filter/sort/pick functions over them. It has no CLI or rendering concerns — everything here is either file I/O ([wmviz/trace/reader.py](wmviz/trace/reader.py)) or side-effect-free selection logic ([wmviz/trace/selectors.py](wmviz/trace/selectors.py)).

### How it fits into the larger codebase

This is the foundation both [wmviz/cli.py](wmviz/cli.py) and [wmviz/preview.py](wmviz/preview.py) build on ([@/wmviz/docs.md](wmviz/docs.md)). The CLI's `runs`, `list`, `pick`, and `show` commands all start by loading an `Index` or `Episode` from here, then filter/sort/pick rows before printing or previewing. The planned Blender renderer (not yet implemented) is expected to consume `Episode`/`Layout` the same way `wmviz/preview.py` does.

The public surface is re-exported through [wmviz/trace/__init__.py](wmviz/trace/__init__.py): `Index`, `IndexRow`, `Episode`, `Layout`, `parse_ref`, `find_runs`, `FORMAT_VERSION`. Everything else (`selectors.py`'s contents) is imported directly from `wmviz.trace.selectors`.

This module is a contract mirror of the `wm` repo's trace writer. The format it reads — `logs/<run>/trace/index.csv` plus one `.npz` per episode under `trace/episodes/` — is defined by `wm`, not by `wmviz`; the authoritative spec is `wm/docs/superpowers/specs/2026-09-09-run-visualization-design.md` §1. `reader.py`'s docstring points back at that spec. `wmviz` does not depend on the `minigrid` package; the object/color index tables (`IDX_TO_OBJECT`, `IDX_TO_COLOR`) in [wmviz/trace/reader.py](wmviz/trace/reader.py) are a copy of `minigrid.core.constants`.

### Core Implementation

**`Index`** ([wmviz/trace/reader.py](wmviz/trace/reader.py)) loads `trace/index.csv` for one run directory into a list of `IndexRow`. `Index.load` raises `FileNotFoundError` with a message suggesting the run wasn't recorded with `--trace auto` if the CSV is missing. `IndexRow.from_csv` parses each CSV row, converting empty-string fields to `None` via `_opt_int`/`_opt_float` (columns like `seed`, `coverage_pct`, `first_key_step`, `first_door_step`, `first_goal_step`, `rooms_visited` are all optional and may be absent for a given phase/episode). `IndexRow.ref(run_name)` formats the canonical episode reference string `<run>/ep<NNNNNN>` (zero-padded to 6 digits) — the same format `parse_ref` parses back (accepting both `ep000012` and bare `12` as the id).

**`find_runs(logs_dir)`** scans a logs directory for subdirectories containing `trace/index.csv`, sorted by name — this backs the CLI's `runs` command.

**`Episode.load(path)`** ([wmviz/trace/reader.py](wmviz/trace/reader.py)) opens one `ep_NNNNNN.npz` file and decodes: `meta` (JSON blob with `format_version`, `env_id`, `phase`, `actor`, `seed`, etc.), per-step `actions`/`rewards`, `terminated`/`truncated` flags, `agent_pos`/`agent_dir`/`carrying` traces, `door_pos`/`door_open`, and a `Layout` built from the episode's `layout` grid (plus `rooms` if present). Two fields are optional per-episode: `obs` (raw pixel observations, only present if the run recorded them) and `dream` (present only for dream-visualization episodes: a dict with `dream_start`, the index of the first imagined state; `dream_frames`, the model's imagined frames for states `dream_start` onwards; and optionally `recon_frames`, its posterior reconstructions of states `0..dream_start-1` — so the model view of state `t` is `recon_frames[t]` before `dream_start` and `dream_frames[t - dream_start]` after, which is what `wmviz/render.py`'s `dream_panels` computes). `Episode.load` enforces two invariants at load time: it rejects any trace whose `meta["format_version"] != FORMAT_VERSION` (currently `1`) and any trace whose `meta["env_family"] != "minigrid"`, both raising `ValueError`.

**`Layout.from_grid(grid, rooms=None)`** decodes a MiniGrid `(W, H, 3)` uint8 grid (object index, color index, state) into per-type coordinate collections: `walls` (a set), `doors`/`keys` (dicts of position → color name), `goals`/`lava` (sets). This mirrors `grid.encode()`'s layout as produced by `wm`, decoded here using the copied `IDX_TO_OBJECT`/`IDX_TO_COLOR` tables rather than importing `minigrid`.

**`selectors.py`** operates purely on `list[IndexRow]` with no I/O:
- `apply_filters(rows, Filters)` — `Filters` is a dataclass of optional predicates (`phase`, `actor`, `after_step`/`before_step` on `start_step`, `success`, `min_cells` on `unique_cells`, and `layout` — either a `layout_hash` prefix or a `"seed:<n>"` string matched against `IndexRow.seed`).
- `sort_rows(rows, key, descending=True)` — `key` must be one of `SORT_KEYS` (`return`, `unique_cells`, `length`, `start_step`, `episode_id`, `coverage_pct`); unknown keys raise `ValueError` naming the valid set. Rows with `coverage_pct is None` sort as `-1.0` (last when descending).
- `pick(rows, selector, at_step=None)` — resolves one row by `SELECTORS` (`best-return`, `most-cells`, `first-success`, `first-door`, `first-key`, `at-step`, `latest`). `first-*` selectors use `_first`, which scans rows ordered by `(end_step, episode_id)` and returns the first matching one — i.e. "first" means earliest in trace chronology, not lowest episode id. `at-step` (needs `at_step`) picks the row whose `start_step` is closest to the target, ties broken by lowest `episode_id`. An empty `rows` list, or no row satisfying the selector's predicate, raises `NoMatch` with a message naming what was searched for and how many candidates were considered.

### Things to Know

- `IndexRow.success` parses from CSV values `"1"`, `"True"`, or `"true"` — any other string (including empty) is falsy.
- `Layout.from_grid` iterates the full grid cell-by-cell; `rooms` (when supplied, e.g. for MultiRoom environments) is reshaped to `(-1, 4)` tuples of ints and stored on `Layout.rooms` — for environments without room metadata this stays the default empty list.
- `_first`'s chronological tie-break (`(end_step, episode_id)`) means `first-door`/`first-key`/`first-success` reflect trace order, which is not the same as CSV row order unless episodes were written in sequence.
- `parse_ref` accepts run names containing anything except `/`, and episode ids as plain digits with an optional `ep` prefix; it raises `ValueError` (not a custom exception) on malformed refs, with the expected `<run>/ep000123` format spelled out in the message.

Created and maintained by Nori.
