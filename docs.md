# Noridoc: wm-viz

Path: @/

### Overview

`wm-viz` (Python package `wmviz`) is the record → select → render companion to the `wm` reinforcement-learning research repo. `wm` records every training/eval episode as a ground-truth trace on disk; `wmviz` reads those traces, lets a user filter/sort/pick among episodes, and previews them via a terminal ASCII map or a matplotlib PNG ([wmviz/preview.py](wmviz/preview.py)), or renders them in Blender ([wmviz/render.py](wmviz/render.py), optional trail/heatmap overlays in [wmviz/overlays.py](wmviz/overlays.py)). Dream episodes (`phase=dream`, where the world model first tracks a real episode and then imagines its continuation) are rendered from the real trajectory with the recorded observation and the model's own view composited into each frame; the aggregate figure commands are still in progress.

### How it fits into the larger codebase

`wm-viz` is a deliberately separate repository from `wm`, so the training loop never depends on Blender (`bpy` wheels only exist for Python 3.11, which is why this package pins `requires-python = ">=3.11,<3.12"` in [pyproject.toml](pyproject.toml)). The two repos are connected only through a file-format contract, not a code dependency: `wm` writes `logs/<run>/trace/index.csv` plus one `logs/<run>/trace/episodes/ep_NNNNNN.npz` per episode (format_version 1 — MiniGrid grid layout via `grid.encode()`, per-step agent position/heading/carrying/door state, actions, and rewards), plus, for dream episodes, the observations and the model's reconstructed/imagined frames — and `wmviz` reads exactly that layout. The authoritative spec for this format lives in the `wm` repo at `wm/docs/superpowers/specs/2026-09-09-run-visualization-design.md`; [wmviz/trace/reader.py](wmviz/trace/reader.py) and [tests/conftest.py](tests/conftest.py) both cite it directly. `wmviz` does not import `minigrid` — grid-decoding constants are copied locally in `wmviz/trace/reader.py`.

Internally the repo has three layers:

```
wmviz/trace/   data layer  — parse index.csv + .npz episodes, filter/sort/pick rows
wmviz/         CLI + preview — wmviz/cli.py commands, wmviz/preview.py (ASCII/PNG)
tests/         contract tests — synthetic writer (mirrors wm's format) + real trimmed fixtures
```

See [wmviz/docs.md](wmviz/docs.md), [wmviz/trace/docs.md](wmviz/trace/docs.md), and [tests/docs.md](tests/docs.md) for each layer's implementation details.

### Core Implementation

The package is installed as a `uv`-managed project exposing a single console script, `wmviz`, pointing at `wmviz.cli:app` ([pyproject.toml](pyproject.toml)). Core runtime dependencies are `numpy`, `typer`, `rich`, `pillow`, `imageio`/`imageio-ffmpeg`; `matplotlib` (the `figures` extra, also in the `dev` dependency group) is needed only for PNG previews and is imported lazily so the CLI works without it; `bpy` (the `blender` extra) is declared but not imported anywhere yet, since Blender rendering has not been built.

The data flow for every CLI command is: resolve a run directory under `--logs`/`WMVIZ_LOGS` → load its `trace/index.csv` into an `Index` of `IndexRow`s → optionally apply `Filters` and sort → either print a table (`list`), print one resolved episode reference (`pick`), or load the full `Episode` (including its decoded `Layout`) from its `.npz` file and render an ASCII/PNG preview (`show`). All of this logic — and no rendering — is described in [wmviz/docs.md](wmviz/docs.md) and [wmviz/trace/docs.md](wmviz/trace/docs.md).

### Things to Know

- No pixel/video data is required to render an episode: an episode's world state (grid layout, agent trajectory, door states) is fully reconstructible from the trace arrays, so previews and future Blender scenes are built from state rather than replaying recorded frames.
- The `.npz`-per-episode + CSV-index format is chosen for zero-dependency reads and crash safety (each episode file is self-contained; a truncated run still leaves earlier episodes intact) — this is a `wm`-side design decision that `wmviz` simply consumes.
- `.gitignore` excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `renders/`, `*.blend1`, and `.superpowers/` (the latter is the Superpowers skill's local scratch directory, not part of the package).

Created and maintained by Nori.
