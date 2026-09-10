# wm-viz

Browse and preview the episodes recorded by the [wm](https://github.com/neural-data-science-lab/wm)
exploration benchmark.

`wm` writes a compact ground-truth trace of every episode it plays: the
maze layout, the agent's position and heading at each step, what it
carried, which doors it opened, actions and rewards. `wmviz` reads those
traces so you can answer questions like *"show me the first episode where
the Plan2Explore agent reached the goal"* or *"which coverage-eval episode
at step 50k visited the most cells"* — and look at it, right in the
terminal or as a PNG.

```
$ wmviz show doorkey6x6/ep000004
doorkey6x6/ep000004  MiniGrid-DoorKey-6x6-v0  phase=train_explorer actor=explorer seed=  steps 0→654
length=327 return=0.182 success=True cells=11 coverage=85% key@102 door@119 goal@327 rooms= layout=9fe74bfc54b6
######
#72#.#
#**#.#
#S*D*#
#*K#E#
######
# wall  D door  K key  G goal  S start  E end  digits = visits
```

Rendering episodes as Blender animations is the next planned step and is
not implemented yet; today `wmviz` covers *record → select → preview*.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (pinned, because
the Blender `bpy` wheel only exists for 3.11).

```bash
git clone https://github.com/maxboettinger/wm-viz
cd wm-viz
uv sync --group dev          # CLI, reader, ASCII + PNG previews, tests
```

Optional extras:

```bash
uv sync --extra figures      # matplotlib PNG previews without the dev group
uv sync --extra blender      # bpy — the Blender renderer (`wmviz render`)
```

Run the CLI with `uv run wmviz …`, or activate the venv and call `wmviz`
directly.

## Quick start

Point `wmviz` at a `wm` logs directory, either with `--logs` or the
`WMVIZ_LOGS` environment variable. Two small real runs are checked in as
test fixtures, so you can try everything without training anything:

```bash
export WMVIZ_LOGS=tests/fixtures      # later: /path/to/wm/logs

wmviz runs                            # which runs have traces?
wmviz list doorkey6x6 --phase eval    # episodes of one run, as a table
wmviz pick doorkey6x6 --first-success # resolve one episode to a reference
wmviz show doorkey6x6/ep000004        # stats + ASCII map
wmviz show doorkey6x6/ep000004 --png ep4.png
```

`runs` and `list` print tables:

```
$ wmviz runs
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓
┃ run            ┃ env                         ┃ exploration  ┃ episodes ┃ last step ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
│ doorkey6x6     │ MiniGrid-DoorKey-6x6-v0     │ plan2explore │ 10       │ 800       │
│ multiroom-n4s5 │ MiniGrid-MultiRoom-N4-S5-v0 │ plan2explore │ 12       │ 800       │
└────────────────┴─────────────────────────────┴──────────────┴──────────┴───────────┘

$ wmviz list multiroom-n4s5 --limit 3
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ ref                     ┃ phase         ┃ actor    ┃ seed ┃ start ┃ len ┃ return ┃ ok ┃ cells ┃ cov% ┃ key@ ┃ door@ ┃ goal@ ┃ rooms ┃ layout ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ multiroom-n4s5/ep000013 │ coverage_eval │ explorer │ 2001 │ 800   │ 120 │ 0      │    │ 4     │ 11   │      │       │       │ 1     │ f71486 │
│ multiroom-n4s5/ep000012 │ coverage_eval │ explorer │ 2000 │ 800   │ 120 │ 0      │    │ 7     │ 21   │      │ 1     │       │ 1     │ 3daa99 │
│ multiroom-n4s5/ep000011 │ eval          │ task     │ 1001 │ 800   │ 120 │ 0      │    │ 1     │ 3    │      │       │       │ 1     │ 30b4c6 │
└─────────────────────────┴───────────────┴──────────┴──────┴───────┴─────┴────────┴────┴───────┴──────┴──────┴───────┴───────┴───────┴────────┘
… 9 more (use --limit)
```

`pick` prints a single reference, which makes it composable in a shell:

```bash
wmviz show $(wmviz pick p2e-n6-s0 --best-return --phase eval)
wmviz show $(wmviz pick p2e-n6-s0 --most-cells --phase coverage_eval --at-step 50000) --png best.png
```

## Command reference

| Command | What it does |
|---|---|
| `wmviz runs` | Lists every run under `--logs` that contains traces, with env, exploration method, episode count and last recorded step |
| `wmviz list RUN` | Prints the episodes of a run as a table. Accepts the filters below plus `--sort`, `--asc`, `--limit` |
| `wmviz pick RUN` | Applies the filters, then one selector, and prints exactly one `<run>/ep000123` reference. Fails with a hint if nothing matches |
| `wmviz show REF` | Prints an episode's stats and an ASCII top-down map; `--png PATH` also writes a matplotlib preview |
| `wmviz render TARGET` | Renders one episode (`<run>/ep000123`, or a run plus one selector) as a Blender mp4 or `--still N` PNG; `--camera` defaults to `topdown`, or `fpv` for dream episodes, which get the real observation and the world model's view composited in (`--dream-layout pip|split`, white frame while the model tracks reality, black once it dreams) |

Episode references look like `<run>/ep000123`; `<run>/123` is accepted
too.

**Filters** (shared by `list` and `pick`):

| Option | Keeps episodes that… |
|---|---|
| `--phase P` | belong to phase `P`: `train_task`, `train_explorer`, `train_random`, `eval`, `coverage_eval`, `dream` |
| `--actor A` | were played by actor `A`: `task`, `explorer`, `random` |
| `--after-step N` / `--before-step N` | started at or after / ended at or before global env step `N` |
| `--success` / `--no-success` | reached the goal / did not |
| `--min-cells N` | visited at least `N` distinct cells |
| `--layout HASH` | were played on the layout whose hash starts with `HASH` |
| `--layout seed:N` | were reset with seed `N` (eval and coverage-eval episodes carry a seed) |

**Sorting** (`list`): `--sort return|unique_cells|length|start_step|episode_id|coverage_pct`
(default `episode_id`, descending; `--asc` flips it; `--limit` caps rows,
default 50).

**Selectors** (`pick`, exactly one):

| Selector | Picks the episode with… |
|---|---|
| `--best-return` | the highest return |
| `--most-cells` | the most distinct cells visited |
| `--first-success` | the earliest goal reach |
| `--first-key` / `--first-door` | the earliest key pickup / door opening |
| `--at-step N` | the closest start step to `N` |
| `--latest` | the highest start step |

## Reading the map

`show` draws the layout top-down. Each cell shows how many times the
agent stood on it: digits for 1–9, `*` for ten or more. `S` and `E` mark
where the episode started and ended, `D`/`K`/`G` are doors, keys and the
goal, `#` walls, `.` is unvisited floor. The PNG preview (`--png`) draws
the same layout with the trajectory as a line coloured by time (dark
start, yellow end), a white circle at the start and a black square at the
end.

The header line above the map is the index row for that episode: phase,
actor, reset seed, global step range, length, return, success, cells and
coverage, the step of the first key / door / goal event, rooms entered
(MultiRoom only) and the layout hash.

## Using the reader from Python

The CLI is a thin layer over `wmviz.trace`, which you can use directly in
notebooks or scripts:

```python
from wmviz.trace.reader import Index, Episode
from wmviz.trace.selectors import Filters, apply_filters, sort_rows, pick

idx = Index.load("tests/fixtures/doorkey6x6")
rows = sort_rows(apply_filters(idx.rows, Filters(phase="eval")), "return")
best = pick(idx.rows, "first-success")

ep = Episode.load(idx.path_of(best))
ep.agent_pos          # (T+1, 2) int64 — x, y per step
ep.agent_dir          # (T+1,) 0=east 1=south 2=west 3=north
ep.door_open          # (T+1, n_doors) bool
ep.layout.rooms       # list of (x, y, w, h) for MultiRoom envs
ep.layout.doors       # {(x, y): colour}
ep.meta["env_id"], ep.length, ep.ret
```

`Layout.from_grid` decodes MiniGrid's `grid.encode()` array into walls,
doors, keys, goals and lava without importing `minigrid`, so `wmviz` has
no dependency on the training stack.

## Trace format

`wm` writes `logs/<run>/trace/index.csv` (one row per episode, appended
as episodes finish) and one `logs/<run>/trace/episodes/ep_NNNNNN.npz`
per episode. Each npz contains a JSON `meta` string and these arrays:

| Array | Shape | Content |
|---|---|---|
| `layout` | `(W, H, 3)` uint8 | MiniGrid `grid.encode()` at reset: object, colour, state per cell |
| `rooms` | `(n_rooms, 4)` | MultiRoom only: `x, y, w, h` per room |
| `actions` | `(T,)` | Action taken at each step |
| `rewards` | `(T,)` float32 | Reward received at each step |
| `terminated`, `truncated` | scalar bool | How the episode ended |
| `agent_pos` | `(T+1, 2)` | Agent x, y before each step and after the last |
| `agent_dir` | `(T+1,)` | Agent heading |
| `carrying` | `(T+1,)` | Object index carried, `-1` for nothing |
| `door_pos` | `(n_doors, 2)` | Door cells, fixed for the episode |
| `door_open` | `(T+1, n_doors)` | Whether each door was open |
| `obs` | `(T+1, H, W, 3)` uint8 | Observations, only when `wm` ran with `--trace-pixels` enabling that phase |
| `dream_start` | scalar int | Dream episodes only: index of the first imagined state |
| `recon_frames` | `(dream_start, H, W, 3)` uint8 | Dream episodes only: the world model's posterior reconstructions of states `0..dream_start-1` |
| `dream_frames` | `(horizon, H, W, 3)` uint8 | Dream episodes only: imagined states from `dream_start` on |

The format is versioned (`format_version` 1 in `meta`); the reader
rejects other versions rather than guessing. The writer lives in
`wm/trace.py` in the `wm` repo; the two projects share no code, only
this file contract.

## Repository layout

```
wmviz/trace/reader.py     Index, IndexRow, Episode, Layout, parse_ref, find_runs
wmviz/trace/selectors.py  Filters, apply_filters, sort_rows, pick
wmviz/preview.py          ascii_map, save_png
wmviz/cli.py              the wmviz command (runs, list, pick, show, render, figure)
scripts/trim_trace.py     trims a full wm trace down to a small fixture
tests/                    unit tests + contract tests against real wm traces
tests/fixtures/           three real, trimmed runs, one of them dream episodes (see tests/fixtures/README.md)
```

## Development

```bash
uv sync --group dev
uv run pytest -q
```

`tests/conftest.py` builds synthetic traces that mirror `wm`'s writer;
`tests/test_fixtures_real.py` checks the reader against real recordings
so a format drift in `wm` shows up here. To refresh the fixtures after a
`wm` format change, follow `tests/fixtures/README.md`.

Note that `MiniGrid-MultiRoom-N4-S5-v0` generates six rooms, not four —
an upstream naming quirk in `minigrid`, documented in the fixtures README.

## Roadmap

- Blender figure commands (`timeline`, `heatmap`) on top of the existing
  `render` pipeline.
