# Real trace fixtures

Trimmed copies of two short `wm` training runs, used as the contract test
(`tests/test_fixtures_real.py`) against the actual trace writer — as opposed
to the synthetic traces built by `tests/conftest.py`.

## Provenance

Produced in the `wm` repo (Part A of the run-visualization plan) with:

```bash
uv run python train_dreamer.py --env MiniGrid-DoorKey-6x6-v0 --obs-mode pixel --device cpu \
  --n-envs 2 --steps 400 --warmup-steps 50 --batch-size 4 --seq-len 16 \
  --eval-every 400 --eval-episodes 2 --expl-eval-episodes 2 --exploration plan2explore \
  --video-every 0 --wandb off --log-dir /tmp/wmviz-fixtures --run-name doorkey6x6
uv run python train_dreamer.py --env MiniGrid-MultiRoom-N4-S5-v0 --obs-mode pixel --device cpu \
  --n-envs 2 --steps 400 --warmup-steps 50 --batch-size 4 --seq-len 16 \
  --eval-every 400 --eval-episodes 2 --expl-eval-episodes 2 --exploration plan2explore \
  --video-every 0 --wandb off --log-dir /tmp/wmviz-fixtures --run-name multiroom-n4s5
```

then trimmed with `scripts/trim_trace.py <src> <dst> 4` (keep the first 4
training episodes plus every eval/coverage episode; drop the rest, dropped
rows removed from `index.csv`):

```bash
uv run python scripts/trim_trace.py /tmp/wmviz-fixtures/doorkey6x6 tests/fixtures/doorkey6x6 4
uv run python scripts/trim_trace.py /tmp/wmviz-fixtures/multiroom-n4s5 tests/fixtures/multiroom-n4s5 4
```

With only 2 envs and the default `--task-collect-frac 0.25`, neither run
produced any `train_task` episodes — expected, and not needed by these tests.

## Contents

- `doorkey6x6/trace/` — 10 episodes (2 train_explorer, 4 eval, 4 coverage_eval), all kept (fewer than 4 training episodes existed).
- `multiroom-n4s5/trace/` — 12 of 14 episodes kept (4 of 6 train_explorer, all 4 eval, all 4 coverage_eval).

## Known quirk: MultiRoom-N4-S5 has 6 rooms, not 4

`MiniGrid-MultiRoom-N4-S5-v0` registers `minNumRooms=maxNumRooms=6` in
`minigrid/__init__.py` — the "N4" in the env id does not match the actual
room count generated (an upstream minigrid naming inconsistency, not a
wm/wmviz bug). Every episode in this fixture has `len(layout.rooms) == 6`;
`test_fixtures_real.py::test_multiroom_has_rooms` asserts 6 accordingly.
