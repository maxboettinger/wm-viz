"""Real traces recorded by wm — the contract test against the actual writer."""
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmviz.cli import app
from wmviz.trace import Episode, Index

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("run", ["doorkey6x6", "multiroom-n4s5"])
def test_real_index_and_episodes_load(run):
    idx = Index.load(FIX / run)
    assert idx.rows and {r.phase for r in idx.rows} & {"eval", "coverage_eval"}
    for r in idx.rows:
        ep = Episode.load(idx.path_of(r))
        assert ep.length == r.length and ep.agent_pos.shape == (r.length + 1, 2)
        assert ep.door_open.shape == (r.length + 1, ep.door_pos.shape[0])
        assert ep.meta["format_version"] == 1


def test_multiroom_has_rooms():
    idx = Index.load(FIX / "multiroom-n4s5")
    ep = Episode.load(idx.path_of(idx.rows[0]))
    # MiniGrid-MultiRoom-N4-S5-v0 registers minNumRooms=maxNumRooms=6 despite the
    # "N4" in its id (upstream minigrid naming quirk, not a wm/wmviz bug) — every
    # episode in this fixture has 6 rooms. See task-13 report for the trace.
    assert len(ep.layout.rooms) == 6 and idx.rows[0].rooms_visited is not None


def test_cli_on_real_fixtures():
    r = CliRunner().invoke(app, ["--logs", str(FIX), "list", "doorkey6x6", "--phase", "eval"])
    assert r.exit_code == 0 and "doorkey6x6/ep" in r.stdout
    ref = CliRunner().invoke(app, ["--logs", str(FIX), "pick", "doorkey6x6", "--latest"]).stdout.strip()
    r = CliRunner().invoke(app, ["--logs", str(FIX), "show", ref])
    assert r.exit_code == 0 and "#" in r.stdout


def test_real_dream_fixture_loads_and_composites():
    import numpy as np

    from wmviz.render import RenderConfig, compose_frame, dream_panels
    idx = Index.load(FIX / "dream-doorkey6x6")
    rows = [r for r in idx.rows if r.phase == "dream"]
    assert {r.actor for r in rows} == {"task", "explorer"}
    ep = Episode.load(idx.path_of(rows[0]))
    ds = ep.dream["dream_start"]
    assert ep.obs.shape[1:] == ep.dream["dream_frames"].shape[1:] and ep.dream["recon_frames"].shape[0] == ds
    real, model, dreaming = dream_panels(ep, ds)
    assert dreaming and model.shape == real.shape
    # 216x384 (not 108x192): two pip insets at frac=0.25 do not fit a 108 px tall base, see test_dream.py
    out = compose_frame([np.zeros((216, 384, 3), np.uint8)], ep, rows[0], ds, RenderConfig(out=Path("x.mp4")))
    assert out.shape == (216, 384, 3)
