import csv
import json
import re

import numpy as np
import pytest

from wmviz.trace import Episode, Index, Layout, find_runs, parse_ref


def test_index_loads_rows_and_types(run_dir):
    idx = Index.load(run_dir)
    assert idx.run_name == "p2e-doorkey6x6-s0" and len(idx.rows) == 7
    r = idx.rows[3]
    assert r.episode_id == 3 and r.phase == "eval" and r.seed == 1001
    assert r.success is True and r.ret == pytest.approx(0.9)
    assert r.first_door_step == 2 and r.first_key_step == 2 and r.rooms_visited is None
    assert idx.rows[0].seed is None and idx.rows[0].first_door_step is None
    assert r.ref(idx.run_name) == "p2e-doorkey6x6-s0/ep000003"
    assert idx.by_id(5).start_step == 300
    assert idx.path_of(r).name == "ep_000003.npz"


def test_index_missing_trace_dir_has_clear_error(tmp_path):
    (tmp_path / "norun").mkdir()
    with pytest.raises(FileNotFoundError, match="trace/index.csv"):
        Index.load(tmp_path / "norun")


def test_index_by_id_error_reports_id_range(run_dir):
    idx = Index.load(run_dir)
    with pytest.raises(KeyError, match=re.escape("no episode 42 (7 episodes, ids 0..6)")):
        idx.by_id(42)


def test_index_load_bad_row_reports_path_and_line_number(run_dir, tmp_path):
    src = run_dir / "trace" / "index.csv"
    dst_dir = tmp_path / "badrun" / "trace"
    dst_dir.mkdir(parents=True)
    dst = dst_dir / "index.csv"
    lines = src.read_text().splitlines(keepends=True)
    fieldnames = lines[0].strip().split(",")
    with open(dst, "w", newline="") as fh:
        fh.writelines(lines)
        bad = {c: "" for c in fieldnames}
        bad["episode_id"] = "not-an-int"
        csv.DictWriter(fh, fieldnames=fieldnames).writerow(bad)
    with pytest.raises(ValueError, match=re.escape(str(dst)) + r":\d+:"):
        Index.load(dst.parent.parent)


def test_episode_load_and_layout(run_dir):
    idx = Index.load(run_dir)
    ep = Episode.load(idx.path_of(idx.by_id(1)))
    assert ep.meta["phase"] == "train_task" and ep.length == 6
    assert ep.agent_pos.shape == (7, 2) and ep.actions.shape == (6,)
    lay = ep.layout
    assert isinstance(lay, Layout) and (lay.width, lay.height) == (7, 7)
    assert (3, 3) in lay.doors and lay.doors[(3, 3)] == "yellow"
    assert lay.keys == {(1, 3): "yellow"} and lay.goals == {(5, 5)}
    assert (3, 1) in lay.walls and (1, 1) not in lay.walls
    assert lay.rooms == []
    assert ep.obs is None and ep.dream is None


def test_episode_with_obs(run_dir):
    idx = Index.load(run_dir)
    ep = Episode.load(idx.path_of(idx.by_id(5)))
    assert ep.obs.shape == (7, 8, 8, 3)


def test_episode_rejects_wrong_format_version(run_dir, tmp_path):
    idx = Index.load(run_dir)
    src = idx.path_of(idx.by_id(0))
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data["meta"])); meta["format_version"] = 2
    data["meta"] = json.dumps(meta)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="format 2"):
        Episode.load(bad)


def test_parse_ref_forms():
    assert parse_ref("run-a/ep000012") == ("run-a", 12)
    assert parse_ref("run-a/12") == ("run-a", 12)
    with pytest.raises(ValueError):
        parse_ref("run-a")


def test_find_runs(logs_dir):
    (logs_dir / "no-trace-run").mkdir()
    assert [p.name for p in find_runs(logs_dir)] == ["p2e-doorkey6x6-s0"]
