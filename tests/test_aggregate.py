"""wmviz/aggregate.py — visit counts and episode grouping (no bpy)."""
from dataclasses import replace

import numpy as np
import pytest

from wmviz.aggregate import (LayoutMismatch, accumulate, cumulative_visit_counts, nearest_to_milestones,
                             same_layout, visit_counts)
from wmviz.trace import Episode, Index


def test_visit_counts_and_cumulative():
    pos = np.array([[1, 1], [2, 1], [2, 1], [2, 2]])
    c = visit_counts(pos, 4, 4)
    assert c.shape == (4, 4) and c[1, 1] == 1 and c[2, 1] == 2 and c[2, 2] == 1 and c.sum() == 4
    cc = cumulative_visit_counts(pos, 4, 4)
    assert cc.shape == (4, 4, 4) and cc[0].sum() == 1 and (cc[-1] == c).all() and cc[2, 2, 1] == 2


def test_same_layout_and_mismatch(run_dir):
    idx = Index.load(run_dir)
    evals = [r for r in idx.rows if r.phase == "eval" and r.seed == 1000]
    assert same_layout(evals) == evals[0].layout_hash
    mixed = list(idx.rows)
    mixed[0] = replace(mixed[0], layout_hash="deadbeef0000")
    with pytest.raises(LayoutMismatch) as e:
        same_layout(mixed)
    assert "--force" in str(e.value) and "deadbeef0000" in str(e.value)
    assert same_layout(mixed, force=True) == mixed[0].layout_hash


def test_nearest_to_milestones(run_dir):
    idx = Index.load(run_dir)
    rows = nearest_to_milestones(idx.rows, [0, 110, 1000])
    assert [r.start_step for r in rows] == [0, 100, 300]
    with pytest.raises(ValueError):
        nearest_to_milestones([], [1])


def test_accumulate_sums_episodes(run_dir):
    idx = Index.load(run_dir)
    eps = [Episode.load(idx.path_of(r)) for r in idx.rows[:2]]
    W, H = eps[0].layout.width, eps[0].layout.height
    total = accumulate(eps, W, H)
    assert total.sum() == sum(len(e.agent_pos) for e in eps)
