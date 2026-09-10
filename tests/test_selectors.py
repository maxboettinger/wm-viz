import pytest

from wmviz.trace import Index
from wmviz.trace.selectors import Filters, NoMatch, apply_filters, pick, sort_rows


@pytest.fixture
def rows(run_dir):
    return Index.load(run_dir).rows


def test_filters(rows):
    assert [r.episode_id for r in apply_filters(rows, Filters(phase="eval"))] == [2, 3, 5]
    assert [r.episode_id for r in apply_filters(rows, Filters(actor="explorer"))] == [0, 4]
    assert [r.episode_id for r in apply_filters(rows, Filters(after_step=150))] == [4, 5, 6]
    assert [r.episode_id for r in apply_filters(rows, Filters(before_step=50))] == [0, 1]
    assert [r.episode_id for r in apply_filters(rows, Filters(success=True))] == [3]
    assert [r.episode_id for r in apply_filters(rows, Filters(layout="seed:1000"))] == [2, 5]
    h = rows[0].layout_hash
    assert len(apply_filters(rows, Filters(layout=h[:6]))) == 7
    assert apply_filters(rows, Filters(min_cells=99)) == []


def test_sort(rows):
    assert [r.episode_id for r in sort_rows(rows, "return")][0] == 3
    assert [r.episode_id for r in sort_rows(rows, "start_step", descending=False)][:2] == [0, 1]
    with pytest.raises(ValueError):
        sort_rows(rows, "bogus")


def test_pick_selectors(rows):
    assert pick(rows, "best-return").episode_id == 3
    assert pick(rows, "most-cells").episode_id == 3        # T=8 → most unique cells
    assert pick(rows, "first-success").episode_id == 3
    assert pick(rows, "first-door").episode_id == 1
    assert pick(rows, "first-key").episode_id == 1
    assert pick(rows, "latest").episode_id == 5
    assert pick(rows, "at-step", at_step=210).episode_id == 4
    assert pick(rows, "at-step", at_step=110).episode_id == 2  # tie with 3 broken by lower episode_id


def test_pick_no_match_messages(rows):
    with pytest.raises(NoMatch, match="no successful episode"):
        pick(apply_filters(rows, Filters(phase="coverage_eval")), "first-success")
    with pytest.raises(NoMatch, match="no episodes"):
        pick([], "best-return")
    with pytest.raises(ValueError, match="at_step"):
        pick(rows, "at-step")
