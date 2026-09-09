import pytest

from wmviz.preview import ascii_map, save_png
from wmviz.trace import Episode, Index


def _ep(run_dir, i):
    idx = Index.load(run_dir)
    return Episode.load(idx.path_of(idx.by_id(i)))


def test_ascii_map_shape_and_symbols(run_dir):
    ep = _ep(run_dir, 3)                       # success, T=8 → ends on the goal (5,5)
    txt = ascii_map(ep)
    lines = txt.splitlines()
    assert len(lines) == 7 and all(len(l) == 7 for l in lines)
    assert lines[0] == "#######" and lines[6] == "#######"
    assert lines[3][3] == "D"                  # row y=3, col x=3
    assert lines[1][1] == "S"
    assert lines[5][5] == "E"
    assert "K" in txt
    assert "G" not in txt                      # goal cell overwritten by the end marker


def test_ascii_map_visit_counts(run_dir):
    ep = _ep(run_dir, 0)                       # never moves past (5,3), no success
    txt = ascii_map(ep)
    assert txt.splitlines()[5][5] == "G"
    assert any(ch.isdigit() for ch in txt)


def test_save_png(run_dir, tmp_path):
    pytest.importorskip("matplotlib")
    out = save_png(_ep(run_dir, 3), tmp_path / "ep.png")
    assert out.exists() and out.stat().st_size > 1000
