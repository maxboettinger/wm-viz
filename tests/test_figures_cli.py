"""figure/timeline/heatmap commands with --backend mpl (no bpy)."""
import pytest
from typer.testing import CliRunner

pytest.importorskip("matplotlib")

from wmviz.cli import app  # noqa: E402

runner = CliRunner()


def _inv(logs_dir, *args):
    return runner.invoke(app, ["--logs", str(logs_dir), *args])


def test_figure_mpl_single_and_keyframes(logs_dir, tmp_path):
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0/ep000003", "--backend", "mpl", "--out", str(tmp_path / "f.png"))
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "f.png").exists()
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0", "--first-success", "--backend", "mpl",
             "--keyframes", "auto", "--out", str(tmp_path / "strip.png"))
    assert r.exit_code == 0, r.stdout
    import imageio.v3 as iio
    single, strip = iio.imread(tmp_path / "f.png"), iio.imread(tmp_path / "strip.png")
    assert strip.shape[1] > single.shape[1]

    # manual keyframe labels must not be double-wrapped ("step 3 · step 3") — keyframe_steps'
    # own caption is passed through untouched.
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0/ep000003", "--backend", "mpl",
             "--keyframes", "0,3", "--out", str(tmp_path / "manual.png"))
    assert r.exit_code == 0, r.stdout
    assert "step 3" in r.stdout and "· step 3" not in r.stdout


def test_figure_empty_keyframes_is_a_bad_spec(logs_dir, tmp_path):
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0/ep000003", "--backend", "mpl", "--keyframes", "",
             "--out", str(tmp_path / "f.png"))
    assert r.exit_code == 2 and "--keyframes" in r.stdout


def test_figure_requires_out(logs_dir):
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0/ep000003", "--backend", "mpl")
    assert r.exit_code == 2 and "--out" in r.stdout


def test_figure_blender_reports_missing_bpy(logs_dir, tmp_path, monkeypatch):
    """--backend blender is resolved (and bpy checked) before the episode is even loaded, same
    contract as `render` — no ImportError traceback from inside render_still."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "bpy" or name.startswith("wmviz.render") or name.startswith("wmviz.scene"):
            raise ImportError("No module named 'bpy'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0/ep000003", "--backend", "blender", "--out", str(tmp_path / "x.png"))
    assert r.exit_code == 1 and "uv sync --extra blender" in r.stdout


def test_timeline_mpl_strip(logs_dir, tmp_path):
    r = _inv(logs_dir, "timeline", "p2e-doorkey6x6-s0", "--seed", "1000", "--milestones", "0,250,1000",
             "--phase", "eval", "--backend", "mpl", "--out", str(tmp_path / "t.png"))
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "t.png").exists() and "step 100" in r.stdout and "step 300" in r.stdout

    # the strip is one labelled panel per milestone: three milestones must be wider than one.
    r1 = _inv(logs_dir, "timeline", "p2e-doorkey6x6-s0", "--seed", "1000", "--milestones", "0",
              "--phase", "eval", "--backend", "mpl", "--out", str(tmp_path / "t1.png"))
    assert r1.exit_code == 0, r1.stdout
    import imageio.v3 as iio
    one, three = iio.imread(tmp_path / "t1.png"), iio.imread(tmp_path / "t.png")
    assert three.shape[1] > one.shape[1]


def test_timeline_milestones_rejects_empty_list(logs_dir, tmp_path):
    r = _inv(logs_dir, "timeline", "p2e-doorkey6x6-s0", "--seed", "1000", "--milestones", ",",
             "--phase", "eval", "--backend", "mpl", "--out", str(tmp_path / "t.png"))
    assert r.exit_code == 2 and "--milestones" in r.stdout


def test_timeline_refuses_mixed_layouts_unless_forced(logs_dir, tmp_path, monkeypatch):
    import wmviz.cli as cli
    orig = cli._same_layout_or_fail
    monkeypatch.setattr(cli, "_same_layout_or_fail", lambda rows, force: orig(
        rows + [type(rows[0])(**{**rows[0].__dict__, "layout_hash": "ffffffffffff"})], force))
    r = _inv(logs_dir, "timeline", "p2e-doorkey6x6-s0", "--seed", "1000", "--milestones", "0",
             "--phase", "eval", "--backend", "mpl", "--out", str(tmp_path / "t.png"))
    assert r.exit_code == 1 and "--force" in r.stdout
    r = _inv(logs_dir, "timeline", "p2e-doorkey6x6-s0", "--seed", "1000", "--milestones", "0",
             "--phase", "eval", "--backend", "mpl", "--force", "--out", str(tmp_path / "t.png"))
    assert r.exit_code == 0, r.stdout


def test_timeline_needs_seed_milestones_and_out(logs_dir):
    r = _inv(logs_dir, "timeline", "p2e-doorkey6x6-s0", "--milestones", "0")
    assert r.exit_code == 2


def test_hold_last_freezes_on_final_item():
    """`_hold_last` backs --video's synced tiling: a shorter episode's sequence repeats its last
    frame instead of running out early, so `grid()` always gets one frame per episode per tick."""
    from wmviz.cli import _hold_last
    assert list(_hold_last(iter([1, 2, 3]), 5)) == [1, 2, 3, 3, 3]
    assert list(_hold_last(iter([1, 2, 3]), 3)) == [1, 2, 3]
    assert list(_hold_last(iter(["only"]), 4)) == ["only"] * 4


def test_heatmap_mpl_and_compare(logs_dir, tmp_path):
    r = _inv(logs_dir, "heatmap", "p2e-doorkey6x6-s0", "--layout", "seed:1000", "--backend", "mpl",
             "--out", str(tmp_path / "h.png"))
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "h.png").exists() and "2 episodes" in r.stdout
    r = _inv(logs_dir, "heatmap", "p2e-doorkey6x6-s0", "--layout", "seed:1000", "--until-step", "150",
             "--backend", "mpl", "--out", str(tmp_path / "h2.png"))
    assert r.exit_code == 0 and "1 episodes" in r.stdout
    r = _inv(logs_dir, "heatmap", "p2e-doorkey6x6-s0", "--layout", "seed:1000", "--compare", "p2e-doorkey6x6-s0",
             "--backend", "mpl", "--out", str(tmp_path / "cmp.png"))
    assert r.exit_code == 0, r.stdout
    import imageio.v3 as iio
    assert iio.imread(tmp_path / "cmp.png").shape[1] > 2 * iio.imread(tmp_path / "h.png").shape[1] * 0.9


def test_heatmap_refuses_mixed_layouts(logs_dir, tmp_path):
    r = _inv(logs_dir, "heatmap", "p2e-doorkey6x6-s0", "--backend", "mpl", "--out", str(tmp_path / "h.png"))
    # the synthetic run has one layout hash, so it succeeds; --layout with a bogus hash must fail clearly
    assert r.exit_code == 0
    r = _inv(logs_dir, "heatmap", "p2e-doorkey6x6-s0", "--layout", "zzzzzz", "--backend", "mpl", "--out", str(tmp_path / "h.png"))
    assert r.exit_code == 1 and "no episodes" in r.stdout.lower()


def test_heatmap_animate_needs_blender(logs_dir, tmp_path):
    r = _inv(logs_dir, "heatmap", "p2e-doorkey6x6-s0", "--layout", "seed:1000", "--backend", "mpl", "--animate",
             "--out", str(tmp_path / "h.mp4"))
    assert r.exit_code == 1 and "blender" in r.stdout.lower()
