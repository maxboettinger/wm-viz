import re

from typer.testing import CliRunner

import wmviz.preview
from wmviz.cli import app

runner = CliRunner()


def _inv(logs_dir, *args):
    return runner.invoke(app, ["--logs", str(logs_dir), *args])


def test_runs(logs_dir):
    r = _inv(logs_dir, "runs")
    assert r.exit_code == 0 and "p2e-doorkey6x6-s0" in r.stdout
    assert re.search(r"│\s*6\s*│", r.stdout)


def test_runs_survives_a_degraded_run(logs_dir):
    """A run whose index.csv references an npz that isn't on disk must not
    take down the whole `runs` table — it gets '?' cells and a note."""
    broken = logs_dir / "broken" / "trace"
    broken.mkdir(parents=True)
    (broken / "index.csv").write_text((logs_dir / "p2e-doorkey6x6-s0" / "trace" / "index.csv").read_text())
    r = _inv(logs_dir, "runs")
    assert r.exit_code == 0
    assert "p2e-doorkey6x6-s0" in r.stdout and "broken" in r.stdout
    assert re.search(r"│\s*\?\s*│", r.stdout)
    assert "run broken:" in r.stdout


def test_list_filters_and_sort(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--phase", "eval", "--sort", "return")
    assert r.exit_code == 0
    assert "ep000003" in r.stdout and "ep000000" not in r.stdout
    assert r.stdout.index("ep000003") < r.stdout.index("ep000002")


def test_list_unknown_run(logs_dir):
    r = _inv(logs_dir, "list", "nope")
    assert r.exit_code == 1 and "trace/index.csv" in r.stdout


def test_list_full_ref_not_truncated_when_piped(logs_dir, monkeypatch):
    """Without the tests' COLUMNS override, Rich would fall back to an 80-col
    table when stdout isn't a tty (as under CliRunner); the CLI passes its own
    width so refs aren't cut off even then."""
    monkeypatch.delenv("COLUMNS", raising=False)
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0")
    assert r.exit_code == 0
    assert "ep000003" in r.stdout


def test_list_unknown_sort_key_exits_2(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--sort", "bogus")
    assert r.exit_code == 2 and "unknown sort key" in r.stdout


def test_list_layout_seed_bad_int_exits_2(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--layout", "seed:abc")
    assert r.exit_code == 2
    assert "--layout seed:<n> needs an integer, got 'abc'" in r.stdout


def test_pick_prints_ref_only(logs_dir):
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0", "--best-return")
    assert r.exit_code == 0 and r.stdout.strip() == "p2e-doorkey6x6-s0/ep000003"
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0", "--phase", "eval", "--at-step", "290")
    assert r.stdout.strip() == "p2e-doorkey6x6-s0/ep000005"
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0", "--phase", "coverage_eval", "--first-success")
    assert r.exit_code == 1 and "no successful episode" in r.stdout and "candidates:" in r.stdout


def test_pick_no_match_with_empty_filter_describes_run(logs_dir):
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0", "--phase", "dream", "--latest")
    assert r.exit_code == 1
    assert "no episodes match the filters" in r.stdout and "phases" in r.stdout


def test_pick_requires_exactly_one_selector(logs_dir):
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0")
    assert r.exit_code == 2 and "exactly one selector" in r.stdout
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0", "--latest", "--best-return")
    assert r.exit_code == 2


def test_show_ascii_and_png(logs_dir, tmp_path):
    r = _inv(logs_dir, "show", "p2e-doorkey6x6-s0/ep000003", "--png", str(tmp_path / "e.png"))
    assert r.exit_code == 0
    assert "#######" in r.stdout and "return" in r.stdout and "eval" in r.stdout
    assert (tmp_path / "e.png").exists()
    r = _inv(logs_dir, "show", "p2e-doorkey6x6-s0/42")
    assert r.exit_code == 1 and "no episode 42" in r.stdout


def test_show_png_without_matplotlib_fails_cleanly(logs_dir, tmp_path, monkeypatch):
    def _boom(ep, out, cell_px=32):
        raise RuntimeError("matplotlib is required for PNG previews: uv sync --extra figures")

    monkeypatch.setattr(wmviz.preview, "save_png", _boom)
    r = _inv(logs_dir, "show", "p2e-doorkey6x6-s0/ep000003", "--png", str(tmp_path / "e.png"))
    assert r.exit_code == 1 and "matplotlib is required" in r.stdout


def test_list_actor_filter(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--actor", "explorer")
    assert r.exit_code == 0
    assert "ep000000" in r.stdout and "ep000004" in r.stdout
    for absent in ("ep000001", "ep000002", "ep000003", "ep000005"):
        assert absent not in r.stdout


def test_list_step_range_filters(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--after-step", "150")
    assert r.exit_code == 0
    assert "ep000004" in r.stdout and "ep000005" in r.stdout
    for absent in ("ep000000", "ep000001", "ep000002", "ep000003"):
        assert absent not in r.stdout

    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--before-step", "50")
    assert r.exit_code == 0
    assert "ep000000" in r.stdout and "ep000001" in r.stdout
    for absent in ("ep000002", "ep000003", "ep000004", "ep000005"):
        assert absent not in r.stdout


def test_list_success_filter(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--success")
    assert r.exit_code == 0
    assert "ep000003" in r.stdout
    for absent in ("ep000000", "ep000001", "ep000002", "ep000004", "ep000005"):
        assert absent not in r.stdout

    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--no-success")
    assert r.exit_code == 0
    assert "ep000003" not in r.stdout
    for present in ("ep000000", "ep000001", "ep000002", "ep000004", "ep000005"):
        assert present in r.stdout


def test_list_min_cells_filter(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--min-cells", "9")
    assert r.exit_code == 0
    assert "ep000003" in r.stdout
    for absent in ("ep000000", "ep000001", "ep000002", "ep000004", "ep000005"):
        assert absent not in r.stdout


def test_list_layout_seed_filter(logs_dir):
    r = _inv(logs_dir, "list", "p2e-doorkey6x6-s0", "--layout", "seed:1000")
    assert r.exit_code == 0
    assert "ep000002" in r.stdout and "ep000005" in r.stdout
    for absent in ("ep000000", "ep000001", "ep000003", "ep000004"):
        assert absent not in r.stdout


def test_pick_actor_latest(logs_dir):
    r = _inv(logs_dir, "pick", "p2e-doorkey6x6-s0", "--actor", "explorer", "--latest")
    assert r.exit_code == 0 and r.stdout.strip() == "p2e-doorkey6x6-s0/ep000004"


def test_render_target_resolution_errors(logs_dir):
    r = _inv(logs_dir, "render", "p2e-doorkey6x6-s0", "--no-render")
    assert r.exit_code == 2 and "selector" in r.stdout
    r = _inv(logs_dir, "render", "p2e-doorkey6x6-s0", "--best-return", "--most-cells", "--no-render")
    assert r.exit_code == 2
    r = _inv(logs_dir, "render", "p2e-doorkey6x6-s0/ep000099", "--no-render")
    assert r.exit_code == 1 and "no episode 99" in r.stdout


def test_render_reports_missing_bpy_or_runs(logs_dir, tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "bpy" or name.startswith("wmviz.render") or name.startswith("wmviz.scene"):
            raise ImportError("No module named 'bpy'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = _inv(logs_dir, "render", "p2e-doorkey6x6-s0/ep000001", "--out", str(tmp_path / "x.mp4"))
    assert r.exit_code == 1 and "uv sync --extra blender" in r.stdout


def test_render_module_imports_without_bpy():
    """`wmviz.render` (RenderConfig, hud_lines, compose_frame, write_mp4) must import on a machine
    without Blender — `figure`'s matplotlib backend relies on it. Checked in a subprocess so the
    blocked `bpy` doesn't leak into this process's module cache."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['bpy'] = None; sys.modules['mathutils'] = None; "
            "import wmviz.cli, wmviz.render; print(wmviz.render.RenderConfig(out='x.mp4').resolution)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "(1920, 1080)"
