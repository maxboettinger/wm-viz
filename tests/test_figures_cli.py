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
