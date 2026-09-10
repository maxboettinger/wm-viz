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


def test_figure_requires_out(logs_dir):
    r = _inv(logs_dir, "figure", "p2e-doorkey6x6-s0/ep000003", "--backend", "mpl")
    assert r.exit_code == 2 and "--out" in r.stdout
