"""matplotlib backend for stills (spec §6 `--backend mpl`): layout, trail, heatmap, agent."""
from __future__ import annotations

import io

import numpy as np

from .aggregate import visit_counts
from .grid import COLOR_RGB, DIR_VEC
from .trace.reader import Episode, IndexRow, Layout


def _plt():
    try:
        import matplotlib
        if matplotlib.get_backend().lower() != "agg":     # don't hijack an interactive backend (e.g. in a notebook)
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PNG figures: uv sync --extra figures") from e


def new_axes(layout: Layout, cell_px: int = 32):
    plt = _plt()
    W, H = layout.width, layout.height
    fig, ax = plt.subplots(figsize=(W * cell_px / 100, H * cell_px / 100), dpi=100)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def draw_layout(ax, layout: Layout) -> None:
    plt = _plt()
    for (x, y) in layout.walls:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(0.35, 0.35, 0.35)))
    for (x, y), col in layout.doors.items():
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=COLOR_RGB[col]))
    for (x, y), col in layout.keys.items():
        ax.plot(x + 0.5, y + 0.5, marker="P", ms=12, color=COLOR_RGB[col], mec="k")
    for (x, y) in layout.goals:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(0.3, 0.85, 0.4)))
    for (x, y) in layout.lava:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(0.95, 0.4, 0.1)))


def draw_heatmap(ax, layout: Layout, counts: np.ndarray, alpha: float = 0.8) -> None:
    plt = _plt()
    cmap = plt.get_cmap("magma")
    peak = max(int(np.asarray(counts).max()), 1)
    for x in range(layout.width):
        for y in range(layout.height):
            c = int(counts[x, y])
            if c > 0 and (x, y) not in layout.walls:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap(0.15 + 0.85 * c / peak), alpha=alpha))


def draw_trail(ax, pos: np.ndarray, upto: int | None = None, lw: float = 3.0) -> None:
    from matplotlib.collections import LineCollection
    pts = np.asarray(pos, dtype=float)[: None if upto is None else upto + 1] + 0.5
    if len(pts) > 1:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidths=lw)
        lc.set_array(np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)
    ax.plot(*pts[0], "o", color="white", mec="k", ms=8)


def draw_agent(ax, pos, direction: int) -> None:
    x, y = float(pos[0]) + 0.5, float(pos[1]) + 0.5
    dx, dy = DIR_VEC[int(direction)]
    ax.plot(x, y, "s", color="black", mec="w", ms=9)
    ax.annotate("", xy=(x + 0.4 * dx, y + 0.4 * dy), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="w", lw=2))


def fig_to_array(fig) -> np.ndarray:
    import imageio.v3 as iio
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    _plt().close(fig)
    buf.seek(0)
    return np.asarray(iio.imread(buf))[:, :, :3]


def figure_image(ep: Episode, step: int | None = None, trail: bool = True, heatmap: bool = True, cell_px: int = 32) -> np.ndarray:
    t = ep.length if step is None else min(max(int(step), 0), ep.length)
    fig, ax = new_axes(ep.layout, cell_px)
    if heatmap:
        draw_heatmap(ax, ep.layout, visit_counts(ep.agent_pos[: t + 1], ep.layout.width, ep.layout.height))
    draw_layout(ax, ep.layout)
    if trail:
        draw_trail(ax, ep.agent_pos, upto=t)
    draw_agent(ax, ep.agent_pos[t], int(ep.agent_dir[t]))
    return fig_to_array(fig)


def heatmap_image(layout: Layout, counts: np.ndarray, cell_px: int = 32) -> np.ndarray:
    fig, ax = new_axes(layout, cell_px)
    draw_heatmap(ax, layout, counts, alpha=0.95)
    draw_layout(ax, layout)
    return fig_to_array(fig)


def keyframe_steps(row: IndexRow, ep: Episode, spec: str) -> list[tuple[int, str]]:
    """Returns `(step, caption)` pairs — the caption is final; callers pass it through untouched."""
    if spec == "auto":
        seen: dict[int, str] = {}
        for s, name in ((row.first_key_step, "key"), (row.first_door_step, "door"), (row.first_goal_step, "goal")):
            if s is not None:
                seen[int(s)] = name    # later (more-advanced) landmarks win a step shared with an earlier one
        seen.setdefault(ep.length, "end")          # a landmark on the last step keeps its own label
        return [(s, f"{name} · step {s}") for s, name in sorted(seen.items())]
    try:
        steps = [int(s) for s in spec.split(",") if s.strip()]
    except ValueError:
        raise ValueError(f"--keyframes must be 'auto' or a comma list of steps, got {spec!r}") from None
    if not steps:
        raise ValueError(f"--keyframes must be 'auto' or a comma list of steps, got {spec!r}")
    clamped = [min(max(s, 0), ep.length) for s in steps]
    return [(c, f"step {c}") for c in clamped]
