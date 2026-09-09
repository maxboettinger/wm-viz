"""Quick looks at one episode without Blender: ASCII map and a matplotlib PNG."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .trace.reader import Episode

COLOR_RGB = {"red": (0.85, 0.2, 0.2), "green": (0.2, 0.7, 0.3), "blue": (0.25, 0.4, 0.9),
             "purple": (0.6, 0.3, 0.8), "yellow": (0.95, 0.8, 0.2), "grey": (0.5, 0.5, 0.5)}


def ascii_map(ep: Episode) -> str:
    lay = ep.layout
    visits = Counter((int(x), int(y)) for x, y in ep.agent_pos)
    start = (int(ep.agent_pos[0, 0]), int(ep.agent_pos[0, 1]))
    end = (int(ep.agent_pos[-1, 0]), int(ep.agent_pos[-1, 1]))
    rows = []
    for y in range(lay.height):
        line = []
        for x in range(lay.width):
            c = (x, y)
            if c in lay.walls:
                ch = "#"
            elif c in lay.doors:
                ch = "D"
            elif c in lay.keys:
                ch = "K"
            elif c in lay.goals:
                ch = "G"
            elif c in lay.lava:
                ch = "~"
            else:
                n = visits.get(c, 0)
                ch = "." if n == 0 else (str(n) if n < 10 else "*")
            if c == end:
                ch = "E"
            elif c == start:
                ch = "S"
            line.append(ch)
        rows.append("".join(line))
    return "\n".join(rows)


def save_png(ep: Episode, out: Path | str, cell_px: int = 32) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PNG previews: uv sync --extra figures") from e
    lay = ep.layout
    W, H = lay.width, lay.height
    fig, ax = plt.subplots(figsize=(W * cell_px / 100, H * cell_px / 100), dpi=100)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
    for (x, y) in lay.walls:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(0.35, 0.35, 0.35)))
    for (x, y), col in lay.doors.items():
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=COLOR_RGB[col]))
    for (x, y), col in lay.keys.items():
        ax.plot(x + 0.5, y + 0.5, marker="P", ms=cell_px * 0.35, color=COLOR_RGB[col], mec="k")
    for (x, y) in lay.goals:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(0.3, 0.85, 0.4)))
    for (x, y) in lay.lava:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=(0.95, 0.4, 0.1)))
    pts = ep.agent_pos.astype(float) + 0.5
    if len(pts) > 1:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidths=cell_px * 0.12)
        lc.set_array(np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)
    ax.plot(*pts[0], "o", color="white", mec="k", ms=cell_px * 0.3)
    ax.plot(*pts[-1], "s", color="black", mec="w", ms=cell_px * 0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out
