"""Quick looks at one episode without Blender: ASCII map and a matplotlib PNG."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .grid import COLOR_RGB  # noqa: F401  (re-exported)
from .mpl import _plt, draw_agent, draw_layout, draw_trail, new_axes
from .trace.reader import Episode


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
    fig, ax = new_axes(ep.layout, cell_px)
    draw_layout(ax, ep.layout)
    draw_trail(ax, ep.agent_pos, lw=cell_px * 0.12)
    draw_agent(ax, ep.agent_pos[-1], int(ep.agent_dir[-1]))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    _plt().close(fig)
    return out
