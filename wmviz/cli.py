"""wmviz — list, pick, preview and render recorded wm episodes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import NoReturn, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .trace import Episode, Index, IndexRow, find_runs, parse_ref
from .trace.selectors import SELECTORS, SORT_KEYS, Filters, NoMatch, apply_filters, pick as pick_row, sort_rows

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
console = Console(highlight=False, width=None if sys.stdout.isatty() else 200)


class State:
    logs: Path = Path("logs")


state = State()


@app.callback()
def _main(logs: Path = typer.Option(None, "--logs", help="Directory with logs/<run>/ (env WMVIZ_LOGS)")):
    state.logs = Path(logs or os.environ.get("WMVIZ_LOGS", "logs"))


def _fail(msg: str, code: int = 1) -> NoReturn:
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code)


def _index(run: str) -> Index:
    try:
        return Index.load(state.logs / run)
    except FileNotFoundError as e:
        _fail(str(e))


def _fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _filters(phase, actor, after_step, before_step, success, min_cells, layout) -> Filters:
    return Filters(phase=phase, actor=actor, after_step=after_step, before_step=before_step,
                   success=success, min_cells=min_cells, layout=layout)


# Shared filter options (Typer has no option groups; repeat explicitly).
_PHASE = typer.Option(None, "--phase")
_ACTOR = typer.Option(None, "--actor")
_AFTER = typer.Option(None, "--after-step")
_BEFORE = typer.Option(None, "--before-step")
_SUCCESS = typer.Option(None, "--success/--no-success")
_MIN_CELLS = typer.Option(None, "--min-cells")
_LAYOUT = typer.Option(None, "--layout", help="layout-hash prefix or seed:<n>")


@app.command()
def runs():
    """Runs under --logs that contain traces."""
    found = find_runs(state.logs) if state.logs.exists() else []
    if not found:
        _fail(f"no runs with trace/index.csv under {state.logs}")
    t = Table("run", "env", "exploration", "episodes", "last step")
    errors: list[tuple[str, Exception]] = []
    for p in found:
        try:
            idx = Index.load(p)
            meta = {}
            if idx.rows:
                with np.load(idx.path_of(idx.rows[-1])) as z:
                    meta = json.loads(str(z["meta"]))
            last = max((r.end_step for r in idx.rows), default=0)
            t.add_row(p.name, meta.get("env_id", ""), meta.get("exploration", ""), str(len(idx.rows)), str(last))
        except (OSError, ValueError, KeyError) as e:
            t.add_row(p.name, "?", "?", "?", "?")
            errors.append((p.name, e))
    console.print(t)
    for name, e in errors:
        console.print(f"[dim]run {name}: {e}[/dim]")


@app.command("list")
def list_cmd(run: str, phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR,
             after_step: Optional[int] = _AFTER, before_step: Optional[int] = _BEFORE,
             success: Optional[bool] = _SUCCESS, min_cells: Optional[int] = _MIN_CELLS,
             layout: Optional[str] = _LAYOUT,
             sort: str = typer.Option("episode_id", "--sort", help=", ".join(SORT_KEYS)),
             asc: bool = typer.Option(False, "--asc"), limit: int = typer.Option(50, "--limit")):
    """Episodes of RUN as a table."""
    idx = _index(run)
    try:
        rows = apply_filters(idx.rows, _filters(phase, actor, after_step, before_step, success, min_cells, layout))
        rows = sort_rows(rows, sort, descending=not asc)
    except ValueError as e:
        _fail(str(e), 2)
    t = Table("ref", "phase", "actor", "seed", "start", "len", "return", "ok", "cells", "cov%",
              "key@", "door@", "goal@", "rooms", "layout")
    for r in rows[:limit]:
        t.add_row(r.ref(idx.run_name), r.phase, r.actor, _fmt(r.seed), str(r.start_step), str(r.length),
                  _fmt(r.ret), "✓" if r.success else "", str(r.unique_cells),
                  "" if r.coverage_pct is None else f"{100 * r.coverage_pct:.0f}",
                  _fmt(r.first_key_step), _fmt(r.first_door_step), _fmt(r.first_goal_step),
                  _fmt(r.rooms_visited), r.layout_hash[:6])
    console.print(t)
    if len(rows) > limit:
        console.print(f"… {len(rows) - limit} more (use --limit)")


@app.command("pick")
def pick_cmd(run: str, phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR,
             after_step: Optional[int] = _AFTER, before_step: Optional[int] = _BEFORE,
             success: Optional[bool] = _SUCCESS, min_cells: Optional[int] = _MIN_CELLS,
             layout: Optional[str] = _LAYOUT,
             best_return: bool = typer.Option(False, "--best-return"),
             most_cells: bool = typer.Option(False, "--most-cells"),
             first_success: bool = typer.Option(False, "--first-success"),
             first_door: bool = typer.Option(False, "--first-door"),
             first_key: bool = typer.Option(False, "--first-key"),
             at_step: Optional[int] = typer.Option(None, "--at-step"),
             latest: bool = typer.Option(False, "--latest")):
    """Resolve one episode and print its reference (for $(...) in shells)."""
    selector = _chosen_selector(best_return, most_cells, first_success, first_door, first_key, at_step, latest)
    if selector is None:
        _fail(f"give exactly one selector: {_SELECTOR_FLAGS}", 2)
    idx = _index(run)
    row = _pick(idx, _filters(phase, actor, after_step, before_step, success, min_cells, layout), selector, at_step)
    print(row.ref(idx.run_name))


def _pick(idx: Index, filters: Filters, selector: str, at_step: Optional[int] = None) -> IndexRow:
    """Filter `idx.rows`, apply one selector. Bad filter/selector values exit 2; no match exits 1 with a hint."""
    try:
        rows = apply_filters(idx.rows, filters)
    except ValueError as e:
        _fail(str(e), 2)
    try:
        return pick_row(rows, selector, at_step=at_step)
    except ValueError as e:
        _fail(str(e), 2)
    except NoMatch as e:
        _fail(str(e) + _no_match_hint(idx, rows))


def _no_match_hint(idx: Index, rows: list[IndexRow]) -> str:
    """Extra context appended to a NoMatch message: what the run contains (no filtered
    rows), or the closest candidates (filtered rows exist but none satisfied the selector)."""
    if not rows:
        if not idx.rows:
            return ""
        phases = sorted({r.phase for r in idx.rows})
        actors = sorted({r.actor for r in idx.rows})
        steps = [r.start_step for r in idx.rows]
        seeds = sorted({r.seed for r in idx.rows if r.seed is not None})
        seed_str = ",".join(str(s) for s in seeds[:8]) + ("…" if len(seeds) > 8 else "")
        return (f" (run has {len(idx.rows)} episodes: phases {','.join(phases)}; "
                f"actors {','.join(actors)}; steps {min(steps)}..{max(steps)}; seeds {seed_str})")
    candidates = sorted(rows, key=lambda r: r.end_step)[:5]
    return "; candidates: " + ", ".join(r.ref(idx.run_name) for r in candidates)


def resolve_episode(logs: Path, ref: str) -> tuple[Index, IndexRow, Episode]:
    run, ep_id = parse_ref(ref)
    idx = Index.load(logs / run)
    row = idx.by_id(ep_id)
    return idx, row, Episode.load(idx.path_of(row))


_SELECTOR_FLAGS = "--best-return, --most-cells, --first-success, --first-door, --first-key, --at-step N, --latest"

_BEST = typer.Option(False, "--best-return")
_MOST = typer.Option(False, "--most-cells")
_FSUCC = typer.Option(False, "--first-success")
_FDOOR = typer.Option(False, "--first-door")
_FKEY = typer.Option(False, "--first-key")
_AT = typer.Option(None, "--at-step")
_LATEST = typer.Option(False, "--latest")


def _chosen_selector(best_return, most_cells, first_success, first_door, first_key, at_step, latest) -> str | None:
    """The one selector flag that is set, by its `SELECTORS` name (`at_step` counts when not None); None if
    no selector was given. More than one exits 2 — callers decide what "none" means for them."""
    chosen = [name for name, on in zip(SELECTORS, (best_return, most_cells, first_success, first_door, first_key,
                                                    at_step is not None, latest)) if on]
    if len(chosen) > 1:
        _fail(f"give exactly one selector, not {len(chosen)} ({', '.join('--' + c for c in chosen)}): {_SELECTOR_FLAGS}", 2)
    return chosen[0] if chosen else None


def _target(target: str, filters: Filters, best_return, most_cells, first_success, first_door, first_key,
            at_step, latest) -> tuple[Index, IndexRow, Episode]:
    """`<run>/ep000123` → that episode; `<run>` + exactly one selector → picked episode."""
    selector = _chosen_selector(best_return, most_cells, first_success, first_door, first_key, at_step, latest)
    if "/" in target:
        if selector is not None:
            _fail("give either <run>/ep000123 or <run> plus one selector, not both", 2)
        try:
            return resolve_episode(state.logs, target)
        except (ValueError, FileNotFoundError, KeyError) as e:
            _fail(e.args[0] if isinstance(e, KeyError) and e.args else str(e))
    if selector is None:
        _fail(f"TARGET is a run name: give exactly one selector ({_SELECTOR_FLAGS}) or pass <run>/ep000123", 2)
    idx = _index(target)
    row = _pick(idx, filters, selector, at_step)
    return idx, row, Episode.load(idx.path_of(row))


def _parse_res(s: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in s.lower().split("x"))
    except ValueError:
        _fail(f"--res must look like 1920x1080, got {s!r}", 2)
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        _fail(f"--res width and height must be positive and even (libx264 needs even dimensions), got {s!r}", 2)
    return w, h


def _default_out(idx: Index, row: IndexRow, suffix: str) -> Path:
    return Path("renders") / idx.run_name / f"ep{row.episode_id:06d}{suffix}"


def _render_options(idx, row, out, *, engine, samples, res, preview, camera, assets, fps=24, frames_per_step=6,
                    discrete=False, trail=False, heatmap=False, hud=False, still=None, save_blend=None,
                    no_render=False, dream_layout="pip") -> dict:
    """Validated `RenderConfig` kwargs (bad values exit 2); the defaults are the still-render shape shared by
    `figure`, `_figure_images` and `timeline --video`. Imports nothing that needs bpy, so argument errors
    are reported independently of the Blender check."""
    cams = tuple(c.strip() for c in camera.split(",") if c.strip())
    if not cams:
        _fail("--camera needs at least one preset (topdown, follow, fpv, orbit, iso)", 2)
    if no_render and save_blend is None:
        _fail("--no-render needs --save-blend (the GUI path writes a .blend and stops)", 2)
    return dict(out=out or _default_out(idx, row, ".png" if still is not None else ".mp4"),
                engine=engine, samples=samples, res=_parse_res(res), fps=fps,
                frames_per_step=frames_per_step, discrete=discrete, preview=preview, cameras=cams,
                trail=trail, heatmap=heatmap, hud=hud, still=still, save_blend=save_blend,
                no_render=no_render, assets=assets, dream_layout=dream_layout)


def _need_bpy():
    try:
        import bpy  # noqa: F401
    except ImportError:
        _fail("Blender rendering needs bpy: run `uv sync --extra blender` (Python 3.11 only)")


@app.command()
def render(target: str,
           phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR, after_step: Optional[int] = _AFTER,
           before_step: Optional[int] = _BEFORE, success: Optional[bool] = _SUCCESS,
           min_cells: Optional[int] = _MIN_CELLS, layout: Optional[str] = _LAYOUT,
           best_return: bool = _BEST, most_cells: bool = _MOST, first_success: bool = _FSUCC,
           first_door: bool = _FDOOR, first_key: bool = _FKEY, at_step: Optional[int] = _AT, latest: bool = _LATEST,
           out: Optional[Path] = typer.Option(None, "--out", help="mp4 (or png with --still); default renders/<run>/<ep>"),
           camera: Optional[str] = typer.Option(None, "--camera", help="topdown,follow,fpv,orbit,iso (comma = side by side); "
                                                                        "default topdown, fpv for dream episodes"),
           engine: str = typer.Option("eevee", "--engine", help="eevee | cycles"),
           samples: int = typer.Option(64, "--samples"),
           res: str = typer.Option("1920x1080", "--res"),
           fps: int = typer.Option(24, "--fps"),
           frames_per_step: int = typer.Option(6, "--frames-per-step"),
           discrete: bool = typer.Option(False, "--discrete", help="no interpolation between steps"),
           preview: bool = typer.Option(False, "--preview", help="960x540, 3 frames per step"),
           trail: bool = typer.Option(False, "--trail"), heatmap: bool = typer.Option(False, "--heatmap"),
           hud: bool = typer.Option(False, "--hud"),
           still: Optional[int] = typer.Option(None, "--still", help="render one frame at this step"),
           save_blend: Optional[Path] = typer.Option(None, "--save-blend"),
           no_render: bool = typer.Option(False, "--no-render", help="stop after --save-blend (GUI path)"),
           assets: Optional[Path] = typer.Option(None, "--assets", help="asset library .blend"),
           dream_layout: str = typer.Option("pip", "--dream-layout", help="pip | split (dream episodes)")):
    """Render one episode as a Blender animation (or a still) — TARGET is <run>/ep000123 or <run> + one selector."""
    _need_bpy()
    idx, row, ep = _target(target, _filters(phase, actor, after_step, before_step, success, min_cells, layout),
                           best_return, most_cells, first_success, first_door, first_key, at_step, latest)
    camera = camera or ("fpv" if ep.dream is not None else "topdown")
    opts = _render_options(idx, row, out, engine=engine, samples=samples, res=res, preview=preview, camera=camera,
                           assets=assets, fps=fps, frames_per_step=frames_per_step, discrete=discrete, trail=trail,
                           heatmap=heatmap, hud=hud, still=still, save_blend=save_blend, no_render=no_render,
                           dream_layout=dream_layout)
    from .render import RenderConfig, render_episode
    try:
        result = render_episode(ep, row, RenderConfig(**opts))
    except ValueError as e:
        _fail(str(e), 2)
    console.print(f"wrote {result}")


def _backend(name: str | None) -> str:
    if name in ("blender", "mpl"):
        return name
    if name is not None:
        _fail("--backend must be blender or mpl", 2)
    try:
        import bpy  # noqa: F401
        return "blender"
    except ImportError:
        console.print("[dim]bpy not available — using --backend mpl[/dim]")
        return "mpl"


def _write_image(img, out: Path) -> None:
    import imageio.v3 as iio
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, img)


@app.command()
def figure(target: str,
           phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR, after_step: Optional[int] = _AFTER,
           before_step: Optional[int] = _BEFORE, success: Optional[bool] = _SUCCESS,
           min_cells: Optional[int] = _MIN_CELLS, layout: Optional[str] = _LAYOUT,
           best_return: bool = _BEST, most_cells: bool = _MOST, first_success: bool = _FSUCC,
           first_door: bool = _FDOOR, first_key: bool = _FKEY, at_step: Optional[int] = _AT, latest: bool = _LATEST,
           out: Optional[Path] = typer.Option(None, "--out", help="PNG path (required)"),
           backend: Optional[str] = typer.Option(None, "--backend", help="blender | mpl (default: blender if bpy imports)"),
           camera: str = typer.Option("topdown", "--camera", help="topdown | iso (blender backend)"),
           keyframes: Optional[str] = typer.Option(None, "--keyframes", help="auto | 3,57,120 → labelled strip"),
           trail: bool = typer.Option(True, "--trail/--no-trail"),
           heatmap: bool = typer.Option(True, "--heatmap/--no-heatmap"),
           preview: bool = typer.Option(False, "--preview"),
           engine: str = typer.Option("eevee", "--engine"), samples: int = typer.Option(64, "--samples"),
           res: str = typer.Option("1920x1080", "--res"),
           assets: Optional[Path] = typer.Option(None, "--assets")):
    """A still of one episode (top-down by default) with trail and heatmap; --keyframes makes a labelled strip."""
    if out is None:
        _fail("figure needs --out <path.png>", 2)
    out = out.with_suffix(".png")
    be = _backend(backend)
    if be == "blender":
        _need_bpy()
    idx, row, ep = _target(target, _filters(phase, actor, after_step, before_step, success, min_cells, layout),
                           best_return, most_cells, first_success, first_door, first_key, at_step, latest)
    from .mpl import figure_image, keyframe_steps
    try:      # any explicit --keyframes value is a spec, even "" (which fails like --milestones ",")
        steps = keyframe_steps(row, ep, keyframes) if keyframes is not None else [(ep.length, "")]
    except ValueError as e:
        _fail(str(e), 2)
    labels = [l for _, l in steps]
    if keyframes is not None:
        console.print("keyframes: " + ", ".join(labels))
    if be == "mpl":
        images = [figure_image(ep, step=s, trail=trail, heatmap=heatmap) for s, _ in steps]
    else:
        from .render import RenderConfig, render_still
        cfg = RenderConfig(**_render_options(idx, row, out, engine=engine, samples=samples, res=res, preview=preview,
                                             camera=camera, assets=assets, trail=trail, heatmap=heatmap))
        images = [render_still(ep, row, cfg, s) for s, _ in steps]
    from .compose import strip
    img = images[0] if len(images) == 1 and keyframes is None else strip(images, labels)
    _write_image(img, out)
    console.print(f"wrote {out}")


def _same_layout_or_fail(rows, force: bool) -> str:
    from .aggregate import LayoutMismatch, same_layout
    try:
        return same_layout(rows, force=force)
    except (LayoutMismatch, ValueError) as e:
        _fail(str(e))


def _milestones(s: str) -> list[int]:
    try:
        out = [int(v) for v in s.split(",") if v.strip()]
    except ValueError:
        _fail(f"--milestones must be a comma list of steps, got {s!r}", 2)
    if not out:
        _fail(f"--milestones must be a comma list of steps, got {s!r}", 2)
    return out


def _figure_images(idx, rows, backend: str, camera, preview, engine, samples, res, assets, out: Path):
    """One still image per row; the blender backend gives each episode its own `<out stem>_<id>_frames/`
    (derived from the real `--out`) so distinct episodes never share a frame cache — sharing one
    fixed placeholder path made every episode's frames indistinguishable and collide across runs."""
    from .mpl import figure_image
    images = []
    for row in rows:
        ep = Episode.load(idx.path_of(row))
        if backend == "mpl":
            images.append(figure_image(ep))
        else:
            from .render import RenderConfig, render_still
            ep_out = out.parent / f"{out.stem}_{row.episode_id:06d}.png"
            opts = _render_options(idx, row, ep_out, engine=engine, samples=samples, res=res, preview=preview,
                                   camera=camera, assets=assets, trail=True, heatmap=True)
            images.append(render_still(ep, row, RenderConfig(**opts), ep.length))
    return images


def _hold_last(it, n):
    """Yield `it`'s items, then repeat its last item until `n` total have been yielded — freezes a
    shorter animation on its last frame instead of ending early, for `--video`'s synced tiling. Callers
    pass `n = max(len(...) for ... in ...)` across the tiled sequences, so `n` never truncates `it`."""
    last = None
    count = 0
    for item in it:
        last = item
        count += 1
        yield item
    for _ in range(n - count):
        yield last


@app.command()
def timeline(run: str,
             seed: int = typer.Option(..., "--seed", help="eval/coverage-eval seed (fixed layout)"),
             milestones: str = typer.Option(..., "--milestones", help="comma list of env steps"),
             out: Path = typer.Option(..., "--out", help="png (or mp4 with --video)"),
             phase: str = typer.Option("coverage_eval", "--phase"),
             backend: Optional[str] = typer.Option(None, "--backend"),
             video: bool = typer.Option(False, "--video", help="tile the animations in sync instead of stills"),
             camera: str = typer.Option("topdown", "--camera"),
             preview: bool = typer.Option(False, "--preview"), force: bool = typer.Option(False, "--force"),
             engine: str = typer.Option("eevee", "--engine"), samples: int = typer.Option(64, "--samples"),
             res: str = typer.Option("1920x1080", "--res"), fps: int = typer.Option(24, "--fps"),
             assets: Optional[Path] = typer.Option(None, "--assets")):
    """Exploration over training: the seed-S episode nearest each milestone, as a labelled strip (or tiled video)."""
    from .aggregate import nearest_to_milestones
    idx = _index(run)
    rows = apply_filters(idx.rows, Filters(phase=phase, layout=f"seed:{seed}"))
    if not rows:
        _fail(f"{run} has no {phase} episodes with seed {seed}" + _no_match_hint(idx, rows))
    picked = nearest_to_milestones(rows, _milestones(milestones))
    _same_layout_or_fail(picked, force)
    labels = [f"step {r.start_step}" for r in picked]
    console.print("episodes: " + ", ".join(f"{r.ref(idx.run_name)} ({l})" for r, l in zip(picked, labels)))
    be = _backend(backend)
    if video and be == "mpl":
        _fail("--video needs the Blender backend (uv sync --extra blender)")
    if be == "blender":
        _need_bpy()
    from .compose import grid, label, strip
    if not video:
        out = out.with_suffix(".png")
        images = _figure_images(idx, picked, be, camera, preview, engine, samples, res, assets, out)
        _write_image(strip(images, labels), out)
    else:
        from .render import RenderConfig, frames_composited, write_mp4
        seqs = []
        for row in picked:
            ep = Episode.load(idx.path_of(row))
            ep_out = out.parent / f"{out.stem}_{row.episode_id:06d}.mp4"
            opts = _render_options(idx, row, ep_out, engine=engine, samples=samples, res=res, preview=preview,
                                   camera=camera, assets=assets, fps=fps, trail=True, heatmap=True)
            # each frames_composited() call renders its episode fully to disk before the next
            # build() resets the scene for the following episode — sequential by construction.
            seqs.append(frames_composited(ep, row, RenderConfig(**opts)))
        n = max(count for count, _ in seqs)
        held = [_hold_last(it, n) for _, it in seqs]
        cols = min(len(seqs), 3)
        frames = (grid([label(next(h), l) for h, l in zip(held, labels)], cols=cols) for _ in range(n))
        out = out.with_suffix(".mp4")
        write_mp4(frames, out, fps)
    console.print(f"wrote {out}")


def _heatmap_rows(run: str, layout, until_step, phase, actor, force) -> tuple[Index, list[IndexRow]]:
    """Episodes of `run` that go into one heatmap (all on one layout unless `force`)."""
    idx = _index(run)
    rows = apply_filters(idx.rows, Filters(phase=phase, actor=actor, layout=layout))
    if until_step is not None:
        rows = [r for r in rows if r.end_step <= until_step]
    if not rows:
        _fail(f"{run}: no episodes match" + (f" --layout {layout}" if layout else "") + _no_match_hint(idx, rows))
    _same_layout_or_fail(rows, force)
    return idx, rows


def _heatmap_image(idx, rows, backend: str, animate: bool, out: Path, preview, engine, samples, res, fps, assets):
    """One heatmap image of `rows` — or, with `animate` (blender only), `(frame count, lazy frames)`."""
    from .aggregate import accumulate
    eps = [(r, Episode.load(idx.path_of(r))) for r in rows]
    lay = eps[0][1].layout
    counts = accumulate([e for _, e in eps], lay.width, lay.height)
    if backend == "mpl":
        assert not animate, "the caller rejects --animate on the mpl backend"
        from .mpl import heatmap_image
        return heatmap_image(lay, counts)
    from .render import RenderConfig, render_heatmap_frames, render_heatmap_still
    cfg = RenderConfig(out=out, engine=engine, samples=samples, res=_parse_res(res), fps=fps, preview=preview,
                       hud=True, assets=assets, extra={"n_episodes": len(eps)})
    if animate:
        return render_heatmap_frames(lay, eps, cfg)
    return render_heatmap_still(lay, counts, cfg)


@app.command()
def heatmap(run: str,
            out: Path = typer.Option(..., "--out", help="png (mp4 with --animate)"),
            layout: Optional[str] = _LAYOUT,
            until_step: Optional[int] = typer.Option(None, "--until-step", help="only episodes that ended by this step"),
            phase: Optional[str] = _PHASE, actor: Optional[str] = _ACTOR,
            animate: bool = typer.Option(False, "--animate", help="floor fills in episode by episode (mp4)"),
            compare: Optional[str] = typer.Option(None, "--compare", help="second run, same layout, side by side"),
            backend: Optional[str] = typer.Option(None, "--backend"),
            preview: bool = typer.Option(False, "--preview"), force: bool = typer.Option(False, "--force"),
            engine: str = typer.Option("eevee", "--engine"), samples: int = typer.Option(64, "--samples"),
            res: str = typer.Option("1920x1080", "--res"), fps: int = typer.Option(6, "--fps"),
            assets: Optional[Path] = typer.Option(None, "--assets")):
    """Visit counts accumulated over the selected episodes of one layout, as a Blender (or matplotlib) still;
    --animate fills the floor in episode by episode (mp4), --compare RUN2 puts a second run side by side."""
    be = _backend(backend)
    if animate and be == "mpl":
        _fail("--animate needs the Blender backend (uv sync --extra blender)")
    if be == "blender":
        _need_bpy()
    runs = [run] + ([compare] if compare else [])
    selected = []
    for name in runs:
        idx, rows = _heatmap_rows(name, layout, until_step, phase, actor, force)
        console.print(f"{name}: {len(rows)} episodes on layout {rows[0].layout_hash[:8]}")
        selected.append((idx, rows))
    if compare:
        _same_layout_or_fail([r for _, rows in selected for r in rows], force)
    from .compose import hstack, label
    # a compared run renders into its own `<out stem>_<run>` frames cache, so the two never share PNGs;
    # with --animate each call renders its run fully to disk before the next one resets the scene.
    results = [_heatmap_image(idx, rows, be, animate, out if len(runs) == 1 else out.with_stem(f"{out.stem}_{name}"),
                              preview, engine, samples, res, fps, assets)
               for name, (idx, rows) in zip(runs, selected)]

    def tiled(images):
        return images[0] if not compare else hstack([label(im, name) for im, name in zip(images, runs)], gap=8)

    if animate:
        from .render import write_mp4
        n = max(count for count, _ in results)
        held = [_hold_last(it, n) for _, it in results]
        out = write_mp4((tiled([next(h) for h in held]) for _ in range(n)), out.with_suffix(".mp4"), fps)
    else:
        out = out.with_suffix(".png")
        _write_image(tiled(results), out)
    console.print(f"wrote {out}")


@app.command()
def show(ref: str, png: Optional[Path] = typer.Option(None, "--png", help="also write a PNG preview")):
    """Stats and an ASCII top-down map of one episode (<run>/ep000123)."""
    from .preview import ascii_map, save_png
    try:
        idx, row, ep = resolve_episode(state.logs, ref)
    except (ValueError, FileNotFoundError, KeyError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        _fail(msg)
    m = ep.meta
    console.print(f"[bold]{row.ref(idx.run_name)}[/bold]  {m.get('env_id')}  phase={row.phase} actor={row.actor}"
                  f" seed={_fmt(row.seed)}  steps {row.start_step}→{row.end_step}")
    console.print(f"length={row.length} return={_fmt(row.ret)} success={row.success} "
                  f"cells={row.unique_cells} coverage={'' if row.coverage_pct is None else f'{100 * row.coverage_pct:.0f}%'} "
                  f"key@{_fmt(row.first_key_step)} door@{_fmt(row.first_door_step)} goal@{_fmt(row.first_goal_step)} "
                  f"rooms={_fmt(row.rooms_visited)} layout={row.layout_hash}")
    console.print(ascii_map(ep))
    console.print("[dim]# wall  D door  K key  G goal  S start  E end  digits = visits[/dim]")
    if png is not None:
        try:
            out = save_png(ep, png)
        except RuntimeError as e:
            _fail(str(e))
        console.print(f"wrote {out}")
