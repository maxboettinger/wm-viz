"""Render pipeline (spec §4): load → build_scene → animate → cameras → overlays → frames → post → mp4.
One episode per call; the scene is factory-reset first. Existing frames are skipped so an
interrupted render resumes — but only frames of the same settings: `<out stem>_frames/render.json`
holds the RenderConfig fingerprint and a mismatching frames directory is wiped first."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as iio
import imageio.v3 as iio3
import numpy as np

from .compose import hstack, hud, pip, split
from .trace.reader import Episode, IndexRow

# `.animate` (and everything under `.scene`) imports bpy at module level; import it lazily so
# RenderConfig, hud_lines, compose_frame and write_mp4 stay usable without Blender (mpl backend).

PREVIEW_RES = (960, 540)
PREVIEW_FPS_PER_STEP = 3


@dataclass
class RenderConfig:
    out: Path
    engine: str = "eevee"
    samples: int = 64
    res: tuple[int, int] = (1920, 1080)
    fps: int = 24
    frames_per_step: int = 6
    discrete: bool = False
    preview: bool = False
    cameras: tuple[str, ...] = ("topdown",)
    trail: bool = False
    heatmap: bool = False
    hud: bool = False
    still: int | None = None
    save_blend: Path | None = None
    no_render: bool = False
    assets: Path | None = None
    dream_layout: str = "pip"
    extra: dict = field(default_factory=dict)      # free-form (JSON-safe), used by figure/heatmap

    @property
    def anim(self):
        from .animate import AnimConfig
        return AnimConfig(frames_per_step=PREVIEW_FPS_PER_STEP if self.preview else self.frames_per_step,
                          discrete=self.discrete)

    @property
    def resolution(self) -> tuple[int, int]:
        return PREVIEW_RES if self.preview else tuple(self.res)

    def frames_root(self) -> Path:
        out = Path(self.out)
        return out.parent / f"{out.stem}_frames"

    def frames_dir(self, camera: str) -> Path:
        return self.frames_root() / camera

    def fingerprint(self) -> dict:
        """Everything that changes a rendered PNG (JSON round-trip safe); the key of frame resume."""
        a = self.anim
        return {"resolution": list(self.resolution), "engine": self.engine, "samples": int(self.samples),
                "frames_per_step": a.frames_per_step, "discrete": bool(a.discrete),
                "cameras": list(self.cameras), "trail": bool(self.trail), "heatmap": bool(self.heatmap),
                "assets": None if self.assets is None else str(self.assets), "extra": self.extra}


def prepare_frames_root(cfg: RenderConfig) -> Path:
    """Create `<out stem>_frames/` stamped with `cfg.fingerprint()` (render.json). A root stamped
    by a different config is deleted first, so resume never reuses frames rendered with other settings."""
    root = cfg.frames_root()
    stamp = root / "render.json"
    fp = cfg.fingerprint()
    if stamp.exists():
        try:
            old = json.loads(stamp.read_text())
        except ValueError:
            old = None
        if old != fp:
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(fp, indent=1, sort_keys=True))
    return root


def build(ep: Episode, cfg: RenderConfig):
    """Reset Blender, build and animate the scene, add every camera. Returns (scene objects, cameras, last frame)."""
    from .animate import animate
    from .cameras import add_camera
    from .overlays import add_heatmap, add_trail
    from .scene.base import configure_render, reset_scene
    from .scene.minigrid import build_scene

    reset_scene()
    sc = build_scene(ep.layout, assets=cfg.assets)
    last = animate(sc, ep, cfg.anim)
    if cfg.trail:
        add_trail(sc, ep, cfg.anim)
    if cfg.heatmap:
        add_heatmap(sc, ep, cfg.anim)
    cams = {name: add_camera(name, ep.layout, ep, cfg.anim) for name in cfg.cameras}
    configure_render(cfg.engine, cfg.samples, cfg.resolution, cfg.fps)
    return sc, cams, last


def render_frames(cam, frames: Sequence[int], out_dir: Path) -> list[Path]:
    import bpy

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.camera = cam
    paths = []
    for f in frames:
        path = out_dir / f"f{int(f):05d}.png"
        if not (path.exists() and path.stat().st_size > 0):     # a 0-byte file is a killed write
            tmp = path.with_name(path.stem + ".tmp.png")          # never leave a truncated final PNG
            scene.frame_set(int(f))
            scene.render.filepath = str(tmp)
            bpy.ops.render.render(write_still=True)
            os.replace(tmp, path)
        paths.append(path)
    return paths


def hud_lines(ep: Episode, row: IndexRow, t: int) -> list[str]:
    ret = float(ep.rewards[:t].sum()) if t > 0 else 0.0
    return [f"{ep.meta.get('run_name', '')}/ep{row.episode_id:06d}   {row.phase} · {row.actor}",
            f"step {t}/{ep.length}   return {ret:.2f}"]


def dream_panels(ep: Episode, t: int) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """(real obs, model view, dreaming?) for state t of a dream episode; model view is None past the stored frames.
    Model view = posterior reconstruction while t < dream_start, imagined frame from then on."""
    if ep.dream is None or ep.obs is None:
        return None, None, False
    ds = int(ep.dream["dream_start"])
    real = ep.obs[min(t, len(ep.obs) - 1)]
    recon = ep.dream.get("recon_frames")
    if t < ds:
        model = recon[t] if recon is not None and t < len(recon) else None
        return real, model, False
    i = t - ds
    frames = ep.dream["dream_frames"]
    return real, (frames[i] if i < len(frames) else None), True


def compose_frame(images: Sequence[np.ndarray], ep: Episode, row: IndexRow, t: int, cfg: RenderConfig) -> np.ndarray:
    """Side-by-side cameras → dream-vs-reality panels (dream episodes only; white frame while the model
    still tracks the real episode, black once it dreams) → HUD."""
    img = images[0] if len(images) == 1 else hstack(images, gap=6)
    if ep.dream is not None and ep.obs is not None:
        real, model, dreaming = dream_panels(ep, t)
        colour = (0, 0, 0) if dreaming else (255, 255, 255)
        if cfg.dream_layout == "split":
            img = split(img, model if model is not None else real, divider=colour)
        else:
            insets = [(real, "real obs")] + ([(model, "dream" if dreaming else "recon")] if model is not None else [])
            img = pip(img, insets, corner="tr", frac=0.25, border=colour)
    if cfg.hud:
        img = hud(img, hud_lines(ep, row, t))
    return img


def write_mp4(frames: Iterable[np.ndarray], path: Path, fps: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = iio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for fr in frames:
            w.append_data(np.ascontiguousarray(fr[:, :, :3]))
    finally:
        w.close()
    return path


def _render_all(ep, row, cfg, cams, frames: Sequence[int]) -> dict[str, list[Path]]:
    """Render every requested frame of every camera to disk (resumable), before any compositing."""
    prepare_frames_root(cfg)
    return {name: render_frames(cam, frames, cfg.frames_dir(name)) for name, cam in cams.items()}


def _compose_from_disk(per_cam: dict[str, list[Path]], ep, row, cfg, frames: Sequence[int]):
    """Lazily composite one frame at a time from already-rendered per-camera PNGs on disk, so a
    caller consuming this generator never holds more than one composited frame in memory."""
    from .animate import frame_step
    for i, f in enumerate(frames):
        imgs = [iio3.imread(per_cam[name][i]) for name in cfg.cameras]
        yield compose_frame(imgs, ep, row, frame_step(f, cfg.anim), cfg)


def _composited(ep, row, cfg, cams, frames: Sequence[int]):
    per_cam = _render_all(ep, row, cfg, cams, frames)
    return _compose_from_disk(per_cam, ep, row, cfg, frames)


def _still_frame(ep: Episode, row: IndexRow, cfg: RenderConfig, cams, step: int) -> np.ndarray:
    """The composited frame of `step` (clamped to the episode) from an already built scene."""
    from .animate import step_frame
    f = step_frame(min(max(step, 0), ep.length), cfg.anim)
    return next(iter(_composited(ep, row, cfg, cams, [f])))


def render_still(ep: Episode, row: IndexRow, cfg: RenderConfig, step: int) -> np.ndarray:
    _, cams, _ = build(ep, cfg)
    return _still_frame(ep, row, cfg, cams, step)


def frames_composited(ep: Episode, row: IndexRow, cfg: RenderConfig) -> tuple[int, Iterable[np.ndarray]]:
    """Renders every frame of the episode's animation to disk eagerly (resumable, same as `_composited`),
    then returns `(frame count, a lazy iterator that composites one frame at a time from disk)` — so a
    caller tiling several episodes' animations together (`timeline --video`) never holds more than one
    composited frame per episode in memory, instead of the whole animation."""
    _, cams, last = build(ep, cfg)
    frames = range(1, last + 1)
    per_cam = _render_all(ep, row, cfg, cams, frames)
    return len(frames), _compose_from_disk(per_cam, ep, row, cfg, frames)


def render_episode(ep: Episode, row: IndexRow, cfg: RenderConfig) -> Path:
    from .scene.base import save_blend

    _, cams, last = build(ep, cfg)
    if cfg.save_blend is not None:
        save_blend(cfg.save_blend)
    if cfg.no_render:
        if cfg.save_blend is None:
            raise ValueError("no_render needs save_blend: nothing would be written")
        return Path(cfg.save_blend)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.still is not None:
        iio3.imwrite(out.with_suffix(".png"), _still_frame(ep, row, cfg, cams, cfg.still))
        return out.with_suffix(".png")
    return write_mp4(_composited(ep, row, cfg, cams, range(1, last + 1)), out.with_suffix(".mp4"), cfg.fps)


def _heatmap_scene(layout, cfg: RenderConfig):
    """Reset Blender and build `layout` with the agent hidden and the static top-down camera — the
    scene shared by both heatmap renders. No episode is animated: the floor tiles carry the picture."""
    from .cameras import add_camera
    from .scene.base import configure_render, keyframe_hidden, reset_scene
    from .scene.minigrid import build_scene
    reset_scene()
    sc = build_scene(layout, assets=cfg.assets)
    keyframe_hidden(sc.agent, 1, True)
    cam = add_camera("topdown", layout, None, cfg.anim)
    configure_render(cfg.engine, cfg.samples, cfg.resolution, cfg.fps)
    return sc, cam


def _keyed(cfg: RenderConfig, *arrays: np.ndarray, **more) -> RenderConfig:
    """`cfg` whose fingerprint also carries a digest of `arrays` (plus `more`). Heatmap frames depend on
    the selected episodes' counts, not only on the render settings, so a reused `--out` must not
    resume the frames of a different selection — the digest makes `prepare_frames_root` wipe them."""
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return replace(cfg, extra={**cfg.extra, "content": h.hexdigest()[:16], **more})


def render_heatmap_still(layout, counts: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    """One top-down frame of `layout` with `counts` (W×H visits) painted on the floor tiles."""
    from .overlays import set_heatmap_static
    cfg = _keyed(cfg, counts)
    sc, cam = _heatmap_scene(layout, cfg)
    set_heatmap_static(sc, counts)
    prepare_frames_root(cfg)
    path = render_frames(cam, [1], cfg.frames_dir("heatmap_still"))[0]
    img = iio3.imread(path)[:, :, :3]
    if cfg.hud:
        img = hud(img, [f"{int(counts.sum())} visits over {cfg.extra.get('n_episodes', '?')} episodes"])
    return img


def render_heatmap_frames(layout, episodes: Sequence[tuple[IndexRow, Episode]], cfg: RenderConfig,
                          frames_per_episode: int = 6) -> tuple[int, Iterable[np.ndarray]]:
    """The floor fills in episode by episode (in `start_step` order): every tile is keyed black at frame 1
    and episode i's cumulative counts at `1 + (i + 1) * frames_per_episode`, with LINEAR interpolation
    (Blender's default Bezier would ease in and out). Frames `2 .. 1 + n * frames_per_episode` are
    rendered, so block i (frames `2 + i * fpe .. 1 + (i + 1) * fpe`) shows episode i's visits fading in
    evenly and ends with them fully painted; the HUD of frame f names episode `(f - 2) // fpe`. Renders
    every frame to disk eagerly (resumable), then returns `(frame count, a lazy iterator reading one PNG
    at a time and applying the HUD)` — same contract as `frames_composited`, so tiling two runs
    (`--compare`) never holds a whole animation in memory."""
    import bpy
    from .aggregate import visit_counts
    from .overlays import heat_color
    from .scene.base import set_linear
    sc, cam = _heatmap_scene(layout, cfg)
    episodes = sorted(episodes, key=lambda re: (re[0].start_step, re[0].episode_id))
    W, H = layout.width, layout.height
    total = np.zeros((W, H), np.int64)
    cum = []
    for _, ep in episodes:
        total = total + visit_counts(ep.agent_pos, W, H)
        cum.append(total.copy())
    # `first_frame` is part of the fingerprint: frames rendered by the old keying (episode i complete
    # at 1 + i * fpe) share file names with these and must not be resumed.
    cfg = _keyed(cfg, np.stack(cum), frames_per_episode=int(frames_per_episode), first_frame=2)
    peak = max(int(cum[-1].max()), 1)
    for tile in sc.floor.values():
        tile.color = heat_color(0.0)
        tile.keyframe_insert("color", frame=1)
    for i, counts in enumerate(cum):
        f = 1 + (i + 1) * frames_per_episode
        for cell, tile in sc.floor.items():
            tile.color = heat_color(int(counts[cell[0], cell[1]]) / peak)
            tile.keyframe_insert("color", frame=f)
    for tile in sc.floor.values():
        set_linear(tile)
    n = len(episodes)
    last = 1 + n * frames_per_episode
    bpy.context.scene.frame_end = last
    prepare_frames_root(cfg)
    paths = render_frames(cam, range(2, last + 1), cfg.frames_dir("heatmap_anim"))

    def frames():
        for k, path in enumerate(paths):                  # frame 2 + k belongs to episode k // fpe
            row = episodes[k // frames_per_episode][0]
            img = iio3.imread(path)[:, :, :3]
            yield hud(img, [f"step {row.start_step}", f"{k // frames_per_episode + 1}/{n} episodes"]) if cfg.hud else img

    return n * frames_per_episode, frames()
