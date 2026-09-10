"""Render pipeline (spec §4): load → build_scene → animate → cameras → overlays → frames → post → mp4.
One episode per call; the scene is factory-reset first. Existing frames are skipped so an
interrupted render resumes — but only frames of the same settings: `<out stem>_frames/render.json`
holds the RenderConfig fingerprint and a mismatching frames directory is wiped first."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as iio
import imageio.v3 as iio3
import numpy as np

from .compose import hstack, hud
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
    extra: dict = field(default_factory=dict)      # free-form, used by figure/heatmap

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
                "assets": None if self.assets is None else str(self.assets)}


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


def compose_frame(images: Sequence[np.ndarray], ep: Episode, row: IndexRow, t: int, cfg: RenderConfig) -> np.ndarray:
    img = images[0] if len(images) == 1 else hstack(images, gap=6)
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


def _composited(ep, row, cfg, cams, frames: Sequence[int]):
    from .animate import frame_step
    prepare_frames_root(cfg)
    per_cam = {name: render_frames(cam, frames, cfg.frames_dir(name)) for name, cam in cams.items()}
    for i, f in enumerate(frames):
        imgs = [iio3.imread(per_cam[name][i]) for name in cfg.cameras]
        yield compose_frame(imgs, ep, row, frame_step(f, cfg.anim), cfg)


def render_still(ep: Episode, row: IndexRow, cfg: RenderConfig, step: int) -> np.ndarray:
    from .animate import step_frame
    _, cams, _ = build(ep, cfg)
    f = step_frame(min(max(step, 0), ep.length), cfg.anim)
    return next(iter(_composited(ep, row, cfg, cams, [f])))


def render_episode(ep: Episode, row: IndexRow, cfg: RenderConfig) -> Path:
    from .animate import step_frame
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
        f = step_frame(min(max(cfg.still, 0), ep.length), cfg.anim)
        img = next(iter(_composited(ep, row, cfg, cams, [f])))
        iio3.imwrite(out.with_suffix(".png"), img)
        return out.with_suffix(".png")
    return write_mp4(_composited(ep, row, cfg, cams, range(1, last + 1)), out.with_suffix(".mp4"), cfg.fps)
