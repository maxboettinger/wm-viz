"""Render pipeline (spec §4): load → build_scene → animate → cameras → overlays → frames → post → mp4.
One episode per call; the scene is factory-reset first. Existing frames are skipped so an
interrupted render resumes."""
from __future__ import annotations

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

    def frames_dir(self, camera: str) -> Path:
        out = Path(self.out)
        return out.parent / f"{out.stem}_frames" / camera


def build(ep: Episode, cfg: RenderConfig):
    """Reset Blender, build and animate the scene, add every camera. Returns (scene objects, cameras, last frame)."""
    from .animate import animate
    from .cameras import add_camera
    from .scene.base import configure_render, reset_scene
    from .scene.minigrid import build_scene

    reset_scene()
    sc = build_scene(ep.layout, assets=cfg.assets)
    last = animate(sc, ep, cfg.anim)
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
        if not path.exists():
            scene.frame_set(int(f))
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
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
        return Path(cfg.save_blend) if cfg.save_blend is not None else Path(cfg.out)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.still is not None:
        f = step_frame(min(max(cfg.still, 0), ep.length), cfg.anim)
        img = next(iter(_composited(ep, row, cfg, cams, [f])))
        iio3.imwrite(out.with_suffix(".png"), img)
        return out.with_suffix(".png")
    return write_mp4(_composited(ep, row, cfg, cams, range(1, last + 1)), out.with_suffix(".mp4"), cfg.fps)
