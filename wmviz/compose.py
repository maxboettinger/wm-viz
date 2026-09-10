"""Frame post-processing with PIL/numpy: stacking, labels, HUD, picture-in-picture, split view.
No bpy — usable for the mpl backend and in tests."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RGB = tuple[int, int, int]


def _font(size: int = 16):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return np.ascontiguousarray(a[:, :, :3])


def _canvas(h: int, w: int, color: RGB) -> np.ndarray:
    return np.tile(np.asarray(color, np.uint8), (h, w, 1))


def hstack(frames: Sequence[np.ndarray], gap: int = 4, gap_color: RGB = (0, 0, 0)) -> np.ndarray:
    frames = [to_rgb(f) for f in frames]
    h = max(f.shape[0] for f in frames)
    w = sum(f.shape[1] for f in frames) + gap * (len(frames) - 1)
    out = _canvas(h, w, gap_color)
    x = 0
    for f in frames:
        out[: f.shape[0], x: x + f.shape[1]] = f
        x += f.shape[1] + gap
    return out


def vstack(frames: Sequence[np.ndarray], gap: int = 4, gap_color: RGB = (0, 0, 0)) -> np.ndarray:
    frames = [to_rgb(f) for f in frames]
    w = max(f.shape[1] for f in frames)
    h = sum(f.shape[0] for f in frames) + gap * (len(frames) - 1)
    out = _canvas(h, w, gap_color)
    y = 0
    for f in frames:
        out[y: y + f.shape[0], : f.shape[1]] = f
        y += f.shape[0] + gap
    return out


def grid(frames: Sequence[np.ndarray], cols: int, gap: int = 4) -> np.ndarray:
    frames = [to_rgb(f) for f in frames]
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    blank = _canvas(h, w, (0, 0, 0))
    rows = []
    for i in range(0, len(frames), cols):
        row = list(frames[i: i + cols]) + [blank] * (cols - len(frames[i: i + cols]))
        rows.append(hstack(row, gap=gap))
    return vstack(rows, gap=gap)


def label(img: np.ndarray, text: str, height: int = 28, bg: RGB = (20, 20, 20), fg: RGB = (240, 240, 240)) -> np.ndarray:
    img = to_rgb(img)
    bar = Image.new("RGB", (img.shape[1], height), bg)
    ImageDraw.Draw(bar).text((6, max(1, (height - 16) // 2)), text, fill=fg, font=_font(max(10, height - 12)))
    return vstack([np.asarray(bar), img], gap=0)


def hud(img: np.ndarray, lines: Sequence[str], corner: str = "tl") -> np.ndarray:
    im = Image.fromarray(to_rgb(img)).convert("RGBA")
    font = _font(16)
    draw = ImageDraw.Draw(im, "RGBA")
    widths = [draw.textlength(s, font=font) for s in lines]
    w, h = int(max(widths) + 16), 22 * len(lines) + 8
    x = 8 if "l" in corner else im.width - w - 8
    y = 8 if "t" in corner else im.height - h - 8
    draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, 150))
    for i, s in enumerate(lines):
        draw.text((x + 8, y + 4 + 22 * i), s, fill=(255, 255, 255, 255), font=font)
    return np.asarray(im.convert("RGB"))


def upscale(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(to_rgb(img)).resize(size, Image.NEAREST))


def pip(base: np.ndarray, insets: Sequence[tuple[np.ndarray, str]], corner: str = "tr", frac: float = 0.25,
        border: RGB = (255, 255, 255), border_px: int = 3) -> np.ndarray:
    out = to_rgb(base).copy()
    H, W = out.shape[:2]
    side = int(H * frac)
    tiles = []
    for img, text in insets:
        t = upscale(img, (side, side))
        t = np.pad(t, ((border_px, border_px), (border_px, border_px), (0, 0)), constant_values=0)
        t[:border_px], t[-border_px:], t[:, :border_px], t[:, -border_px:] = border, border, border, border
        tiles.append(label(t, text, height=18))
    block = vstack(tiles, gap=4, gap_color=(0, 0, 0))
    bh, bw = block.shape[:2]
    x = W - bw - 8 if "r" in corner else 8
    y = 8 if "t" in corner else H - bh - 8
    out[y: y + bh, x: x + bw] = block
    return out


def split(left: np.ndarray, right: np.ndarray, divider: RGB = (255, 255, 255), divider_px: int = 6) -> np.ndarray:
    left = to_rgb(left)
    h = left.shape[0]
    rw = int(round(right.shape[1] * h / right.shape[0]))
    right_up = upscale(right, (rw, h))
    bar = _canvas(h, divider_px, divider)
    return hstack([left, bar, right_up], gap=0)


def strip(images: Sequence[np.ndarray], labels: Sequence[str], gap: int = 8) -> np.ndarray:
    return hstack([label(i, l) for i, l in zip(images, labels)], gap=gap, gap_color=(20, 20, 20))
