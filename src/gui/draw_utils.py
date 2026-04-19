"""
draw_utils.py - Shared OpenCV/PIL drawing helpers for overlay rendering.
"""
import cv2
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

# ── Chinese font helpers ──────────────────────────────────────────────────────

_ZH_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",    # Microsoft YaHei (Win10/11)
    "C:/Windows/Fonts/simsun.ttc",  # SimSun
    "C:/Windows/Fonts/simhei.ttf",  # SimHei
]
_ZH_FONT_CACHE: dict = {}


def get_zh_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _ZH_FONT_CACHE:
        return _ZH_FONT_CACHE[size]
    for path in _ZH_FONT_PATHS:
        try:
            f = ImageFont.truetype(path, size)
            _ZH_FONT_CACHE[size] = f
            return f
        except (IOError, OSError):
            continue
    f = ImageFont.load_default()
    _ZH_FONT_CACHE[size] = f
    return f


def measure_text_zh(text: str, size: int) -> Tuple[int, int]:
    """Return (width, height) of text at given pixel font size."""
    font = get_zh_font(size)
    dummy = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(dummy)
    bb = d.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def put_text_zh(
    frame: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    size: int,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """Render Unicode/Chinese text onto an OpenCV BGR frame using PIL."""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(xy, text, font=get_zh_font(size), fill=color_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── Bounding box drawing ──────────────────────────────────────────────────────

def draw_fancy_bbox(frame: np.ndarray, bbox, color: tuple, label: str) -> np.ndarray:
    """Draw a corner-bracket style bounding box with label."""
    x, y, w, h = [int(v) for v in bbox]
    x2, y2 = x + w, y + h
    br, g, bl = color[2], color[1], color[0]
    bgr = (bl, g, br)
    corner = min(w, h) // 5
    thickness = 2

    for (px, py, dx, dy) in [
        (x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (px, py), (px + dx * corner, py), bgr, thickness + 1)
        cv2.line(frame, (px, py), (px, py + dy * corner), bgr, thickness + 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bgr, 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    if label:
        font_size = 14
        tw, th = measure_text_zh(label, font_size)
        pad = 4
        chip_x1 = x
        chip_y1 = max(0, y - th - pad * 2)
        chip_x2 = x + tw + pad * 2
        chip_y2 = y
        cv2.rectangle(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), bgr, -1)
        frame = put_text_zh(frame, label, (chip_x1 + pad, chip_y1 + pad), font_size, (15, 15, 15))
    return frame


def draw_state_badge(frame: np.ndarray, label: str, color_rgb: tuple) -> np.ndarray:
    """Top-left status badge with given label and color."""
    if not label:
        return frame
    bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    text = f"  {label}  "
    font_size = 15
    tw, th = measure_text_zh(text, font_size)
    pad = 6
    x1, y1 = 12, 12
    x2, y2 = x1 + tw + pad, y1 + th + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 28), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
    frame = put_text_zh(frame, text, (x1 + pad // 2, y1 + pad // 2), font_size, color_rgb)
    return frame


def draw_crosshair(
    frame: np.ndarray,
    error_x: float,
    error_y: float,
    offset_x: int = 0,
    offset_y: int = 0,
    calibrating: bool = False,
) -> np.ndarray:
    """Draw center crosshair + servo error indicator arrow.

    offset_x/offset_y: pixel offset of the calibrated center from frame center.
    calibrating: when True, draw the crosshair in a highlighted color.
    """
    h, w = frame.shape[:2]
    # Nominal frame center
    fx, fy = w // 2, h // 2
    # Calibrated servo center
    cx = fx + offset_x
    cy = fy + offset_y

    # Static crosshair at calibrated center
    color = (0, 255, 200) if calibrating else (180, 180, 180)
    size = 24
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
    cv2.circle(frame, (cx, cy), 5, color, 1)

    if calibrating:
        # Show frame center as dim reference
        cv2.line(frame, (fx - 10, fy), (fx + 10, fy), (80, 80, 80), 1)
        cv2.line(frame, (fx, fy - 10), (fx, fy + 10), (80, 80, 80), 1)
        frame = put_text_zh(frame, "校准模式 — 点击画面设置中心点",
                            (cx + 8, cy - 22), 12, (0, 255, 200))

    # Error arrow (object center relative to calibrated center)
    arrow_scale = min(w, h) * 0.3
    ex = int(cx + error_x * arrow_scale)
    ey = int(cy + error_y * arrow_scale)
    err_mag = (error_x ** 2 + error_y ** 2) ** 0.5
    if err_mag > 0.02:
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (80, 200, 255), 2, tipLength=0.3)

    return frame


def draw_approach_bar(frame: np.ndarray, area_ratio: float, threshold: float) -> np.ndarray:
    """Bottom progress bar showing approach completion (bbox area / frame area)."""
    h, w = frame.shape[:2]
    bar_h = 10
    bar_y = h - bar_h - 5
    bar_x = 20
    bar_w = w - 40

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    # Threshold marker
    thresh_x = bar_x + int(threshold * bar_w / 0.5)  # scale: 0–50% maps to full bar
    thresh_x = min(thresh_x, bar_x + bar_w)
    cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3), (100, 100, 255), 2)

    # Fill
    fill_ratio = min(area_ratio / 0.5, 1.0)
    fill_w = int(fill_ratio * bar_w)
    fill_color = (80, 220, 80) if area_ratio >= threshold else (80, 140, 220)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

    # Label
    label = f"逼近: {area_ratio * 100:.1f}% / {threshold * 100:.0f}%"
    frame = put_text_zh(frame, label, (bar_x, bar_y - 20), 13, (200, 200, 200))
    return frame
