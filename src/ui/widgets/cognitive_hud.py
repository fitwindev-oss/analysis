"""
Game-style HUD overlays for V6 cognitive_reaction.

Used by both CameraView (live measurement) and VideoPlayerWidget
(replay) so the operator and subject see the same visual feedback in
both contexts. Pure BGR-frame drawing — no Qt dependency, so the
helpers can be unit-tested headlessly.

Three overlays, drawn in this order on the frame:

  1. Top progress bar   — n_done/n_total + small live CRI badge
  2. Center grade burst — fades in/out around each stimulus resolution
  3. Bottom counters    — running totals for great/good/normal/bad/miss

State management is the caller's responsibility:
- live: RecorderState carries cog_progress_done, cog_recent_grade,
  cog_recent_grade_age_frames, cog_grade_counts, cog_live_cri.
- replay: VideoPlayerWidget computes the same values from result.json
  trials at the current playback time.

All functions mutate ``bgr`` in place.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Color palette (BGR)
# ─────────────────────────────────────────────────────────────────────────────
GRADE_BGR: dict[str, tuple[int, int, int]] = {
    "great":  (0, 215, 255),    # gold
    "good":   (50, 230, 50),    # green
    "normal": (50, 220, 230),   # yellow-cyan
    "bad":    (60, 60, 230),    # red-orange
    "miss":   (60, 60, 200),    # darker red
}

GRADE_TEXT: dict[str, str] = {
    "great":  "GREAT!",
    "good":   "GOOD",
    "normal": "NORMAL",
    "bad":    "BAD",
    "miss":   "MISS",
}

# Used in the bottom counters row
GRADE_ICON: dict[str, str] = {
    "great":  "*",
    "good":   "+",
    "normal": "o",
    "bad":    "x",
    "miss":   "-",
}

# Order matters — counters are rendered left→right in this order.
GRADE_ORDER: tuple[str, ...] = ("great", "good", "normal", "bad", "miss")

# How long the center grade burst stays visible (in display ticks).
# At 30 Hz the value 24 ≈ 0.8 s.
GRADE_MSG_HOLD_FRAMES: int = 24

# Background bar tint (BGR, semi-transparent via copyTo blend)
_BG_DARK = (15, 15, 15)
_BG_DARK_ALPHA = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Public draw functions
# ─────────────────────────────────────────────────────────────────────────────
def draw_progress_bar(bgr: np.ndarray,
                       n_done: int, n_total: int,
                       cri_live: Optional[float] = None) -> None:
    """Top-of-frame progress bar.

    Renders a compact dark strip with "N/M  ▰▰░░" style text and an
    optional live CRI badge on the right. Sized so it survives both
    landscape (640w) and portrait (720w rotated 1280h) layouts.
    """
    h, w = bgr.shape[:2]
    if w < 60 or h < 40:
        return
    n_total = max(0, int(n_total))
    n_done = max(0, min(int(n_done), n_total))

    # Bar geometry — relative to image width
    margin = max(8, w // 80)
    bar_h = max(28, h // 22)
    y0 = margin
    y1 = y0 + bar_h
    x0 = margin
    x1 = w - margin

    # Translucent dark backdrop
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _BG_DARK, -1)
    cv2.addWeighted(overlay, _BG_DARK_ALPHA, bgr, 1.0 - _BG_DARK_ALPHA,
                    0, bgr)
    # Subtle border
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (60, 60, 60), 1, cv2.LINE_AA)

    # Text: "5 / 10"
    text_count = f"{n_done} / {n_total}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs_count = max(0.5, h / 720.0 * 0.8)
    (tw_c, th_c), _ = cv2.getTextSize(text_count, font, fs_count, 2)
    tx_c = x0 + 8
    ty_c = y0 + (bar_h + th_c) // 2 - 2
    cv2.putText(bgr, text_count, (tx_c + 1, ty_c + 1),
                font, fs_count, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, text_count, (tx_c, ty_c),
                font, fs_count, (240, 240, 240), 2, cv2.LINE_AA)

    # Live CRI badge on the right
    cri_w = 0
    if cri_live is not None and np.isfinite(cri_live):
        cri_text = f"CRI {float(cri_live):.0f}"
        fs_cri = fs_count
        (tw_r, th_r), _ = cv2.getTextSize(cri_text, font, fs_cri, 2)
        tx_r = x1 - tw_r - 8
        ty_r = ty_c
        cv2.putText(bgr, cri_text, (tx_r + 1, ty_r + 1),
                    font, fs_cri, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, cri_text, (tx_r, ty_r),
                    font, fs_cri, _cri_color(cri_live), 2, cv2.LINE_AA)
        cri_w = tw_r + 16

    # Bar geometry — between count text and CRI badge
    bar_x0 = tx_c + tw_c + 12
    bar_x1 = x1 - cri_w - 8
    bar_y0 = y0 + 6
    bar_y1 = y1 - 6
    if bar_x1 - bar_x0 > 10:
        cv2.rectangle(bgr, (bar_x0, bar_y0), (bar_x1, bar_y1),
                      (50, 50, 50), -1)
        if n_total > 0:
            frac = n_done / n_total
            fill_x = bar_x0 + int(round(frac * (bar_x1 - bar_x0)))
            if fill_x > bar_x0:
                # Gradient-ish fill — solid fitwin-green
                cv2.rectangle(bgr, (bar_x0, bar_y0), (fill_x, bar_y1),
                              (170, 245, 0), -1)
        cv2.rectangle(bgr, (bar_x0, bar_y0), (bar_x1, bar_y1),
                      (90, 90, 90), 1, cv2.LINE_AA)


def draw_grade_message(bgr: np.ndarray, grade: Optional[str],
                        rt_ms: Optional[float],
                        age_frames: int) -> None:
    """Center-of-frame grade burst — fades over GRADE_MSG_HOLD_FRAMES.

    ``grade`` ∈ great/good/normal/bad/miss. ``age_frames`` is the
    number of repaints since the burst started (0 = just fired).
    Returns silently if the burst has aged out or grade is unknown.
    """
    if grade is None or grade not in GRADE_TEXT:
        return
    if age_frames < 0 or age_frames >= GRADE_MSG_HOLD_FRAMES:
        return
    h, w = bgr.shape[:2]
    if w < 80 or h < 80:
        return

    # Linear fade out from full opacity to ~0 over the hold window
    t = age_frames / float(GRADE_MSG_HOLD_FRAMES)   # 0..1
    alpha = max(0.05, 1.0 - t)
    # GREAT pulses (size oscillation), others stay steady
    pulse = 1.0 + (0.10 * np.sin(2 * np.pi * t * 2.0)
                   if grade == "great" else 0.0)

    color = GRADE_BGR[grade]
    text = GRADE_TEXT[grade]
    if grade == "great":
        text = "✨ " + text + " ✨"

    font = cv2.FONT_HERSHEY_DUPLEX
    fs = max(0.9, h / 540.0 * 1.6) * pulse
    thick = 3 if grade == "great" else 2
    (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
    cx = w // 2 - tw // 2
    # Center vertically slightly above middle so it doesn't fight
    # the LED cue (which usually lives near the edges).
    cy = h // 2 - h // 12 + th // 2

    # Drop shadow + main text on a translucent overlay so we can blend
    overlay = bgr.copy()
    cv2.putText(overlay, text, (cx + 2, cy + 2),
                font, fs, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (cx, cy),
                font, fs, color, thick, cv2.LINE_AA)

    # RT subtitle (only when we know it AND grade is not miss)
    if (rt_ms is not None and np.isfinite(rt_ms)
            and grade not in ("miss",)):
        sub = f"{int(round(float(rt_ms)))} ms"
        fs_sub = fs * 0.45
        (sw, sh), _ = cv2.getTextSize(sub, font, fs_sub, 2)
        sx = w // 2 - sw // 2
        sy = cy + th + int(sh * 1.2)
        cv2.putText(overlay, sub, (sx + 1, sy + 1),
                    font, fs_sub, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, sub, (sx, sy),
                    font, fs_sub, (220, 220, 220), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0, bgr)


def draw_grade_counters(bgr: np.ndarray, grade_counts: dict) -> None:
    """Bottom-of-frame running totals — one chip per grade in
    GRADE_ORDER. Layout adapts to image width."""
    h, w = bgr.shape[:2]
    if w < 80 or h < 80:
        return
    margin = max(8, w // 80)
    bar_h = max(26, h // 24)
    y1 = h - margin
    y0 = y1 - bar_h
    x0 = margin
    x1 = w - margin

    overlay = bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _BG_DARK, -1)
    cv2.addWeighted(overlay, _BG_DARK_ALPHA, bgr, 1.0 - _BG_DARK_ALPHA,
                    0, bgr)
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (60, 60, 60), 1, cv2.LINE_AA)

    # Lay out chips evenly across the strip
    n = len(GRADE_ORDER)
    chip_w = (x1 - x0 - 2 * 6) // n
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.45, h / 720.0 * 0.7)
    for i, g in enumerate(GRADE_ORDER):
        cnt = int(grade_counts.get(g, 0)) if grade_counts else 0
        cx0 = x0 + 6 + i * chip_w
        # Color stripe on the left edge of each chip
        cv2.rectangle(bgr, (cx0, y0 + 4), (cx0 + 4, y1 - 4),
                      GRADE_BGR[g], -1)
        # Chip label: "GREAT 3"
        text = f"{GRADE_TEXT[g].rstrip('!')} {cnt}"
        (tw, th), _ = cv2.getTextSize(text, font, fs, 2)
        tx = cx0 + 10
        ty = y0 + (bar_h + th) // 2 - 2
        cv2.putText(bgr, text, (tx + 1, ty + 1),
                    font, fs, (0, 0, 0), 3, cv2.LINE_AA)
        col = GRADE_BGR[g] if cnt > 0 else (140, 140, 140)
        cv2.putText(bgr, text, (tx, ty), font, fs, col, 2, cv2.LINE_AA)


def draw_full_hud(bgr: np.ndarray, *,
                   n_done: int, n_total: int,
                   recent_grade: Optional[str], recent_rt_ms: Optional[float],
                   recent_age_frames: int,
                   grade_counts: Optional[dict],
                   live_cri: Optional[float] = None) -> None:
    """Convenience wrapper that draws all three overlays in order so
    callers don't repeat themselves."""
    draw_progress_bar(bgr, n_done, n_total, cri_live=live_cri)
    draw_grade_message(bgr, recent_grade, recent_rt_ms, recent_age_frames)
    draw_grade_counters(bgr, grade_counts or {})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cri_color(cri: float) -> tuple[int, int, int]:
    """Color the live CRI badge by performance band."""
    if cri >= 85.0:
        return GRADE_BGR["great"]
    if cri >= 70.0:
        return GRADE_BGR["good"]
    if cri >= 55.0:
        return GRADE_BGR["normal"]
    if cri >= 40.0:
        return GRADE_BGR["bad"]
    return GRADE_BGR["miss"]
