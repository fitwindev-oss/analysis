"""
CameraView — live preview panel for every available camera.

Consumes frames from RecordWorker.camera_frame(cam_id, bgr, frame_idx, t_ns)
and, when enabled, MediaPipe skeleton overlays from PoseLiveWorker via
``on_pose_overlay(cam_id, kpts33_px, vis33)``. Throttles repaint to
``config.PLOT_UPDATE_MS`` to protect the GUI thread from burst traffic.

Tile count follows ``config.CAMERAS`` at construction time (after startup
camera detection has pruned missing devices).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy,
)

import time

import config
from src.pose.mediapipe_backend import MP33, MP33_CONNECTIONS


_VIS_THRESH = 0.5           # drop landmarks with lower visibility
_KPT_COLOR  = (0, 255, 255)  # yellow (BGR)
_EDGE_COLOR = (0, 200, 0)    # green  (BGR)

# V6 — positional cue ring (cognitive_reaction). Drawn as a layered
# circle (outer halo + bright core) to read LED-like at small scales.
# Tuned for the project palette's FITWIN green (#AAF500 ≈ BGR (0,245,170)).
_CUE_HALO_COLOR  = (0, 245, 170)   # BGR — soft halo
_CUE_CORE_COLOR  = (255, 255, 255) # BGR — bright core
_CUE_LABEL_COLOR = (255, 255, 255) # BGR — label text

# V6-hit — colors for the "hit" state (tracked body part inside the
# cue's tolerance ring). Bright cyan-yellow + thicker ring + checkmark
# read as a clear positive ack distinct from the resting cue.
_HIT_HALO_COLOR  = (80, 255, 255)   # BGR — bright cyan-yellow halo
_HIT_CORE_COLOR  = (50, 230, 50)    # BGR — vivid green core
_HIT_TICK_COLOR  = (50, 230, 50)    # BGR — checkmark color
_TRACKED_KPT_COLOR_REST = (0, 230, 230)  # BGR — yellow when tracking, no hit
_TRACKED_KPT_COLOR_HIT  = (50, 230, 50)  # BGR — green when in hit zone

# V6-hit — body-part name → primary MP33 keypoint index. Mirrors the
# server-side BODY_PART_TO_KEYPOINTS table in src.analysis.cognitive_reaction
# but indexed directly so the GUI doesn't need to import that module.
BODY_PART_TO_KP_INDEX: dict[str, int] = {
    "right_hand": MP33["right_wrist"],
    "left_hand":  MP33["left_wrist"],
    "right_foot": MP33["right_foot_index"],
    "left_foot":  MP33["left_foot_index"],
}

# Tolerance for declaring a hit, in normalised image-diagonal units.
# Matches src.analysis.cognitive_reaction.analyze_cognitive_reaction's
# default ``hit_tolerance_norm`` so the live feedback ring agrees with
# the offline report's pass/fail call.
_DEFAULT_HIT_TOLERANCE_NORM = 0.12


class _SingleCamTile(QWidget):
    """One cam — header + image area. Keeps latest frame, repaint throttled."""

    def __init__(self, cam_id: str, label: str, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self._latest: Optional[np.ndarray] = None   # BGR
        # Pose overlay buffers (None = don't draw)
        self._kpts:   Optional[np.ndarray] = None
        self._vis:    Optional[np.ndarray] = None
        # V6 — positional cue (normalised image coords). When non-None,
        # draws a glowing ring centered at (x_norm * w, y_norm * h) on
        # every repaint until cleared.
        self._cue_xy: Optional[tuple[float, float]] = None
        self._cue_label: Optional[str] = None
        # ``_cue_phase`` increments on every repaint while a cue is set,
        # so the ring radius can pulse and read more LED-like.
        self._cue_phase: int = 0
        # V6-hit — index of the MP33 keypoint we treat as the "tracked"
        # body part for live hit detection. None = no tracking (no
        # special highlight; cue still draws but never goes to "hit").
        self._track_kpt_idx: Optional[int] = None
        self._hit_tolerance_norm: float = _DEFAULT_HIT_TOLERANCE_NORM
        # Sticky hit latch — once the tracked keypoint enters the
        # tolerance ring, hold the "hit" state for ``_hit_hold_frames``
        # repaints so brief overshoots still register as a clean ack.
        self._hit_hold_frames_left: int = 0
        self._HIT_HOLD_FRAMES = 12   # ~0.4 s at 30 Hz UI repaint
        # V6-G3 — game HUD state, set by the parent CameraView from
        # the recorder's RecorderState. None means "no HUD" (other
        # tests). Repaint reads these to draw progress / grade burst /
        # counters via the cognitive_hud module.
        self._hud_state: Optional[dict] = None
        # Dimensions the pose was computed on — may differ from display if the
        # caller feeds us a different-resolution camera. We key all drawing
        # off the frame's own pixel dimensions so rescaling is not needed.

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        header = QLabel(f"{cam_id} · {label}")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "color:#90caf9; font-weight:bold; font-size:11px; padding:2px;")
        root.addWidget(header)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            "background:#111; border:1px solid #333; color:#666;")
        self._img.setMinimumSize(240, 180)
        self._img.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._render_placeholder()
        root.addWidget(self._img, stretch=1)

    # ── public API ─────────────────────────────────────────────────────────
    def set_frame(self, bgr: np.ndarray) -> None:
        self._latest = bgr

    def set_pose(self, kpts33: Optional[np.ndarray],
                 vis33: Optional[np.ndarray]) -> None:
        self._kpts = kpts33
        self._vis  = vis33

    def set_positional_cue(self, x_norm: Optional[float],
                            y_norm: Optional[float],
                            label: Optional[str] = None) -> None:
        """Show / hide the V6 positional cue ring on this tile.

        Pass ``None`` for either coord to hide. Coordinates are
        normalised image space (0..1, top-left origin) so they survive
        any resolution change between cameras.
        """
        if x_norm is None or y_norm is None:
            self._cue_xy = None
            self._cue_label = None
            self._cue_phase = 0
            self._hit_hold_frames_left = 0
            return
        # New cue → reset the hit latch so the previous trial's hit
        # doesn't bleed into the new one.
        if self._cue_xy != (float(x_norm), float(y_norm)):
            self._hit_hold_frames_left = 0
        self._cue_xy = (float(x_norm), float(y_norm))
        self._cue_label = label

    def set_track_body_part(self, body_part: Optional[str]) -> None:
        """Set the MP33 keypoint we'll treat as the "tracked" body part
        for live hit detection. Pass ``None`` to disable.
        """
        if not body_part:
            self._track_kpt_idx = None
            return
        self._track_kpt_idx = BODY_PART_TO_KP_INDEX.get(body_part)

    def set_hud_state(self, hud: Optional[dict]) -> None:
        """V6-G3 — drive the in-frame game HUD. ``hud`` is a dict with
        keys: n_done, n_total, recent_grade, recent_rt_ms,
        recent_age_frames, grade_counts, live_cri. Pass None to hide."""
        self._hud_state = hud

    def clear(self) -> None:
        self._latest = None
        self._kpts = None
        self._vis = None
        self._cue_xy = None
        self._cue_label = None
        self._cue_phase = 0
        self._hit_hold_frames_left = 0
        self._render_placeholder()

    def repaint_if_dirty(self) -> bool:
        """Render the latest frame. Returns the RAW hit state (no
        latch) so the parent CameraView can broadcast on transitions
        without the latch's hold-window smearing the rising edge."""
        if self._latest is None:
            return False
        bgr = self._latest
        self._latest = None
        # Copy once if any overlay is going to draw, so we don't mutate
        # the caller's buffer (recorder thread may still be reading it).
        has_skeleton = (self._kpts is not None and self._vis is not None)
        has_cue      = (self._cue_xy is not None)
        has_hud      = (self._hud_state is not None)
        if has_skeleton or has_cue or has_hud:
            bgr = bgr.copy()
        # V6-hit — evaluate hit BEFORE drawing skeleton so we can use
        # the same colored marker on the tracked keypoint when it's in
        # the hit zone (visible cue ↔ tracked dot ↔ skeleton coloring
        # all stay in sync).
        raw_hit = False
        is_hit = False
        if has_cue and has_skeleton and self._track_kpt_idx is not None:
            raw_hit = self._evaluate_hit(bgr.shape[1], bgr.shape[0])
            if raw_hit:
                self._hit_hold_frames_left = self._HIT_HOLD_FRAMES
                is_hit = True
            elif self._hit_hold_frames_left > 0:
                # Hit latch still active — keep "hit" state for a few
                # repaints so brief overshoots don't flicker the ack.
                self._hit_hold_frames_left -= 1
                is_hit = True
        if has_skeleton:
            self._draw_skeleton(bgr, self._kpts, self._vis,
                                track_kpt_idx=self._track_kpt_idx,
                                track_hit=is_hit)
        if has_cue:
            self._cue_phase = (self._cue_phase + 1) % 60
            self._draw_positional_cue(
                bgr, self._cue_xy, self._cue_label, self._cue_phase,
                hit=is_hit)
        # V6-G3 — game HUD overlay (progress bar / grade burst /
        # counters). Drawn last so it sits on top of skeleton + cue.
        if has_hud:
            from src.ui.widgets.cognitive_hud import draw_full_hud
            h = self._hud_state
            draw_full_hud(
                bgr,
                n_done=int(h.get("n_done", 0) or 0),
                n_total=int(h.get("n_total", 0) or 0),
                recent_grade=h.get("recent_grade"),
                recent_rt_ms=h.get("recent_rt_ms"),
                recent_age_frames=int(h.get("recent_age_frames", 0) or 0),
                grade_counts=h.get("grade_counts") or {},
                live_cri=h.get("live_cri"),
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._img.width(), self._img.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img.setPixmap(pix)
        return raw_hit

    # ── drawing ────────────────────────────────────────────────────────────
    def _evaluate_hit(self, w: int, h: int) -> bool:
        """Is the tracked keypoint inside the cue's hit-tolerance ring?

        Returns False if either the cue or the tracked keypoint is
        unavailable (no pose, missing visibility, NaN coords). Distance
        is measured in image-normalised units so it survives camera
        resolution changes — same metric the offline analyzer uses.
        """
        if (self._cue_xy is None or self._track_kpt_idx is None
                or self._kpts is None or self._vis is None):
            return False
        idx = self._track_kpt_idx
        if idx < 0 or idx >= 33 or w <= 1 or h <= 1:
            return False
        if self._kpts.shape[0] < 33 or self._vis.shape[0] < 33:
            return False
        if self._vis[idx] < _VIS_THRESH:
            return False
        p = self._kpts[idx]
        if np.any(np.isnan(p)):
            return False
        # Convert tracked keypoint to normalised coords (same convention
        # as cue: 0..1, top-left origin) and use diagonal-normalised
        # Euclidean distance — matches the analyzer's err_norm.
        x_norm = float(p[0]) / float(w - 1)
        y_norm = float(p[1]) / float(h - 1)
        cx, cy = self._cue_xy
        # Aspect-correct distance via image-diagonal:
        # err_norm = sqrt((dx*w)^2 + (dy*h)^2) / sqrt(w^2 + h^2)
        dx_px = (x_norm - cx) * (w - 1)
        dy_px = (y_norm - cy) * (h - 1)
        diag = float(np.hypot(w, h))
        if diag <= 0:
            return False
        err_norm = float(np.hypot(dx_px, dy_px)) / diag
        return err_norm <= self._hit_tolerance_norm

    @staticmethod
    def _draw_skeleton(bgr: np.ndarray, kpts33: np.ndarray,
                       vis33: np.ndarray,
                       track_kpt_idx: Optional[int] = None,
                       track_hit: bool = False) -> None:
        """Draw MP33 landmarks + connections in-place on a BGR frame.

        ``track_kpt_idx`` (if given) is highlighted larger and in a
        different color to make it obvious which body part is being
        tracked — green when it's inside the cue's hit zone, yellow
        otherwise. The other 32 dots use the muted joint color.
        """
        if kpts33 is None or vis33 is None:
            return
        if kpts33.shape[0] < 33 or vis33.shape[0] < 33:
            return
        h, w = bgr.shape[:2]
        # Edges first (lines under dots)
        for a, b in MP33_CONNECTIONS:
            if vis33[a] < _VIS_THRESH or vis33[b] < _VIS_THRESH:
                continue
            pa = kpts33[a]; pb = kpts33[b]
            if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                continue
            x1 = int(max(0, min(w - 1, pa[0])))
            y1 = int(max(0, min(h - 1, pa[1])))
            x2 = int(max(0, min(w - 1, pb[0])))
            y2 = int(max(0, min(h - 1, pb[1])))
            cv2.line(bgr, (x1, y1), (x2, y2), _EDGE_COLOR, 2, cv2.LINE_AA)
        # Joint dots
        for i in range(33):
            if vis33[i] < _VIS_THRESH:
                continue
            p = kpts33[i]
            if np.any(np.isnan(p)):
                continue
            x = int(max(0, min(w - 1, p[0])))
            y = int(max(0, min(h - 1, p[1])))
            if i == track_kpt_idx:
                # Tracked body part — bigger ring + filled center,
                # color shifts when in hit zone for instant feedback.
                color = _TRACKED_KPT_COLOR_HIT if track_hit \
                    else _TRACKED_KPT_COLOR_REST
                cv2.circle(bgr, (x, y), 9, color, 2, cv2.LINE_AA)
                cv2.circle(bgr, (x, y), 4, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(bgr, (x, y), 3, _KPT_COLOR, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_positional_cue(bgr: np.ndarray,
                              xy_norm: tuple[float, float],
                              label: Optional[str],
                              phase: int,
                              hit: bool = False) -> None:
        """Draw an LED-style ring + label at the cued spot.

        ``xy_norm`` is in [0,1] image-normalised coords (top-left origin).
        ``phase`` cycles 0..59 to drive a gentle radius pulse so the cue
        reads as "alive" rather than a static decal.
        ``hit`` switches the look to "ack" mode — bright colors + a
        checkmark — when the tracked body part has reached the target.
        """
        h, w = bgr.shape[:2]
        if w <= 1 or h <= 1:
            return
        cx = int(round(xy_norm[0] * (w - 1)))
        cy = int(round(xy_norm[1] * (h - 1)))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        # Base radius scales with the smaller image dimension so the cue
        # stays visible across portrait + landscape camera streams.
        base = max(14, min(w, h) // 18)
        # Pulse: ±15 % rest, ±25 % when hit (faster + deeper modulation
        # for an obvious "ack" feel).
        if hit:
            pulse = 1.0 + 0.25 * np.sin(2 * np.pi * phase / 30.0)
        else:
            pulse = 1.0 + 0.15 * np.sin(2 * np.pi * phase / 60.0)
        r_outer = int(round(base * 1.6 * pulse))
        r_mid   = int(round(base * 1.0 * pulse))
        r_core  = int(round(base * 0.55))
        halo_color = _HIT_HALO_COLOR if hit else _CUE_HALO_COLOR
        core_color = _HIT_CORE_COLOR if hit else _CUE_CORE_COLOR
        ring_thick = 5 if hit else 3
        halo_alpha = 0.55 if hit else 0.35
        # Halo — translucent overlay so the camera image still shows
        # through. Stronger blend in the hit state so the ack reads
        # against busy backgrounds.
        overlay = bgr.copy()
        cv2.circle(overlay, (cx, cy), r_outer, halo_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, halo_alpha, bgr, 1.0 - halo_alpha, 0, bgr)
        # Crisp ring + bright core
        cv2.circle(bgr, (cx, cy), r_mid,  halo_color, ring_thick, cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), r_core, core_color, -1, cv2.LINE_AA)
        # V6-hit — checkmark inside the core when in hit state. Drawn
        # as two anti-aliased line segments so it scales cleanly with
        # ``base``.
        if hit:
            check_scale = max(0.6, base / 18.0)
            # Checkmark vertices in pixels relative to (cx, cy)
            p1 = (cx - int(7 * check_scale), cy + int(1 * check_scale))
            p2 = (cx - int(2 * check_scale), cy + int(6 * check_scale))
            p3 = (cx + int(8 * check_scale), cy - int(6 * check_scale))
            tk = max(2, int(round(3 * check_scale)))
            cv2.line(bgr, p1, p2, _HIT_TICK_COLOR, tk + 1, cv2.LINE_AA)
            cv2.line(bgr, p2, p3, _HIT_TICK_COLOR, tk + 1, cv2.LINE_AA)
            # White inner stroke to lift the tick off the green core
            cv2.line(bgr, p1, p2, (255, 255, 255), max(1, tk - 1), cv2.LINE_AA)
            cv2.line(bgr, p2, p3, (255, 255, 255), max(1, tk - 1), cv2.LINE_AA)
        # Optional label below the ring
        if label:
            text = ("✓ " if hit else "") + label.replace("pos_", "")
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = max(2, min(w - tw - 2, cx - tw // 2))
            ty = min(h - 4, cy + r_outer + th + 6)
            label_color = _HIT_TICK_COLOR if hit else _CUE_LABEL_COLOR
            # Drop-shadow so text reads against any background
            cv2.putText(bgr, text, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(bgr, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color,
                        2, cv2.LINE_AA)

    def _render_placeholder(self) -> None:
        w, h = 240, 180
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(blank, self.cam_id, (10, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        cv2.putText(blank, "idle", (10, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        img = QImage(blank.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._img.setPixmap(QPixmap.fromImage(img))


class CameraView(QWidget):
    """Row of cam tiles (one per config.CAMERAS entry) + repaint timer.

    T8: minimum size + sizePolicy hints to portrait-rotation 9:16.
    The Logitech StreamCam is mounted in landscape but the capture
    pipeline rotates 90° → portrait 720×1280. Without an aspect-aware
    size hint, the parent layout gives this widget the full available
    width and the video shows with huge left/right letterbox bars.

    By setting Preferred sizePolicy with a min portrait shape, the
    parent layout can choose to keep us narrow (less letterbox).
    """

    # Camera capture is rotated 90° clockwise → final visible aspect
    # 9:16 (height ≈ 1.78 × width).
    _ASPECT_W_OVER_H = 9 / 16   # = 0.5625

    # V6-G3 — fires when at least one tile reports a hit transition
    # for the active cognitive_reaction cue. The MeasureTab connects
    # this signal to RecordWorker.feed_hit_indicator so the recorder
    # can time RT relative to stim fire instant. Bool payload mirrors
    # the new is_hit state; t_ns is monotonic_ns for RT precision.
    cog_hit_state_changed = pyqtSignal(bool, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tiles: dict[str, _SingleCamTile] = {}
        # Last broadcast hit state — only emit when it flips so the
        # recorder doesn't get hammered every repaint.
        self._last_emitted_hit: bool = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        for cam in config.CAMERAS:
            tile = _SingleCamTile(cam["id"], cam["label"])
            self._tiles[cam["id"]] = tile
            lay.addWidget(tile, stretch=1)

        # Tighter aspect-ratio hint so parent layouts don't over-stretch
        # the widget horizontally (T8).
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.MinimumExpanding)
        self.setMinimumSize(220, 380)   # portrait ~9:16 minimum

        # Repaint timer — drives _flush() which pulls latest frames from
        # each tile into the QLabel. Must come AFTER the size policy
        # block so __init__ doesn't return early.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._flush)
        self._timer.start(config.PLOT_UPDATE_MS)   # ~33 ms → 30 fps

    def sizeHint(self) -> QSize:
        # Hint a portrait shape so the parent QHBoxLayout doesn't
        # default to making us very wide. ~340w × 600h is a comfortable
        # 24"-monitor target without letterbox.
        return QSize(340, 600)

    # ── public API ─────────────────────────────────────────────────────────
    def on_camera_frame(self, cam_id: str, bgr: np.ndarray,
                        frame_idx: int, t_ns: int) -> None:
        tile = self._tiles.get(cam_id)
        if tile is not None:
            tile.set_frame(bgr)

    def on_pose_overlay(self, cam_id: str,
                        kpts33: Optional[np.ndarray],
                        vis33: Optional[np.ndarray]) -> None:
        tile = self._tiles.get(cam_id)
        if tile is not None:
            tile.set_pose(kpts33, vis33)

    def clear_overlay(self, cam_id: Optional[str] = None) -> None:
        if cam_id is None:
            for t in self._tiles.values():
                t.set_pose(None, None)
        else:
            t = self._tiles.get(cam_id)
            if t is not None:
                t.set_pose(None, None)

    def set_positional_cue(self, x_norm: Optional[float],
                            y_norm: Optional[float],
                            label: Optional[str] = None,
                            cam_id: Optional[str] = None) -> None:
        """Mirror the V6 cognitive_reaction cue onto every (or one) tile.

        ``x_norm`` / ``y_norm`` are normalised image coords; pass None to
        hide the cue. The recorder fires the cue once per stim from any
        thread; this method is safe to call from the GUI thread via
        signal-slot.
        """
        if cam_id is None:
            for t in self._tiles.values():
                t.set_positional_cue(x_norm, y_norm, label)
        else:
            t = self._tiles.get(cam_id)
            if t is not None:
                t.set_positional_cue(x_norm, y_norm, label)

    def set_track_body_part(self, body_part: Optional[str]) -> None:
        """Tell every tile which MP33 keypoint to treat as the "tracked"
        body part for live hit detection (V6 cognitive_reaction).

        ``body_part`` ∈ {right_hand, left_hand, right_foot, left_foot}
        or None to disable. Hit state is only evaluated on tiles that
        also receive pose overlays via ``on_pose_overlay``; the live
        pose worker only runs on one camera so other tiles still draw
        the cue but never light up.
        """
        for t in self._tiles.values():
            t.set_track_body_part(body_part)

    def set_cog_hud_state(self, hud: Optional[dict]) -> None:
        """V6-G3 — broadcast a single HUD state dict to every tile so
        all camera previews show the same gamified overlay during a
        cognitive_reaction session. Pass None to clear."""
        for t in self._tiles.values():
            t.set_hud_state(hud)

    def reset(self) -> None:
        for t in self._tiles.values():
            t.clear()

    # ── internals ──────────────────────────────────────────────────────────
    def _flush(self) -> None:
        any_hit = False
        for t in self._tiles.values():
            tile_hit = t.repaint_if_dirty()
            if tile_hit:
                any_hit = True
        # V6-G3 — emit only when the aggregated hit state flips so the
        # recorder doesn't get hammered every repaint.
        if any_hit != self._last_emitted_hit:
            self._last_emitted_hit = any_hit
            try:
                self.cog_hit_state_changed.emit(
                    bool(any_hit), int(time.monotonic_ns()))
            except Exception:
                pass
