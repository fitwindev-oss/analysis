"""
VideoPlayerWidget — timestamp-synced playback of a session's camera video.

Takes a session directory + cam_id, opens the mp4 with OpenCV, and displays
the frame closest to whatever force-timeline second is set via
``set_time(t_force_s)``. Optionally overlays the MediaPipe skeleton if the
corresponding ``poses_<cam>.npz`` exists.

V6 — when the session is ``cognitive_reaction``, this widget also reads
``stimulus_log.csv`` and overlays the on-screen LED cue + ✓ check at the
exact frames the trainer saw during recording. The hit detection mirrors
the live CameraView logic so replay agrees with what the operator saw.

Not a free-running player — the PlaybackController drives ``set_time``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

from src.analysis.pose2d import (
    resolve_pose_frame, load_session_pose2d,
    get_record_start_wall_s,
)
from src.pose.mediapipe_backend import MP33, MP33_CONNECTIONS


_VIS_THRESH = 0.5
_KPT_COLOR  = (0, 255, 255)   # yellow BGR
_EDGE_COLOR = (0, 200, 0)     # green  BGR

# V6 cue overlay — must mirror src.ui.widgets.camera_view so the
# replay view looks identical to the live measurement.
_CUE_HALO_COLOR  = (0, 245, 170)
_CUE_CORE_COLOR  = (255, 255, 255)
_CUE_LABEL_COLOR = (255, 255, 255)
_HIT_HALO_COLOR  = (80, 255, 255)
_HIT_CORE_COLOR  = (50, 230, 50)
_HIT_TICK_COLOR  = (50, 230, 50)
_TRACKED_KPT_COLOR_REST = (0, 230, 230)
_TRACKED_KPT_COLOR_HIT  = (50, 230, 50)

_BODY_PART_TO_KP_INDEX: dict[str, int] = {
    "right_hand": MP33["right_wrist"],
    "left_hand":  MP33["left_wrist"],
    "right_foot": MP33["right_foot_index"],
    "left_foot":  MP33["left_foot_index"],
}

# How long the LED cue stays on screen during recording (matches
# SessionRecorder._fire_stim's banner_hold_s for cognitive_reaction).
_CUE_HOLD_S = 1.5
# Replay-time hit tolerance — same default as the offline analyzer
# and live CameraView so all three views agree on what counts as a hit.
_HIT_TOLERANCE_NORM = 0.12


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_dir: Optional[Path] = None
        self._cam_id: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._current_idx: int = -1
        self._overlay_enabled = False
        self._pose_series = None   # optional Pose2DSeries
        # V6 — cognitive_reaction cue replay state. Populated when the
        # session is a cognitive_reaction recording so set_time can
        # overlay the on-screen LED + ✓ check at the right frames.
        self._cue_events: list[dict] = []          # [{t_stim_s, x, y, label}]
        self._cue_track_kpt_idx: Optional[int] = None
        self._cue_phase_counter: int = 0
        self._is_cognitive_reaction: bool = False
        # V6-G4 — gamified HUD state for cognitive_reaction replay.
        # Each trial is a dict with t_stim_s, rt_ms, hit, grade. The
        # HUD at any t_force_s shows the running totals across all
        # trials whose t_stim has elapsed by t.
        self._hud_trials: list[dict] = []
        self._cog_n_total: int = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self._img = QLabel("(비디오 없음)")
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(
            "QLabel { background:#0a0a0a; color:#666; border:1px solid #333; }")
        self._img.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img.setMinimumHeight(240)
        lay.addWidget(self._img)

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, session_dir: str | Path, cam_id: str) -> bool:
        """Open the video. Returns True on success."""
        self.unload()
        sd = Path(session_dir)
        video_path = sd / f"{cam_id}.mp4"
        if not video_path.exists():
            self._img.setText(f"(비디오 없음: {cam_id}.mp4)")
            return False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._img.setText(f"(비디오 열기 실패: {cam_id}.mp4)")
            return False
        self._cap = cap
        self._session_dir = sd
        self._cam_id = cam_id
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_idx = -1
        # Load pose series if available (for overlay)
        series = load_session_pose2d(sd)
        self._pose_series = series.get(cam_id) if series else None
        # V6 — load cognitive_reaction cue events so the replay can
        # paint the on-screen LED + ✓ check at the same moments the
        # operator saw them during recording.
        self._load_cognitive_cues(sd)
        # Show first frame
        self.set_time(0.0)
        return True

    def _load_cognitive_cues(self, session_dir: Path) -> None:
        """Read stimulus_log.csv + session.json to prep cue overlay state.

        For non-cognitive_reaction sessions this leaves all the cue
        fields cleared so the existing playback path is untouched.
        """
        self._cue_events = []
        self._cue_track_kpt_idx = None
        self._is_cognitive_reaction = False
        meta: dict = {}
        try:
            mp = session_dir / "session.json"
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return
        if (meta or {}).get("test") != "cognitive_reaction":
            return
        self._is_cognitive_reaction = True
        # Body part → MP33 index for live hit check
        body_part = (meta or {}).get("cog_track_body_part") or "right_hand"
        self._cue_track_kpt_idx = _BODY_PART_TO_KP_INDEX.get(body_part)
        # Stimulus log: convert wall times → record-relative seconds so
        # they line up with PlaybackController's clock (t_force_s).
        stim_path = session_dir / "stimulus_log.csv"
        if not stim_path.exists():
            return
        try:
            import pandas as pd
            df = pd.read_csv(stim_path)
        except Exception:
            return
        rec_start = get_record_start_wall_s(session_dir)
        if rec_start is None:
            # Fall back to forces.csv first sample
            try:
                df_f = pd.read_csv(session_dir / "forces.csv",
                                    usecols=["t_wall"], nrows=1)
                rec_start = float(df_f["t_wall"].iloc[0])
            except Exception:
                return
        for _, row in df.iterrows():
            try:
                tw = float(row.get("t_wall", float("nan")))
                if not np.isfinite(tw):
                    continue
                t_stim_s = float(tw - rec_start)
                tx = row.get("target_x_norm", None)
                ty = row.get("target_y_norm", None)
                lbl = str(row.get("target_label",
                                   row.get("response_type", "")))
                # Skip stim without a positional target (legacy reaction
                # rows leaking in, etc.)
                if tx is None or ty is None:
                    continue
                tx_f = float(tx); ty_f = float(ty)
                if not (np.isfinite(tx_f) and np.isfinite(ty_f)):
                    continue
                self._cue_events.append({
                    "t_stim_s": t_stim_s,
                    "x_norm":   tx_f,
                    "y_norm":   ty_f,
                    "label":    lbl,
                })
            except Exception:
                continue
        # Sort by stim time so _active_cue_at can early-exit.
        self._cue_events.sort(key=lambda e: e["t_stim_s"])

        # V6-G4 — load result.json trials so the HUD can replay grades
        # in lockstep with the cues. Falls back to no-grade HUD when
        # result.json is missing or empty (analysis hasn't run yet).
        self._hud_trials = []
        self._cog_n_total = (meta or {}).get("n_stimuli") \
            or len(self._cue_events) or 0
        try:
            rp = session_dir / "result.json"
            if rp.exists():
                payload = json.loads(rp.read_text(encoding="utf-8"))
                trials = ((payload or {}).get("result") or {}).get("trials") or []
                # Match each trial to a cue event by index (analyzer
                # builds the trial list in stim_log order).
                for i, tr in enumerate(trials):
                    if not isinstance(tr, dict):
                        continue
                    t_stim_s = tr.get("t_stim_s")
                    if t_stim_s is None and i < len(self._cue_events):
                        t_stim_s = self._cue_events[i]["t_stim_s"]
                    if t_stim_s is None:
                        continue
                    self._hud_trials.append({
                        "t_stim_s": float(t_stim_s),
                        "rt_ms":    tr.get("rt_ms"),
                        "hit":      bool(tr.get("hit", False)),
                        "grade":    tr.get("grade"),
                    })
                self._hud_trials.sort(key=lambda t: t["t_stim_s"])
        except Exception:
            self._hud_trials = []

    def unload(self) -> None:
        if self._cap is not None:
            try: self._cap.release()
            except Exception: pass
        self._cap = None
        self._session_dir = None
        self._cam_id = None
        self._pose_series = None
        self._current_idx = -1
        self._cue_events = []
        self._cue_track_kpt_idx = None
        self._cue_phase_counter = 0
        self._is_cognitive_reaction = False
        self._hud_trials = []
        self._cog_n_total = 0
        self._img.clear()
        self._img.setText("(비디오 없음)")

    def set_overlay_enabled(self, on: bool) -> None:
        self._overlay_enabled = bool(on) and self._pose_series is not None
        # Redraw current frame with new overlay state
        if self._cap is not None and self._current_idx >= 0:
            self._redraw_current()

    def has_pose(self) -> bool:
        return self._pose_series is not None

    def duration_s(self) -> float:
        """Session-relative duration (from force record start to video end).

        Uses timestamps.csv when available so it's accurate even if the
        video's internal fps differs from wall fps."""
        if self._session_dir is None or self._cam_id is None:
            return 0.0
        try:
            from src.analysis.pose2d import (
                load_video_timestamps, get_record_start_wall_s,
            )
            walls = load_video_timestamps(self._session_dir, self._cam_id)
            rec  = get_record_start_wall_s(self._session_dir)
            if walls is not None and rec is not None and len(walls) > 0:
                return max(0.0, float(walls[-1] - rec))
        except Exception:
            pass
        # Fallback — assume nominal fps covers wait + duration
        try:
            import json
            meta = json.loads((self._session_dir / "session.json").read_text(
                encoding="utf-8"))
            return float(meta.get("duration_s", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ── playback hook ──────────────────────────────────────────────────────
    def set_time(self, t_force_s: float) -> None:
        if self._cap is None:
            return
        # Timestamp-based mapping (same as analysis sync)
        frame_idx = resolve_pose_frame(
            float(t_force_s), self._session_dir, self._cam_id, self._fps)
        frame_idx = max(0, min(self._n_frames - 1, frame_idx))
        # V6 — cue overlay needs the current force-time even if the
        # video frame didn't change (e.g. paused on a stim instant). So
        # we always re-render when a cue is active or just expired.
        cue = self._active_cue_at(float(t_force_s))
        hud = self._hud_state_at(float(t_force_s))
        if frame_idx == self._current_idx and cue is None \
                and not self._is_cognitive_reaction:
            return
        if frame_idx != self._current_idx:
            # Seek: small forward moves can use sequential read; otherwise set
            if 0 < frame_idx - self._current_idx <= 8:
                target = self._current_idx + 1
                while target <= frame_idx:
                    ok, frame = self._cap.read()
                    if not ok:
                        return
                    target += 1
            else:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = self._cap.read()
                if not ok:
                    return
            self._current_idx = frame_idx
        else:
            # Same frame — re-grab from the stored frame index for
            # cue-only redraw. CAP_PROP_POS_FRAMES is sticky so this
            # is fine even after a previous seek.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_idx)
            ok, frame = self._cap.read()
            if not ok:
                return
        self._render(frame, cue=cue, hud=hud)

    def _active_cue_at(self, t_force_s: float) -> Optional[dict]:
        """Return the cue event whose [t_stim, t_stim + _CUE_HOLD_S]
        window contains ``t_force_s``, else None. Only ever populated
        for cognitive_reaction sessions."""
        if not self._cue_events:
            return None
        # Linear scan is fine — typically <= 10 stims per session.
        for ev in self._cue_events:
            t0 = ev["t_stim_s"]
            if t0 <= t_force_s <= t0 + _CUE_HOLD_S:
                return ev
        return None

    def _hud_state_at(self, t_force_s: float) -> Optional[dict]:
        """V6-G4 — assemble the gamified HUD state for replay.

        At the given playback time, returns a dict shaped like the
        live measurement HUD (see CameraView.set_cog_hud_state). The
        most recent grade burst stays visible for 0.8 s after each
        trial's resolution time (stim + 1.5 s cue hold).
        """
        if not self._is_cognitive_reaction:
            return None
        # Lazy import to keep the analyzer cost off the hot path
        from src.analysis.cognitive_reaction import live_cri_after
        from src.ui.widgets.cognitive_hud import GRADE_MSG_HOLD_FRAMES

        n_done = 0
        recent_grade: Optional[str] = None
        recent_rt_ms: Optional[float] = None
        recent_age_frames: int = GRADE_MSG_HOLD_FRAMES + 1   # expired
        grade_counts: dict = {"great": 0, "good": 0, "normal": 0,
                              "bad": 0, "miss": 0}
        trials_done: list[dict] = []

        # Iterate trials in time order. A trial counts as "done" once
        # its cue-hold window has elapsed (= grade resolved).
        for tr in self._hud_trials:
            t_stim = float(tr["t_stim_s"])
            t_resolved = t_stim + _CUE_HOLD_S
            if t_force_s < t_stim:
                # Future trial — not yet started
                break
            if t_force_s < t_resolved:
                # In flight: stim fired but grade not yet locked in
                continue
            n_done += 1
            g = tr.get("grade") or "miss"
            if g not in grade_counts:
                grade_counts[g] = 0
            grade_counts[g] += 1
            trials_done.append(tr)
            # Has its grade burst window started?
            burst_age_s = t_force_s - t_resolved
            burst_window_s = (GRADE_MSG_HOLD_FRAMES / 30.0)
            if 0.0 <= burst_age_s <= burst_window_s:
                recent_grade = g
                recent_rt_ms = tr.get("rt_ms")
                recent_age_frames = int(round(burst_age_s * 30.0))

        # Live CRI = compute_cri across resolved trials so far
        live_cri = live_cri_after(trials_done) if trials_done else 0.0
        return {
            "n_done":            n_done,
            "n_total":           int(self._cog_n_total or 0),
            "recent_grade":      recent_grade,
            "recent_rt_ms":      recent_rt_ms,
            "recent_age_frames": recent_age_frames,
            "grade_counts":      grade_counts,
            "live_cri":          float(live_cri),
        }

    # ── rendering ──────────────────────────────────────────────────────────
    def _redraw_current(self) -> None:
        if self._cap is None or self._current_idx < 0:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_idx)
        ok, frame = self._cap.read()
        if ok:
            # No cue context here (called from set_overlay_enabled toggle).
            # Cue redraw happens via the next set_time tick.
            self._render(frame)

    def _render(self, bgr: np.ndarray, cue: Optional[dict] = None,
                 hud: Optional[dict] = None) -> None:
        has_skeleton = (self._overlay_enabled and self._pose_series is not None
                        and 0 <= self._current_idx < len(self._pose_series))
        has_cue = cue is not None
        has_hud = hud is not None
        # Track-keypoint highlight needs both pose AND a tracked body
        # part (set only for cognitive_reaction sessions).
        track_idx = self._cue_track_kpt_idx if has_cue else None
        if has_skeleton or has_cue or has_hud:
            bgr = bgr.copy()
        # V6 — evaluate hit before drawing skeleton so the tracked
        # joint dot can switch color in lockstep with the cue.
        is_hit = False
        if has_cue and has_skeleton and track_idx is not None:
            kpts = self._pose_series.kpts_mp33[self._current_idx]
            vis  = self._pose_series.vis_mp33[self._current_idx]
            is_hit = _is_keypoint_in_cue(
                kpts, vis, track_idx,
                cue["x_norm"], cue["y_norm"],
                bgr.shape[1], bgr.shape[0],
                _HIT_TOLERANCE_NORM,
            )
        if has_skeleton:
            _draw_skeleton(bgr,
                           self._pose_series.kpts_mp33[self._current_idx],
                           self._pose_series.vis_mp33[self._current_idx],
                           track_kpt_idx=track_idx, track_hit=is_hit)
        if has_cue:
            self._cue_phase_counter = (self._cue_phase_counter + 1) % 60
            _draw_positional_cue(
                bgr, (cue["x_norm"], cue["y_norm"]),
                cue.get("label"), self._cue_phase_counter, hit=is_hit)
        # V6-G4 — replay HUD (progress / grade burst / counters)
        if has_hud:
            from src.ui.widgets.cognitive_hud import draw_full_hud
            draw_full_hud(
                bgr,
                n_done=int(hud.get("n_done", 0) or 0),
                n_total=int(hud.get("n_total", 0) or 0),
                recent_grade=hud.get("recent_grade"),
                recent_rt_ms=hud.get("recent_rt_ms"),
                recent_age_frames=int(hud.get("recent_age_frames", 0) or 0),
                grade_counts=hud.get("grade_counts") or {},
                live_cri=hud.get("live_cri"),
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


def _draw_skeleton(bgr: np.ndarray, kpts33: np.ndarray,
                   vis33: np.ndarray,
                   track_kpt_idx: Optional[int] = None,
                   track_hit: bool = False) -> None:
    h, w = bgr.shape[:2]
    for a, b in MP33_CONNECTIONS:
        if vis33[a] < _VIS_THRESH or vis33[b] < _VIS_THRESH:
            continue
        pa, pb = kpts33[a], kpts33[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        x1 = int(max(0, min(w - 1, pa[0])))
        y1 = int(max(0, min(h - 1, pa[1])))
        x2 = int(max(0, min(w - 1, pb[0])))
        y2 = int(max(0, min(h - 1, pb[1])))
        cv2.line(bgr, (x1, y1), (x2, y2), _EDGE_COLOR, 2, cv2.LINE_AA)
    for i in range(33):
        if vis33[i] < _VIS_THRESH:
            continue
        p = kpts33[i]
        if np.any(np.isnan(p)):
            continue
        x = int(max(0, min(w - 1, p[0])))
        y = int(max(0, min(h - 1, p[1])))
        if i == track_kpt_idx:
            color = _TRACKED_KPT_COLOR_HIT if track_hit \
                else _TRACKED_KPT_COLOR_REST
            cv2.circle(bgr, (x, y), 9, color, 2, cv2.LINE_AA)
            cv2.circle(bgr, (x, y), 4, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(bgr, (x, y), 3, _KPT_COLOR, -1, cv2.LINE_AA)


def _is_keypoint_in_cue(kpts33: np.ndarray, vis33: np.ndarray,
                         kpt_idx: int,
                         cue_x_norm: float, cue_y_norm: float,
                         w: int, h: int,
                         hit_tolerance_norm: float) -> bool:
    """Replay-time hit check. Mirrors CameraView._evaluate_hit so live
    and replay agree on what counts as a ✓ acceptance."""
    if kpt_idx < 0 or kpt_idx >= 33 or w <= 1 or h <= 1:
        return False
    if kpts33.shape[0] < 33 or vis33.shape[0] < 33:
        return False
    if vis33[kpt_idx] < _VIS_THRESH:
        return False
    p = kpts33[kpt_idx]
    if np.any(np.isnan(p)):
        return False
    diag = float(np.hypot(w, h))
    if diag <= 0:
        return False
    cue_px = (float(cue_x_norm) * (w - 1), float(cue_y_norm) * (h - 1))
    err_px = float(np.hypot(p[0] - cue_px[0], p[1] - cue_px[1]))
    return (err_px / diag) <= float(hit_tolerance_norm)


def _draw_positional_cue(bgr: np.ndarray,
                          xy_norm: tuple[float, float],
                          label: Optional[str],
                          phase: int,
                          hit: bool = False) -> None:
    """Replay-side LED cue + ✓ check overlay. Same visual language as
    src.ui.widgets.camera_view._draw_positional_cue so the recording
    and the replay look identical."""
    h, w = bgr.shape[:2]
    if w <= 1 or h <= 1:
        return
    cx = int(round(xy_norm[0] * (w - 1)))
    cy = int(round(xy_norm[1] * (h - 1)))
    cx = max(0, min(w - 1, cx)); cy = max(0, min(h - 1, cy))
    base = max(14, min(w, h) // 18)
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
    overlay = bgr.copy()
    cv2.circle(overlay, (cx, cy), r_outer, halo_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, halo_alpha, bgr, 1.0 - halo_alpha, 0, bgr)
    cv2.circle(bgr, (cx, cy), r_mid, halo_color, ring_thick, cv2.LINE_AA)
    cv2.circle(bgr, (cx, cy), r_core, core_color, -1, cv2.LINE_AA)
    if hit:
        s = max(0.6, base / 18.0)
        p1 = (cx - int(7 * s), cy + int(1 * s))
        p2 = (cx - int(2 * s), cy + int(6 * s))
        p3 = (cx + int(8 * s), cy - int(6 * s))
        tk = max(2, int(round(3 * s)))
        cv2.line(bgr, p1, p2, _HIT_TICK_COLOR, tk + 1, cv2.LINE_AA)
        cv2.line(bgr, p2, p3, _HIT_TICK_COLOR, tk + 1, cv2.LINE_AA)
        cv2.line(bgr, p1, p2, (255, 255, 255), max(1, tk - 1), cv2.LINE_AA)
        cv2.line(bgr, p2, p3, (255, 255, 255), max(1, tk - 1), cv2.LINE_AA)
    if label:
        text = ("✓ " if hit else "") + str(label).replace("pos_", "")
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = max(2, min(w - tw - 2, cx - tw // 2))
        ty = min(h - 4, cy + r_outer + th + 6)
        label_color = _HIT_TICK_COLOR if hit else _CUE_LABEL_COLOR
        cv2.putText(bgr, text, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2,
                    cv2.LINE_AA)
