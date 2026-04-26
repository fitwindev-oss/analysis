"""
ReplayPanel — embedded synchronized replay for one session.

Layout (Phase G redesign — mirrors the Measure tab for video + encoder
bars; graphs live in a dedicated right column so the replay reads like a
richer version of what the trainer just saw during recording):

    ┌──────────────────────┬──────────────────────────┐
    │                      │  ┌──────────┬──────────┐ │
    │  [enc1] VIDEO [enc2] │  │   CoP    │  Coord   │ │
    │   bar          bar   │  │ (large)  │ (large)  │ │
    │                      │  ├──────────┴──────────┤ │
    │                      │  │  Encoder TS (L/R)   │ │
    │                      │  ├──────────────────────┤│
    │                      │  │  Force TS           │ │
    │                      │  ├──────────────────────┤│
    │                      │  │  Angle stack        │ │
    │                      │  └─────────────────────┘ │
    └──────────────────────┴──────────────────────────┘
    ▶ [slider] t/total  speed  Skeleton
    Angle: [slot1∨][slot2∨][slot3∨]    Coord: [slot1∨][slot2∨]

Conditional visibility:
    - Balance tests OR uses_encoder=False: encoder bars + encoder TS both
      show a "비활성" placeholder (layout stays consistent).
    - Pose unavailable: angle/coord selectors disabled with a hint.

All plots share a single ``PlaybackController`` clock.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QLabel,
    QComboBox, QSlider, QCheckBox, QFrame, QScrollArea, QSizePolicy,
)

import numpy as np

import config
from src.analysis.common import (
    ForceSession, load_force_session,
    load_departure_events, compute_departure_events,
)
from src.analysis.pose2d import (
    load_session_pose2d, ANGLE_NAMES, Pose2DSeries, resolve_pose_frame,
)
from src.ui.widgets.video_player import VideoPlayerWidget
from src.ui.widgets.force_timeline import ForceTimelineWidget
from src.ui.widgets.encoder_timeline import EncoderTimelineWidget
from src.ui.widgets.encoder_bar import EncoderBar
from src.ui.widgets.cop_trail import CopTrailWidget
from src.ui.widgets.angle_timeline import AngleTimelineStack, N_SLOTS as N_ANGLE
from src.ui.widgets.joint_coord_trail import (
    JointCoordTrail, COORD_CHOICES, N_SLOTS as N_COORD,
)
from src.ui.widgets.replay_colors import ANGLE_COLORS, COORD_COLORS
from src.ui.widgets.playback_controller import PlaybackController
from src.ui.widgets.departure_slider import DepartureSlider


def _encoder_active_for_meta(meta: dict) -> bool:
    """Mirror of MeasureTab's rule: encoder UI is active iff the test is
    not a balance test AND uses_encoder is True (defaulting True for
    historical sessions where the flag was absent)."""
    test_type = (meta or {}).get("test", "")
    if test_type in ("balance_eo", "balance_ec"):
        return False
    return bool((meta or {}).get("uses_encoder", True))


_OFF_LABEL = "— (off)"


class ReplayPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_dir: Optional[Path] = None
        self._cam_id: Optional[str] = None
        self._pose: Optional[Pose2DSeries] = None
        # ForceSession cached so we can index enc1/enc2 at the playback
        # cursor time for the live encoder-bar updates during replay.
        self._force: Optional[ForceSession] = None
        self._enc_active: bool = True
        # Departure event start times (in session seconds), used by the
        # ``[`` / ``]`` shortcuts to step between events. Populated by
        # _apply_departures() during load_session.
        self._departure_times: list[float] = []
        self._controller = PlaybackController(parent=self)
        self._build_ui()
        self._wire_signals()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Horizontal split: LEFT = video+encoder bars, RIGHT = data stack.
        # The entire analysis area lives in the right column now — force
        # timeline is no longer a full-width strip at the bottom. This
        # mirrors the Measure tab (video centre, dashboard right) so users
        # have the same spatial mental model across record + replay.
        # Expose as attribute so AppWindow.save_splitter can persist it (T4).
        self._outer_split = QSplitter(Qt.Orientation.Horizontal)
        outer = self._outer_split

        # ── LEFT column: encoder bars flanking the video player ───────────
        video_col = QWidget()
        vlay = QHBoxLayout(video_col)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(4)
        self._enc1_bar = EncoderBar(
            label="L (enc1)",
            max_mm=float(getattr(config, "ENCODER_MAX_DISPLAY_MM", 2000)),
            available=bool(getattr(config, "ENCODER1_AVAILABLE", True)),
        )
        self._enc2_bar = EncoderBar(
            label="R (enc2)",
            max_mm=float(getattr(config, "ENCODER_MAX_DISPLAY_MM", 2000)),
            available=bool(getattr(config, "ENCODER2_AVAILABLE", False)),
        )
        self._video = VideoPlayerWidget()
        vlay.addWidget(self._enc1_bar)
        vlay.addWidget(self._video, stretch=1)
        vlay.addWidget(self._enc2_bar)

        # ── RIGHT column: fixed-size [CoP | Coord] on top, scrollable
        # timeseries stack below. This way expanding the Angles selection
        # never squeezes CoP/Coord (they're protected at a fixed height)
        # and never pushes the controls row off-screen (they sit in the
        # outer root layout below the scrollable area).
        self._cop    = CopTrailWidget()
        self._coord  = JointCoordTrail()
        self._angles = AngleTimelineStack()
        self._enc_timeline = EncoderTimelineWidget()
        self._timeline     = ForceTimelineWidget()

        # CoP/Coord: match the Measure tab's CoP dimensions so all three
        # user types (researcher/trainer/subject) can read them at a glance.
        # Match the constructor min in CopTrailWidget (380×300, plate
        # aspect 1.27) so both Board1/Board2 outlines stay fully on
        # screen at narrow window widths (U2-2).
        self._cop.setMinimumSize(380, 300)
        self._coord.setMinimumSize(380, 300)
        # Encoder / Force get a tight min so they stay readable even after
        # many angle slots are added; Angle stack has its own internal sizing.
        self._enc_timeline.setMinimumHeight(160)
        self._timeline.setMinimumHeight(160)
        self._angles.setMinimumHeight(200)

        # Top row: CoP | Coord, side-by-side. U2-2: switched from
        # ``setFixedHeight(360)`` to a flexible minimum + expanding
        # policy. The fixed height was causing aspect-locked CoP plots
        # to crop their Y axis when the row was wider than the data
        # aspect (1.29:1) demanded — board outlines disappeared at the
        # plot edges. With a free height the row can grow to satisfy
        # both children's natural aspect, keeping board outlines
        # always fully visible.
        top_row_w = QWidget()
        top_row_lay = QHBoxLayout(top_row_w)
        top_row_lay.setContentsMargins(0, 0, 0, 0)
        top_row_lay.setSpacing(4)
        top_row_lay.addWidget(self._cop, stretch=1)
        top_row_lay.addWidget(self._coord, stretch=1)
        top_row_w.setMinimumHeight(380)
        top_row_w.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.MinimumExpanding)

        # Scroll area wrapping the reorderable timeseries stack. When the
        # sum of encoder+force+angle heights exceeds the available space,
        # the scrollbar appears so the user can see everything.
        self._ts_container = QWidget()
        self._ts_layout = QVBoxLayout(self._ts_container)
        self._ts_layout.setContentsMargins(0, 0, 0, 0)
        self._ts_layout.setSpacing(6)

        # Wrap each timeseries in a section that includes [▲][▼] buttons
        # + a title. Sections are kept in ``self._ts_sections`` so we can
        # shuffle them without rebuilding the child widgets (their state
        # stays intact across reorder).
        self._ts_sections: list[QWidget] = [
            self._make_ts_section("엔코더 (L/R)", self._enc_timeline),
            self._make_ts_section("Force (Total / L / R)", self._timeline),
            self._make_ts_section("관절 각도", self._angles),
        ]
        for sec in self._ts_sections:
            self._ts_layout.addWidget(sec)
        self._refresh_reorder_buttons()

        self._ts_scroll = QScrollArea()
        self._ts_scroll.setWidgetResizable(True)
        self._ts_scroll.setWidget(self._ts_container)
        self._ts_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ts_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ts_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Right column container
        right_col = QWidget()
        right_col.setMinimumWidth(420)
        rc_lay = QVBoxLayout(right_col)
        rc_lay.setContentsMargins(0, 0, 0, 0)
        rc_lay.setSpacing(4)
        # Departure summary badge — populated by load_session() when the
        # session's events.csv has entries. Hidden when there are zero
        # events so the right column doesn't waste vertical space.
        self._departure_badge = QLabel()
        self._departure_badge.setVisible(False)
        self._departure_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._departure_badge.setStyleSheet(
            "QLabel { background:#3d2a00; color:#FFB74D; "
            "border:1px solid #FF9800; border-radius:4px; "
            "padding:4px 10px; font-size:11px; font-weight:bold; }"
        )
        rc_lay.addWidget(self._departure_badge)
        rc_lay.addWidget(top_row_w)
        rc_lay.addWidget(self._ts_scroll, stretch=1)

        outer.addWidget(video_col)
        outer.addWidget(right_col)
        outer.setStretchFactor(0, 3)
        outer.setStretchFactor(1, 5)
        outer.setSizes([560, 900])
        root.addWidget(outer, stretch=1)

        # Controls row 1: play controls
        ctrl1 = QHBoxLayout()
        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(46)
        self._btn_play.setStyleSheet(
            "QPushButton { background:#2E7D32; color:white; font-size:14px; "
            "padding:6px; font-weight:bold; }")
        ctrl1.addWidget(self._btn_play)

        self._slider = DepartureSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        ctrl1.addWidget(self._slider, stretch=1)

        self._time_label = QLabel("0.00 / 0.00 s")
        self._time_label.setStyleSheet("color:#ddd; padding:0 6px;")
        ctrl1.addWidget(self._time_label)

        self._speed = QComboBox()
        for s, label in [(0.25, "0.25×"), (0.5, "0.5×"), (1.0, "1×"),
                         (2.0, "2×"), (4.0, "4×")]:
            self._speed.addItem(label, s)
        self._speed.setCurrentIndex(2)
        ctrl1.addWidget(self._speed)

        self._overlay = QCheckBox("🎯 Skeleton")
        self._overlay.setToolTip(
            "2D 포즈 처리된 세션에서만 활성화. 영상 위에 MediaPipe "
            "skeleton 오버레이를 표시합니다.")
        self._overlay.setEnabled(False)
        ctrl1.addWidget(self._overlay)
        root.addLayout(ctrl1)

        # Controls row 2: angle selectors
        ctrl2 = QHBoxLayout()
        ctrl2.addWidget(self._pill("Angle", "#FFF176"))
        self._angle_selects: list[QComboBox] = []
        for i in range(N_ANGLE):
            cb = QComboBox()
            cb.addItem(_OFF_LABEL, None)
            for name in ANGLE_NAMES:
                cb.addItem(name, name)
            cb.setStyleSheet(
                f"QComboBox {{ color:{ANGLE_COLORS[i]}; font-weight:bold; }}")
            cb.setMinimumWidth(130)
            ctrl2.addWidget(cb)
            self._angle_selects.append(cb)

        ctrl2.addSpacing(20)
        ctrl2.addWidget(self._pill("Coord", "#CE93D8"))
        self._coord_selects: list[QComboBox] = []
        for i in range(N_COORD):
            cb = QComboBox()
            cb.addItem(_OFF_LABEL, None)
            for j in COORD_CHOICES:
                cb.addItem(j, j)
            cb.setStyleSheet(
                f"QComboBox {{ color:{COORD_COLORS[i]}; font-weight:bold; }}")
            cb.setMinimumWidth(140)
            ctrl2.addWidget(cb)
            self._coord_selects.append(cb)

        ctrl2.addStretch(1)
        self._pose_hint = QLabel("(2D 포즈 처리 후 사용 가능)")
        self._pose_hint.setStyleSheet("color:#888; font-size:11px;")
        self._pose_hint.setVisible(False)
        ctrl2.addWidget(self._pose_hint)

        root.addLayout(ctrl2)

    # ── Reorder sections ───────────────────────────────────────────────────
    def _make_ts_section(self, title: str, content: QWidget) -> QWidget:
        """Wrap a timeseries widget with a header bar that has title +
        up/down reorder buttons.

        The header button states (enabled/disabled) are refreshed after
        every move by ``_refresh_reorder_buttons`` so the top section's
        "up" and the bottom section's "down" are grayed out.
        """
        sec = QWidget()
        sec.setProperty("ts_title", title)      # for debug / tests
        lay = QVBoxLayout(sec)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(4)
        btn_up = QPushButton("▲")
        btn_up.setFixedSize(26, 22)
        btn_up.setToolTip("이 그래프를 위로")
        btn_up.setStyleSheet(
            "QPushButton { background:#2a2a2a; color:#ddd; border:1px solid #444; }"
            "QPushButton:hover:!disabled { background:#3a3a3a; }"
            "QPushButton:disabled { color:#555; }"
        )
        btn_up.clicked.connect(lambda: self._move_section(sec, -1))

        btn_down = QPushButton("▼")
        btn_down.setFixedSize(26, 22)
        btn_down.setToolTip("이 그래프를 아래로")
        btn_down.setStyleSheet(btn_up.styleSheet())
        btn_down.clicked.connect(lambda: self._move_section(sec, +1))

        lbl = QLabel(title)
        lbl.setStyleSheet(
            "QLabel { color:#90caf9; font-weight:bold; font-size:11px; "
            "padding:2px 6px; }")

        header.addWidget(btn_up)
        header.addWidget(btn_down)
        header.addWidget(lbl, stretch=1)
        lay.addLayout(header)
        lay.addWidget(content)

        # Store refs on the section for later enable/disable updates
        sec.setProperty("btn_up_obj", id(btn_up))
        sec.setProperty("btn_down_obj", id(btn_down))
        sec._btn_up = btn_up       # type: ignore[attr-defined]
        sec._btn_down = btn_down   # type: ignore[attr-defined]
        return sec

    def _move_section(self, sec: QWidget, delta: int) -> None:
        """Swap ``sec`` with its neighbor in the scrollable TS layout."""
        idx = self._ts_layout.indexOf(sec)
        if idx < 0:
            return
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= self._ts_layout.count():
            return
        # Qt doesn't offer a swap primitive — remove + reinsert.
        self._ts_layout.removeWidget(sec)
        self._ts_layout.insertWidget(new_idx, sec)
        # Keep the in-memory section list aligned with the layout order
        self._ts_sections.remove(sec)
        self._ts_sections.insert(new_idx, sec)
        self._refresh_reorder_buttons()

    def _refresh_reorder_buttons(self) -> None:
        n = len(self._ts_sections)
        for i, sec in enumerate(self._ts_sections):
            try:
                sec._btn_up.setEnabled(i > 0)       # type: ignore[attr-defined]
                sec._btn_down.setEnabled(i < n - 1) # type: ignore[attr-defined]
            except AttributeError:
                pass

    @staticmethod
    def _pill(text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"QLabel {{ color:{color}; font-weight:bold; padding:2px 6px; "
            f"border:1px solid #444; border-radius:4px; }}")
        return lbl

    def _wire_signals(self) -> None:
        self._btn_play.clicked.connect(self._controller.toggle)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._speed.currentIndexChanged.connect(self._on_speed_changed)
        self._overlay.toggled.connect(self._video.set_overlay_enabled)

        for i, cb in enumerate(self._angle_selects):
            cb.currentIndexChanged.connect(lambda _v, i=i: self._on_angle_select(i))
        for i, cb in enumerate(self._coord_selects):
            cb.currentIndexChanged.connect(lambda _v, i=i: self._on_coord_select(i))

        # Any cursor drag (force / encoder / angle) seeks playback
        self._timeline.seek_requested.connect(self._controller.seek)
        self._enc_timeline.seek_requested.connect(self._controller.seek)
        self._angles.seek_requested.connect(self._controller.seek)

        self._controller.time_changed.connect(self._on_time_changed)
        self._controller.state_changed.connect(self._on_state_changed)

        self._slider_dragging = False

        # ── Keyboard shortcuts (scoped to the ReplayPanel widget) ─────────
        # The ReportsTab handles Escape → back to browse; we handle the
        # playback keys here. Shortcut scope is WidgetWithChildrenShortcut
        # so keys don't leak when the user is typing in some unrelated
        # QLineEdit elsewhere in the app.
        def sc(key, handler):
            QShortcut(QKeySequence(key), self,
                      activated=handler,
                      context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

        sc("Space",      self._controller.toggle)
        sc("Left",       lambda: self._seek_relative(-1.0))
        sc("Right",      lambda: self._seek_relative(+1.0))
        sc("Shift+Left", lambda: self._seek_relative(-10.0))
        sc("Shift+Right",lambda: self._seek_relative(+10.0))
        sc("Home",       lambda: self._controller.seek(0.0))
        sc("End",        lambda: self._controller.seek(self._controller.duration_s))
        # Playback-speed shortcuts: 1..5 → 0.25×, 0.5×, 1×, 2×, 4×.
        # (indices into the speed combobox in _build_ui order)
        for i, key in enumerate(["1", "2", "3", "4", "5"]):
            sc(key, lambda idx=i: self._set_speed_by_index(idx))
        sc("K",          lambda: self._overlay.toggle() if self._overlay.isEnabled() else None)
        # Departure event navigation (Phase U3-3)
        sc("[",          self._seek_prev_departure)
        sc("]",          self._seek_next_departure)

    def _seek_relative(self, delta_s: float) -> None:
        dur = self._controller.duration_s
        if dur <= 0:
            return
        cur = float(self._controller.current_t_s)
        new_t = max(0.0, min(dur, cur + float(delta_s)))
        self._controller.seek(new_t)

    def _set_speed_by_index(self, idx: int) -> None:
        if 0 <= idx < self._speed.count():
            self._speed.setCurrentIndex(idx)

    # ── Departure navigation (Phase U3-3) ──────────────────────────────────
    def _seek_next_departure(self) -> None:
        """Jump to the next off-plate event after the current cursor.

        Wraps to the first event if cursor is past the last one. No-op
        when no departure events exist.
        """
        if not self._departure_times:
            return
        cur = float(self._controller.current_t_s)
        # Find the smallest event time strictly greater than current
        for t in self._departure_times:
            if t > cur + 1e-3:
                self._controller.seek(t)
                return
        # Fell off the end — wrap to first
        self._controller.seek(self._departure_times[0])

    def _seek_prev_departure(self) -> None:
        if not self._departure_times:
            return
        cur = float(self._controller.current_t_s)
        # Find the largest event time strictly less than current
        for t in reversed(self._departure_times):
            if t < cur - 1e-3:
                self._controller.seek(t)
                return
        self._controller.seek(self._departure_times[-1])

    # ── public API ─────────────────────────────────────────────────────────
    def load_session(self, session_dir: str | Path,
                     cam_id: Optional[str] = None) -> bool:
        """Load a session for replay. Returns True if anything loaded."""
        self.unload()
        sd = Path(session_dir)
        if cam_id is None:
            cams = config.CAMERAS
            cam_id = cams[0]["id"] if cams else "C0"
        self._session_dir = sd
        self._cam_id = cam_id

        # Session meta — decides encoder UI state (balance / uses_encoder)
        meta = {}
        meta_path = sd / "session.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        self._enc_active = _encoder_active_for_meta(meta)

        force_ok = self._timeline.load(sd)
        self._cop.load(sd)
        video_ok = self._video.load(sd, cam_id)

        # Encoder timeline: load data always, then toggle the "비활성"
        # overlay based on session meta. This keeps the layout stable
        # regardless of test type.
        self._enc_timeline.load(sd)
        self._enc_timeline.set_available(self._enc_active)

        # Cache ForceSession for live encoder-bar updates during playback.
        try:
            self._force = load_force_session(sd)
        except Exception:
            self._force = None

        # Encoder bars: honour both per-session active state AND config.
        # When inactive, EncoderBar paints the "비활성" placeholder.
        self._enc1_bar.set_available(
            self._enc_active and bool(getattr(config, "ENCODER1_AVAILABLE", True)))
        self._enc2_bar.set_available(
            self._enc_active and bool(getattr(config, "ENCODER2_AVAILABLE", False)))

        # Pose series (optional)
        pose_map = load_session_pose2d(sd)
        self._pose = pose_map.get(cam_id)
        pose_available = self._pose is not None
        if pose_available:
            self._angles.load(sd, cam_id, self._pose)
            self._coord.load(self._pose)
        self._update_pose_selector_enabled(pose_available)

        # Duration: force timeline is master
        dur = self._timeline.duration_s() or self._video.duration_s()
        self._controller.set_duration(dur)
        self._controller.seek(0.0)
        self._time_label.setText(f"0.00 / {dur:.2f} s")

        # Skeleton overlay availability
        self._overlay.setEnabled(self._video.has_pose())
        if not self._video.has_pose():
            self._overlay.setChecked(False)

        # ── Departure events (Phase U3-3) ─────────────────────────────
        # Read events.csv and broadcast intervals to every timeline +
        # the slider tick overlay + the summary badge. Sessions that
        # never had a departure (or were recorded pre-U3-3) get an
        # empty list, so the visualisation gracefully renders nothing.
        self._apply_departures(sd, dur, meta)

        return force_ok or video_ok

    def _apply_departures(self, session_dir: Path,
                          duration_s: float, meta: dict) -> None:
        """Compute departure events on the fly + broadcast to UI.

        Recomputed from the loaded ForceSession's total_n using the
        current ``DEPARTURE_THRESHOLD_N`` (20 N — matches CMJ analyser).
        Sessions recorded under a previous threshold automatically get
        their bands realigned to the current definition, so the replay
        view always agrees with the analysis charts. The stored
        events.csv is preserved as a raw-data artefact for offline
        analysis pipelines that want the recording-time labels.
        """
        events: list[dict] = []
        if self._force is not None:
            try:
                events = compute_departure_events(self._force)
            except Exception:
                events = []
        intervals: list[tuple[float, float]] = [
            (ev["t_start_s"], ev["t_end_s"]) for ev in events
        ]
        # Push to every timeline widget that supports the API
        try: self._timeline.set_departures(intervals)
        except Exception: pass
        try: self._enc_timeline.set_departures(intervals)
        except Exception: pass
        try: self._angles.set_departures(intervals)
        except Exception: pass
        # Slider tick overlay + cache start times for [/] navigation
        starts = sorted(ev["t_start_s"] for ev in events)
        self._departure_times = starts
        try:
            self._slider.set_departure_ticks(starts, duration_s)
        except Exception:
            pass
        # Badge — recompute summary directly from the (re)derived events
        # so it matches the bands the user is looking at, never the
        # potentially-stale meta block from session.json.
        n_events = len(events)
        total_off = sum(ev["duration_s"] for ev in events)
        longest = max((ev["duration_s"] for ev in events), default=0.0)
        if n_events <= 0:
            self._departure_badge.setVisible(False)
        else:
            self._departure_badge.setText(
                f"⚠ 이탈 {n_events}회   총 {total_off:.2f}s   "
                f"최장 {longest:.2f}s"
            )
            self._departure_badge.setVisible(True)

    def unload(self) -> None:
        self._controller.pause()
        self._controller.seek(0.0)
        self._video.unload()
        self._cop.unload()
        self._coord.clear()
        self._timeline.unload()
        self._enc_timeline.unload()
        self._angles.clear()
        # Clear departure visualisation
        try: self._slider.set_departure_ticks([], 0.0)
        except Exception: pass
        try: self._departure_badge.setVisible(False)
        except Exception: pass
        self._departure_times: list[float] = []
        self._session_dir = None
        self._cam_id = None
        self._pose = None
        self._force = None
        self._enc_active = True
        # Reset encoder bars — default to config flags until next load
        self._enc1_bar.set_available(
            bool(getattr(config, "ENCODER1_AVAILABLE", True)))
        self._enc2_bar.set_available(
            bool(getattr(config, "ENCODER2_AVAILABLE", False)))
        self._enc1_bar.set_value(None)
        self._enc2_bar.set_value(None)
        self._overlay.setChecked(False)
        self._overlay.setEnabled(False)
        # Reset selector combos silently
        for cb in self._angle_selects + self._coord_selects:
            cb.blockSignals(True)
            cb.setCurrentIndex(0)
            cb.blockSignals(False)
        self._update_pose_selector_enabled(False)

    # ── selectors ──────────────────────────────────────────────────────────
    def _update_pose_selector_enabled(self, pose_available: bool) -> None:
        for cb in self._angle_selects + self._coord_selects:
            cb.setEnabled(pose_available)
        self._pose_hint.setVisible(not pose_available)

    def _on_angle_select(self, slot: int) -> None:
        name = self._angle_selects[slot].currentData()
        self._angles.set_angle(slot, name)

    def _on_coord_select(self, slot: int) -> None:
        name = self._coord_selects[slot].currentData()
        self._coord.set_joint(slot, name)

    # ── signal handlers ────────────────────────────────────────────────────
    def _on_time_changed(self, t_s: float) -> None:
        self._video.set_time(t_s)
        self._cop.set_time(t_s)
        self._timeline.set_cursor(t_s)
        self._enc_timeline.set_cursor(t_s)
        self._angles.set_cursor(t_s)
        # Coord trail uses the same frame index as video
        if self._pose is not None and self._session_dir and self._cam_id:
            frame_idx = resolve_pose_frame(
                float(t_s), self._session_dir, self._cam_id, self._pose.fps)
            self._coord.set_frame_index(frame_idx)

        # Encoder bars — index force.enc1/enc2 at cursor time so the bar
        # animates exactly like it did during recording. Only update when
        # the channel is active (otherwise the bar shows "비활성").
        if self._force is not None and len(self._force.t_s) > 0 and self._enc_active:
            idx = int(np.clip(
                np.searchsorted(self._force.t_s, t_s),
                0, len(self._force.t_s) - 1))
            if self._enc1_bar.is_available():
                self._enc1_bar.set_value(float(self._force.enc1[idx]))
            if self._enc2_bar.is_available():
                self._enc2_bar.set_value(float(self._force.enc2[idx]))

        dur = self._controller.duration_s
        self._time_label.setText(f"{t_s:.2f} / {dur:.2f} s")
        if not self._slider_dragging and dur > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(1000 * t_s / dur))
            self._slider.blockSignals(False)

    def _on_state_changed(self, playing: bool) -> None:
        self._btn_play.setText("⏸" if playing else "▶")

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._apply_slider_position()

    def _on_slider_changed(self, _v: int) -> None:
        if self._slider_dragging:
            self._apply_slider_position()

    def _apply_slider_position(self) -> None:
        dur = self._controller.duration_s
        if dur <= 0:
            return
        frac = self._slider.value() / 1000.0
        self._controller.seek(frac * dur)

    def _on_speed_changed(self, _i: int) -> None:
        s = self._speed.currentData()
        if s is not None:
            self._controller.set_speed(float(s))
