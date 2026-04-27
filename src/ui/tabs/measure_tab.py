"""
Measure tab — test selection, live recording, and trainer dashboard.

Layout (Phase 2):

    ┌─────────────────────────────────────────────────────────────────┐
    │ 측정 대상: 홍길동 (6c12a11e) · 95kg · 181cm · 트레이너: 김코치      │
    ├──────────────────────┬──────────────────────────────────────────┤
    │  옵션 / 제어          │   피험자 카메라 뷰 (상단 3분할)             │
    │  ─────────            │   ─────────────────────────                │
    │  테스트 선택           │   ... CameraView x3 ...                    │
    │  측정 시간 / 옵션      │                                             │
    │  [시작] [중지]         │   트레이너 대시보드 (하단)                  │
    │                       │   ─────────────                             │
    │  로그                  │   Total / Per-board / CoP / rolling force  │
    │  [tail log ...]       │                                             │
    └──────────────────────┴──────────────────────────────────────────┘

Phase 2 sub-steps:
  2.3 — UI shell (this file) with camera/dashboard placeholders
  2.4 — fill CameraView widgets
  2.5 — fill ForceDashboard
  2.6 — fill StabilityOverlay (wait phase)
  2.7 — DB session wiring
  2.8 — protocol checklist mode
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QSplitter, QGroupBox, QPlainTextEdit, QMessageBox, QSizePolicy,
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
    QScrollArea,
)

import config
from src.db import models as db_models
from src.db.models import Session as DbSession, Subject
from src.capture.session_recorder import RecorderConfig, RecorderState
from src.ui.widgets.test_options_panel import TestOptionsPanel
from src.ui.widgets.camera_view import CameraView
from src.ui.widgets.force_dashboard import ForceDashboard
from src.ui.widgets.stability_overlay import StabilityOverlay
from src.ui.widgets.protocol_header import ProtocolHeader, TEST_KO
from src.ui.workers.record_worker import RecordWorker
from src.ui.workers.analysis_worker import AnalysisWorker
from src.ui.workers.pose_worker import PoseWorker
from src.ui.workers.pose_live_worker import PoseLiveWorker
from src.analysis.encoder import RealtimeRepCounter


# Seconds between tests in protocol mode — long enough for subject to step
# off the plate before the next DAQ zero-cal begins.
BETWEEN_TEST_PAUSE_S = 8.0


def _encoder_active_for_test(test_type: str, uses_encoder: bool) -> bool:
    """Encoder UI is active iff the test is not a balance test AND the
    session opted into encoder usage. Shared by Measure (live) and Replay
    (playback) so the rule stays consistent."""
    if test_type in ("balance_eo", "balance_ec"):
        return False
    return bool(uses_encoder)


class MeasureTab(QWidget):
    """Host widget for test selection + live recording + trainer dashboard."""

    session_completed = pyqtSignal(dict)   # result summary

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_subject: Optional[Subject] = None
        self._worker: Optional[RecordWorker] = None
        self._starting: bool = False          # double-click defense
        # Protocol state
        self._protocol_queue: list[dict] = []
        self._protocol_idx: int = 0
        self._protocol_results: list[dict] = []
        self._abort_protocol: bool = False
        # Currently running config snapshot (used for overlay instead of UI)
        self._current_opts: Optional[dict] = None
        # Between-test countdown
        self._transition_timer: Optional[QTimer] = None
        self._transition_end_ns: int = 0
        # Active analysis workers, keyed by session_id so we can update DB
        # status when they finish. They run after recording so multiple
        # analyses may be in-flight concurrently.
        self._analysis_workers: dict[str, AnalysisWorker] = {}
        self._pose_workers:     dict[str, PoseWorker]     = {}
        # Realtime pose overlay worker (at most one per live session)
        self._pose_live_worker: Optional[PoseLiveWorker]  = None
        # Live rep counter — created per free_exercise recording, None otherwise
        self._rep_counter: Optional[RealtimeRepCounter] = None
        self._build_ui()
        self._refresh_controls()

    # ── public API ──────────────────────────────────────────────────────────
    def set_active_subject(self, subject: Subject) -> None:
        self._active_subject = subject
        self._subject_title.setText(
            f"측정 대상: {subject.name} ({subject.id})"
        )
        self._subject_details.setText(
            f"{subject.weight_kg:.1f} kg · {subject.height_cm:.1f} cm · "
            f"{subject.gender or '—'}  |  트레이너: {subject.trainer or '—'}"
        )
        self._subject_panel.setStyleSheet(
            "QFrame { background:#14371C; border:1px solid #AAF500; "
            "border-radius:10px; }"
        )
        self._refresh_controls()

    # ── UI build ───────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── TOP BAR ───────────────────────────────────────────────────────
        # Layout goals:
        #   - FITWIN 가로 로고는 AppWindow 탭바 우상단 corner widget으로
        #     이미 모든 탭 공용. Measure 탭 내부에는 별도 로고 없음.
        #   - 3개 주요 영역을 균형 배분:
        #       [≡] [피험자 패널 : 3]  [상태 패널 : 4]  [액션 버튼 묶음 : 3]
        #     상태 패널이 가장 크게 (4) — wait / countdown / recording 등
        #     실시간 정보가 가장 많이 변하는 영역.
        from src.ui.widgets.sidebar_toggle import make_toggle_button
        top = QHBoxLayout()
        top.setSpacing(10)
        top.setContentsMargins(0, 0, 0, 0)

        # Sidebar collapse toggle (leftmost)
        self._sidebar_toggle = make_toggle_button("설정 패널 접기 / 펼치기")
        top.addWidget(self._sidebar_toggle, 0)

        # Subject info card — compact branded panel (stretch 3).
        #
        # U2-1: Removed CSS ``padding`` — Qt's stylesheet box model and
        # QVBoxLayout's contentsMargins were stacking, eating ~12-16 px
        # of vertical space at non-100% DPI. Now only the layout margin
        # controls inner spacing → text always has full headroom.
        # Title font 12pt → 11pt (16 px line) keeps two-line content
        # comfortable inside 82-100 px panel height.
        self._subject_panel = QFrame()
        self._subject_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self._subject_panel.setStyleSheet(
            "QFrame { background:#1C1C1E; border:1px solid #2C2C2E; "
            "border-radius:10px; }"
        )
        self._subject_panel.setMinimumHeight(82)
        self._subject_panel.setMaximumHeight(100)
        sp = QVBoxLayout(self._subject_panel)
        sp.setContentsMargins(14, 10, 14, 10); sp.setSpacing(4)
        self._subject_title = QLabel("측정 대상: (미선택)")
        f = QFont(); f.setPointSize(11); f.setBold(True)
        self._subject_title.setFont(f)
        self._subject_details = QLabel(
            "Subjects 탭에서 '측정 대상으로 선택' 후 여기로 돌아오세요."
        )
        self._subject_details.setStyleSheet(
            "color:#A1A1A6; font-size:11px;")
        self._subject_details.setWordWrap(True)
        sp.addWidget(self._subject_title)
        sp.addWidget(self._subject_details)
        top.addWidget(self._subject_panel, 3)

        # Status / stability panel (stretch 4) — largest lane because this
        # is where the most time-critical information appears. Height
        # range matches the subject panel + action cluster so the top
        # bar reads as one cohesive row.
        self._overlay = StabilityOverlay()
        self._overlay.setMinimumHeight(82)
        self._overlay.setMaximumHeight(100)
        top.addWidget(self._overlay, 4)

        # Action cluster (stretch 3) — Start (primary) + 2 stop buttons in
        # their own sub-row so they don't fight the other panels for width.
        action_wrap = QWidget()
        action_wrap.setMinimumHeight(82)
        action_wrap.setMaximumHeight(100)
        action_lay = QHBoxLayout(action_wrap)
        action_lay.setContentsMargins(0, 0, 0, 0)
        action_lay.setSpacing(6)

        self._btn_start = QPushButton("▶ 측정 시작")
        self._btn_start.setProperty("kind", "primary")
        self._btn_start.setMinimumHeight(60)
        self._btn_start.clicked.connect(self._on_start_clicked)
        self._btn_start.setToolTip("측정을 시작합니다 (단축키 F5)")

        self._btn_stop = QPushButton("■ 중지")
        self._btn_stop.setProperty("kind", "danger")
        self._btn_stop.setMinimumHeight(60)
        self._btn_stop.setMinimumWidth(80)
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip(
            "현재 테스트를 중지합니다 (단축키 Esc).\n"
            "프로토콜 모드에서는 현재 테스트만 취소하고 다음으로 진행합니다.")

        self._btn_abort = QPushButton("■■ 전체중지")
        self._btn_abort.setProperty("kind", "danger")
        self._btn_abort.setMinimumHeight(60)
        self._btn_abort.setMinimumWidth(88)
        self._btn_abort.clicked.connect(self._on_abort_clicked)
        self._btn_abort.setEnabled(False)
        self._btn_abort.setToolTip("프로토콜 전체를 중단합니다 (단축키 Shift+Esc).")

        action_lay.addWidget(self._btn_start, 2)
        action_lay.addWidget(self._btn_stop,  1)
        action_lay.addWidget(self._btn_abort, 1)
        top.addWidget(action_wrap, 3)

        root.addLayout(top)

        # ── MAIN SPLITTER: [sidebar | center | right] ─────────────────────
        self._main_split = QSplitter(Qt.Orientation.Horizontal)
        self._main_split.setChildrenCollapsible(True)
        # T1: Wrap sidebar in QScrollArea so reaction-time / free-exercise
        # option groups stay reachable on 1440-px-wide screens.
        sidebar_inner = self._build_sidebar()
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(sidebar_inner)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sidebar_scroll.setMinimumWidth(280)

        self._main_split.addWidget(sidebar_scroll)            # 0
        self._main_split.addWidget(self._build_center())     # 1
        self._main_split.addWidget(self._build_right())      # 2
        self._main_split.setStretchFactor(0, 0)
        self._main_split.setStretchFactor(1, 3)
        self._main_split.setStretchFactor(2, 2)
        self._main_split.setSizes([320, 700, 500])
        root.addWidget(self._main_split, stretch=1)

        # Wire the toggle
        from src.ui.widgets.sidebar_toggle import attach_toggle
        attach_toggle(self._sidebar_toggle, self._main_split,
                       panel_index=0, expanded_size=320)

        # ── Keyboard shortcuts ────────────────────────────────────────────
        # All shortcuts are scoped to this widget (parent=self) so they
        # don't fire when Reports / Subjects tab is the active view. Each
        # handler additionally checks enablement state to mirror button
        # gating exactly (a shortcut should be a no-op when the equivalent
        # button is disabled).
        self._register_shortcuts()

    def _register_shortcuts(self) -> None:
        def sc(key, handler):
            QShortcut(QKeySequence(key), self,
                      activated=handler,
                      context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

        sc("F5",         self._shortcut_start)
        sc("Escape",     self._shortcut_stop)
        sc("Shift+Escape", self._shortcut_abort)
        # Reaction-time manual triggers — same number/letter mapping as
        # the button labels so trainers can reliably fire stimuli without
        # taking their hand off the keyboard.
        sc("1",          lambda: self._shortcut_reaction("left_shift"))
        sc("2",          lambda: self._shortcut_reaction("right_shift"))
        sc("3",          lambda: self._shortcut_reaction("jump"))
        sc("Space",      self._shortcut_reaction_random)
        # Squat VRT cue — U = "Up"
        sc("U",          self._shortcut_squat_vrt_cue)
        # Multi-set strength assessment (Phase V1-E)
        sc("S",          self._shortcut_set_end)        # Set end
        sc("E",          self._shortcut_session_end)    # End session
        sc("P",          self._shortcut_rest_pause)     # Pause/resume rest
        sc("N",          self._shortcut_rest_skip)      # Next set (skip rest)

    def _shortcut_start(self) -> None:
        if self._btn_start.isEnabled():
            self._on_start_clicked()

    def _shortcut_stop(self) -> None:
        if self._btn_stop.isEnabled():
            self._on_stop_clicked()

    def _shortcut_abort(self) -> None:
        if self._btn_abort.isEnabled():
            self._on_abort_clicked()

    def _shortcut_reaction(self, response: str) -> None:
        # Only active when the reaction manual buttons are enabled — that
        # gating already covers "recording + reaction test selected".
        if self._btn_left.isEnabled():      # all reaction buttons share state
            self._fire_manual(response)

    def _shortcut_reaction_random(self) -> None:
        if self._btn_rand.isEnabled():
            self._fire_manual_random()

    def _shortcut_squat_vrt_cue(self) -> None:
        if self._btn_vrt_cue.isEnabled():
            self._fire_manual("squat_ascent")

    # ── Multi-set strength shortcuts (Phase V1-E) ─────────────────────────
    def _shortcut_set_end(self) -> None:
        if self._btn_set_end.isEnabled():
            self._on_set_end_clicked()

    def _shortcut_session_end(self) -> None:
        if self._btn_session_end.isEnabled():
            self._on_session_end_clicked()

    def _shortcut_rest_pause(self) -> None:
        if self._btn_rest_pause.isEnabled():
            self._on_rest_pause_clicked()

    def _shortcut_rest_skip(self) -> None:
        if self._btn_rest_skip.isEnabled():
            self._on_rest_skip_clicked()

    def _build_sidebar(self) -> QWidget:
        """Left sidebar — mode / test options / protocol queue / manual / log.
        Collapsible via the top-bar ≡ button. Wrapped in a QScrollArea
        (T1) so that test modes with many option groups (reaction, free
        exercise) don't overflow the visible area on 1440px-wide
        screens. Minimum width (T5) keeps the option labels legible."""
        w = QWidget()
        w.setMinimumWidth(280)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Mode toggle: 단일 / 프로토콜
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("모드"))
        self._mode_single   = QRadioButton("단일")
        self._mode_protocol = QRadioButton("프로토콜")
        self._mode_single.setChecked(True)
        self._mode_group = QButtonGroup(self)
        for b in (self._mode_single, self._mode_protocol):
            self._mode_group.addButton(b)
            mode_row.addWidget(b)
        mode_row.addStretch(1)
        self._mode_single.toggled.connect(self._on_mode_toggled)
        lay.addLayout(mode_row)

        self._options = TestOptionsPanel()
        lay.addWidget(self._options)

        # Protocol queue (hidden in single mode)
        self._protocol_box = QGroupBox("프로토콜 순서")
        pl = QVBoxLayout(self._protocol_box)
        self._queue_list = QListWidget()
        self._queue_list.setStyleSheet(
            "QListWidget { background:#1a1a1a; color:#ddd; }")
        self._queue_list.setMaximumHeight(140)
        pl.addWidget(self._queue_list)
        qbtn = QHBoxLayout()
        self._btn_add    = QPushButton("+ 추가")
        self._btn_remove = QPushButton("- 제거")
        self._btn_up     = QPushButton("↑")
        self._btn_down   = QPushButton("↓")
        self._btn_clear  = QPushButton("초기화")
        self._btn_add.setToolTip("현재 옵션 패널의 테스트를 큐 끝에 추가")
        self._btn_remove.setToolTip("선택한 큐 항목 제거")
        self._btn_up.setToolTip("선택한 항목을 한 칸 위로")
        self._btn_down.setToolTip("선택한 항목을 한 칸 아래로")
        self._btn_clear.setToolTip("큐 전체 비우기")
        self._btn_add.clicked.connect(self._add_to_queue)
        self._btn_remove.clicked.connect(self._remove_from_queue)
        self._btn_up.clicked.connect(lambda: self._move_item(-1))
        self._btn_down.clicked.connect(lambda: self._move_item(1))
        self._btn_clear.clicked.connect(self._clear_queue)
        for b in (self._btn_add, self._btn_remove,
                  self._btn_up, self._btn_down, self._btn_clear):
            qbtn.addWidget(b)
        pl.addLayout(qbtn)
        self._protocol_box.setVisible(False)
        lay.addWidget(self._protocol_box)

        # Manual reaction buttons (only visible when reaction selected)
        self._manual_box = QGroupBox("수동 트리거 (반응시간)")
        mr = QHBoxLayout(self._manual_box)
        self._btn_left  = QPushButton("1 · 좌측")
        self._btn_right = QPushButton("2 · 우측")
        self._btn_jump  = QPushButton("3 · 점프")
        self._btn_rand  = QPushButton("SPACE")
        self._btn_left.setToolTip("좌측 이동 자극 트리거 (키보드 1)")
        self._btn_right.setToolTip("우측 이동 자극 트리거 (키보드 2)")
        self._btn_jump.setToolTip("점프 자극 트리거 (키보드 3)")
        self._btn_rand.setToolTip("랜덤 자극 트리거 (키보드 Space)")
        self._btn_left.clicked.connect(lambda: self._fire_manual("left_shift"))
        self._btn_right.clicked.connect(lambda: self._fire_manual("right_shift"))
        self._btn_jump.clicked.connect(lambda: self._fire_manual("jump"))
        self._btn_rand.clicked.connect(self._fire_manual_random)
        for b in (self._btn_left, self._btn_right, self._btn_jump, self._btn_rand):
            b.setEnabled(False)
            mr.addWidget(b)
        self._manual_box.setVisible(False)
        lay.addWidget(self._manual_box)

        # Squat VRT cue — bottom-of-descent trigger for visual-response
        # RFD measurement (patent 2 §4, Phase S1d). One big button because
        # the trainer presses it per rep; keyboard shortcut U is wired in
        # _register_shortcuts below.
        self._vrt_box = QGroupBox("시각-반응 RFD 큐 (스쿼트)")
        vlay = QHBoxLayout(self._vrt_box)
        self._btn_vrt_cue = QPushButton("⚡  UP 신호 (U)")
        self._btn_vrt_cue.setProperty("kind", "primary")
        self._btn_vrt_cue.setMinimumHeight(48)
        self._btn_vrt_cue.setToolTip(
            "피험자가 스쿼트 최대 하강선을 통과했을 때 누르세요 — "
            "시각/청각 상승 신호와 함께 VRT 측정용 stim 시각이 기록됩니다. "
            "단축키: U")
        self._btn_vrt_cue.clicked.connect(
            lambda: self._fire_manual("squat_ascent"))
        self._btn_vrt_cue.setEnabled(False)
        vlay.addWidget(self._btn_vrt_cue)
        self._vrt_box.setVisible(False)
        lay.addWidget(self._vrt_box)

        # ── Multi-set strength assessment (Phase V1-E) ─────────────────
        # Visible only for ``strength_3lift``. Two layouts share the
        # same group box, swapped by _refresh_strength_controls based
        # on the current state.phase:
        #   recording        → "세트 X / N" + "세트 종료" + "종료"
        #   inter_set_rest   → countdown + pause/resume + skip + end
        self._strength_box = QGroupBox("3대 운동 다세트")
        sblay = QVBoxLayout(self._strength_box)
        sblay.setSpacing(6)
        # Set indicator label (always visible inside the box)
        self._lbl_set_indicator = QLabel("준비 중")
        self._lbl_set_indicator.setStyleSheet(
            "QLabel { color:#FFEB3B; font-weight:bold; font-size:13px; "
            "padding:4px; }")
        self._lbl_set_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sblay.addWidget(self._lbl_set_indicator)
        # Buttons row 1 — recording-phase controls
        rec_row = QHBoxLayout()
        rec_row.setSpacing(6)
        self._btn_set_end = QPushButton("✓ 세트 종료 (S)")
        self._btn_set_end.setProperty("kind", "primary")
        self._btn_set_end.setMinimumHeight(48)
        self._btn_set_end.setToolTip(
            "현재 세트를 종료하고 휴식 단계로 전환합니다 (단축키 S).\n"
            "마지막 세트일 경우 측정이 완료됩니다.")
        self._btn_set_end.clicked.connect(self._on_set_end_clicked)
        self._btn_set_end.setEnabled(False)
        rec_row.addWidget(self._btn_set_end, 2)
        self._btn_session_end = QPushButton("◼ 세션 종료 (E)")
        self._btn_session_end.setMinimumHeight(48)
        self._btn_session_end.setToolTip(
            "남은 세트를 건너뛰고 지금까지의 데이터로 측정을 종료합니다.\n"
            "취소와 달리 기록은 유지됩니다.")
        self._btn_session_end.clicked.connect(self._on_session_end_clicked)
        self._btn_session_end.setEnabled(False)
        rec_row.addWidget(self._btn_session_end, 1)
        sblay.addLayout(rec_row)
        # Buttons row 2 — rest-phase controls
        rest_row = QHBoxLayout()
        rest_row.setSpacing(6)
        self._btn_rest_pause = QPushButton("⏸ 일시정지")
        self._btn_rest_pause.setMinimumHeight(40)
        self._btn_rest_pause.setToolTip(
            "휴식 카운트다운을 일시정지합니다. 다시 누르면 재개합니다.")
        self._btn_rest_pause.clicked.connect(self._on_rest_pause_clicked)
        self._btn_rest_pause.setEnabled(False)
        self._btn_rest_skip = QPushButton("⏭ 다음 세트 시작")
        self._btn_rest_skip.setMinimumHeight(40)
        self._btn_rest_skip.setToolTip(
            "휴식을 건너뛰고 다음 세트를 즉시 시작합니다.")
        self._btn_rest_skip.clicked.connect(self._on_rest_skip_clicked)
        self._btn_rest_skip.setEnabled(False)
        rest_row.addWidget(self._btn_rest_pause, 1)
        rest_row.addWidget(self._btn_rest_skip, 1)
        sblay.addLayout(rest_row)
        self._strength_box.setVisible(False)
        lay.addWidget(self._strength_box)

        # Log
        log_box = QGroupBox("로그")
        llay = QVBoxLayout(log_box)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setStyleSheet(
            "QPlainTextEdit { background:#111; color:#ccc; "
            "font-family: Consolas, monospace; font-size: 11px; }"
        )
        llay.addWidget(self._log)
        lay.addWidget(log_box, stretch=1)

        # React to test changes to toggle manual-box visibility
        self._options.test_changed.connect(self._on_test_changed)
        return w

    def _build_center(self) -> QWidget:
        """Center column — top overlays + camera video with encoder bars
        flanking on both sides.

        Minimum width (T5) ensures the camera + encoder bars stay
        legible even when the side panels are dragged narrow.
        """
        from src.ui.widgets.encoder_bar import EncoderBar
        import config as _cfg

        w = QWidget()
        w.setMinimumWidth(360)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Protocol progress header (only visible in protocol mode)
        self._protocol_header = ProtocolHeader()
        lay.addWidget(self._protocol_header)

        # Camera + encoder bars row
        cam_row = QHBoxLayout()
        cam_row.setSpacing(4)
        self._enc1_bar = EncoderBar(
            label="L (enc1)",
            max_mm=float(getattr(_cfg, "ENCODER_MAX_DISPLAY_MM", 2000)),
            available=bool(getattr(_cfg, "ENCODER1_AVAILABLE", True)),
        )
        self._enc2_bar = EncoderBar(
            label="R (enc2)",
            max_mm=float(getattr(_cfg, "ENCODER_MAX_DISPLAY_MM", 2000)),
            available=bool(getattr(_cfg, "ENCODER2_AVAILABLE", False)),
        )
        self._camera_view = CameraView()
        self._camera_view.setMinimumHeight(380)
        # T8: encoder bars stay at fixed-ish narrow widths; the camera
        # claims its preferred portrait aspect (set in CameraView). The
        # surrounding stretches absorb any leftover horizontal space so
        # the camera doesn't get over-stretched into wide letterbox.
        cam_row.addWidget(self._enc1_bar, 0)
        cam_row.addStretch(1)
        cam_row.addWidget(self._camera_view, 0,
                          Qt.AlignmentFlag.AlignHCenter)
        cam_row.addStretch(1)
        cam_row.addWidget(self._enc2_bar, 0)
        lay.addLayout(cam_row, stretch=1)
        return w

    def _build_right(self) -> QWidget:
        """Right column — CoP (top) + VGRF (bottom), stacked vertically.

        Minimum width (T5) is 380 px which is the smallest size that
        keeps the full plate footprint (558 mm × 432 mm = 1.29:1) and
        both Board1/Board2 outlines visible without aspect-lock
        cropping the right edge.
        """
        w = QWidget()
        w.setMinimumWidth(380)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._dashboard = ForceDashboard(orientation="vertical")
        self._dashboard.setMinimumWidth(360)
        lay.addWidget(self._dashboard, stretch=1)
        return w

    # ── event handlers ─────────────────────────────────────────────────────
    def _on_mode_toggled(self, single_on: bool) -> None:
        self._protocol_box.setVisible(not single_on)

    def _on_test_changed(self, test_key: str) -> None:
        # Manual reaction buttons box — visible for classic reaction
        # AND V6 cognitive_reaction (Space = random positional cue).
        self._manual_box.setVisible(test_key in ("reaction", "cognitive_reaction"))
        # Squat / overhead squat expose the VRT cue button (Phase S1d).
        self._vrt_box.setVisible(test_key in ("squat", "overhead_squat"))
        # Strength 3-lift exposes the multi-set control box (Phase V1-E).
        self._strength_box.setVisible(test_key == "strength_3lift")
        # T6: force a relayout pass — without this, switching modes
        # while the window is at minimum width can leave option boxes
        # partially clipped until the next resize event.
        self.updateGeometry()
        if hasattr(self, "_main_split") and self._main_split is not None:
            self._main_split.refresh()

    # ── Multi-set strength assessment handlers (Phase V1-E) ────────────────
    def _on_set_end_clicked(self) -> None:
        """Operator pressed "세트 종료" — forward to the recorder.
        No-op when no recording or when not strength_3lift."""
        if self._worker is not None:
            self._worker.end_set()
            self._append_log("[ui] set end requested")

    def _on_session_end_clicked(self) -> None:
        """Operator pressed "세션 종료" — finalize early, preserving data
        already recorded. Distinct from cancel (which discards)."""
        if self._worker is not None:
            self._worker.end_session()
            self._append_log("[ui] session end requested")

    def _on_rest_pause_clicked(self) -> None:
        """Toggle pause/resume on the inter-set rest countdown."""
        if self._worker is None:
            return
        # Use the button's current label to decide which way to toggle.
        # _refresh_strength_controls keeps the label in sync with state.
        if self._btn_rest_pause.text().startswith("▶"):
            self._worker.resume_rest()
            self._append_log("[ui] rest resumed")
        else:
            self._worker.pause_rest()
            self._append_log("[ui] rest paused")

    def _on_rest_skip_clicked(self) -> None:
        if self._worker is not None:
            self._worker.skip_rest()
            self._append_log("[ui] rest skipped")

    def _refresh_strength_controls(self, st: RecorderState) -> None:
        """Drive the strength-box buttons + indicator from RecorderState.

        Called from _on_state on every state update. Decides which row
        of buttons is enabled/labelled based on the current phase.
        """
        if not self._strength_box.isVisible():
            return
        # Indicator label
        n = max(st.n_sets, 1)
        idx = st.current_set_idx + 1   # 1-based for display
        if st.phase == "recording":
            warmup_tag = ""
            opts = self._current_opts or {}
            if opts.get("warmup_set") and st.current_set_idx == 0:
                warmup_tag = "  (워밍업)"
            self._lbl_set_indicator.setText(
                f"● 세트 {idx} / {n} 측정 중{warmup_tag}")
            self._lbl_set_indicator.setStyleSheet(
                "QLabel { color:#FF5252; font-weight:bold; font-size:14px; "
                "padding:4px; }")
            self._btn_set_end.setEnabled(True)
            self._btn_session_end.setEnabled(True)
            self._btn_rest_pause.setEnabled(False)
            self._btn_rest_skip.setEnabled(False)
            self._btn_rest_pause.setText("⏸ 일시정지")
        elif st.phase == "inter_set_rest":
            self._lbl_set_indicator.setText(
                f"⏱ 세트 {idx} / {n} 종료   휴식 {st.rest_remaining_s:.1f} s   "
                f"다음: 세트 {min(idx + 1, n)} / {n}")
            self._lbl_set_indicator.setStyleSheet(
                "QLabel { color:#FFD54F; font-weight:bold; font-size:14px; "
                "padding:4px; }")
            self._btn_set_end.setEnabled(False)
            self._btn_session_end.setEnabled(True)
            self._btn_rest_pause.setEnabled(True)
            self._btn_rest_skip.setEnabled(True)
            self._btn_rest_pause.setText("▶ 재개" if st.rest_paused
                                          else "⏸ 일시정지")
        else:
            # idle / wait / countdown / done / cancelled — disable all
            # multi-set controls until recording starts.
            self._lbl_set_indicator.setText("측정 시작 대기")
            self._lbl_set_indicator.setStyleSheet(
                "QLabel { color:#9e9e9e; font-size:13px; padding:4px; }")
            for b in (self._btn_set_end, self._btn_session_end,
                      self._btn_rest_pause, self._btn_rest_skip):
                b.setEnabled(False)
            self._btn_rest_pause.setText("⏸ 일시정지")

    # Protocol queue management
    def _add_to_queue(self) -> None:
        opts = self._options.options()
        label = self._summarize_opts(opts)
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, opts)
        self._queue_list.addItem(item)

    def _remove_from_queue(self) -> None:
        row = self._queue_list.currentRow()
        if row >= 0:
            self._queue_list.takeItem(row)

    def _move_item(self, delta: int) -> None:
        row = self._queue_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self._queue_list.count():
            return
        item = self._queue_list.takeItem(row)
        self._queue_list.insertItem(new_row, item)
        self._queue_list.setCurrentRow(new_row)

    def _clear_queue(self) -> None:
        self._queue_list.clear()

    @staticmethod
    def _summarize_opts(opts: dict) -> str:
        test = opts.get("test", "?")
        dur = opts.get("duration_s", 0.0)
        parts = [test, f"{dur:.0f}s"]
        if test in ("balance_eo", "balance_ec"):
            parts.append(opts.get("stance", "two"))
        if test == "reaction":
            parts.append(f"×{opts.get('n_stimuli', 0)}")
            parts.append(opts.get("trigger", "auto"))
        if test == "encoder" and opts.get("encoder_prompt"):
            parts.append(f"\"{opts['encoder_prompt']}\"")
        return "  ·  ".join(parts)

    # Start / stop
    def _on_start_clicked(self) -> None:
        # F7: double-click defense — immediately block further clicks
        if self._starting or (self._worker is not None and self._worker.isRunning()):
            return
        if self._active_subject is None:
            QMessageBox.warning(self, "피험자 미선택",
                                "먼저 Subjects 탭에서 측정 대상을 선택해주세요.")
            return
        self._starting = True
        self._btn_start.setEnabled(False)

        # Build queue based on mode
        if self._mode_protocol.isChecked():
            if self._queue_list.count() == 0:
                QMessageBox.warning(self, "프로토콜 비어있음",
                                    "'+ 추가' 버튼으로 테스트를 먼저 추가해주세요.")
                self._starting = False
                self._refresh_controls()
                return
            self._protocol_queue = [
                self._queue_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self._queue_list.count())
            ]
        else:
            self._protocol_queue = [self._options.options()]

        self._protocol_idx = 0
        self._protocol_results = []
        self._abort_protocol = False
        self._update_protocol_header()
        self._start_next_in_queue()

    def _start_next_in_queue(self) -> None:
        self._stop_transition_timer()
        if self._abort_protocol or self._protocol_idx >= len(self._protocol_queue):
            self._starting = False
            self._on_protocol_complete()
            return
        opts = dict(self._protocol_queue[self._protocol_idx])
        if len(self._protocol_queue) > 1:
            opts["session_name_suffix"] = (
                f"p{self._protocol_idx + 1}of{len(self._protocol_queue)}")
        # Snapshot the running item's options so UI renderers (overlay, etc.)
        # read from here rather than the — possibly stale — options panel.
        self._current_opts = dict(opts)
        # Strip UI-only keys before constructing RecorderConfig
        cfg_kwargs = {k: v for k, v in opts.items() if not k.startswith("_")}
        # Pull sex / birthdate / height from the DB row so the recorder
        # writes them into session.json — required by the strength_3lift
        # 1RM grade lookup (Phase V1-bugfix). Optional fields default to
        # None on older subject rows; the analyzer falls back to a DB
        # lookup by subject_id when meta is missing them.
        subj = self._active_subject
        cfg = RecorderConfig(
            subject_id=subj.id,
            subject_name=subj.name,
            subject_kg=subj.weight_kg,
            subject_sex=getattr(subj, "gender", None),
            subject_birthdate=getattr(subj, "birthdate", None),
            subject_height_cm=getattr(subj, "height_cm", None),
            **cfg_kwargs,
        )
        self._append_log(
            f"[{self._protocol_idx + 1}/{len(self._protocol_queue)}] "
            f"starting: {self._summarize_opts(opts)}")

        self._camera_view.reset()
        self._dashboard.reset()
        self._dashboard.set_subject_weight(self._active_subject.weight_kg)
        self._overlay.reset()
        # Encoder-bar availability: honour both the hardware flag AND the
        # per-session uses_encoder / test_type. Balance tests OR an explicit
        # "엔코더 사용" uncheck -> both bars show "비활성".
        enc_active = _encoder_active_for_test(
            cfg.test, getattr(cfg, "uses_encoder", True))
        self._enc1_bar.set_available(enc_active and config.ENCODER1_AVAILABLE)
        self._enc2_bar.set_available(enc_active and config.ENCODER2_AVAILABLE)
        # Reset encoder bars so stale values don't persist between tests
        if enc_active and config.ENCODER1_AVAILABLE:
            self._enc1_bar.set_value(0.0)
        if enc_active and config.ENCODER2_AVAILABLE:
            self._enc2_bar.set_value(0.0)

        # Real-time rep counter for free_exercise. Requires enc1 hardware
        # AND the user enabled "엔코더 사용". For any other combination we
        # hide the Reps readout to avoid showing a frozen 0.
        if (cfg.test == "free_exercise" and enc_active
                and config.ENCODER1_AVAILABLE):
            self._rep_counter = RealtimeRepCounter(min_rom_mm=80.0)
            self._dashboard.set_rep_counter_visible(True)
            self._dashboard.set_rep_count(0)
        else:
            self._rep_counter = None
            self._dashboard.set_rep_counter_visible(False)
        self._update_protocol_header()
        self._worker = RecordWorker(cfg, parent=self)
        self._worker.state_changed.connect(self._on_state)
        self._worker.log_message.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.camera_frame.connect(self._camera_view.on_camera_frame)
        self._worker.daq_frame.connect(self._dashboard.on_daq_frame)
        self._worker.daq_frame.connect(self._on_daq_frame_encoders)

        # Start realtime pose overlay if the user opted in
        want_live  = bool(opts.get("_live_pose", False))
        live_cam   = str(opts.get("_live_cam_id", config.POSE_REALTIME_CAM_ID))
        live_cplx  = int(opts.get("_live_complexity",
                                  config.POSE_REALTIME_COMPLEXITY))
        if want_live and any(c["id"] == live_cam for c in config.CAMERAS):
            self._start_pose_live(live_cam, live_cplx)

        self._worker.start()
        self._starting = False
        self._refresh_controls()

    def _on_stop_clicked(self) -> None:
        # F3: in protocol mode this only cancels the CURRENT test; protocol
        # continues to the next item in _on_finished.
        if self._worker is not None:
            self._worker.cancel()
            self._append_log("cancel requested (current test)")
        elif self._transition_timer is not None:
            # Stop during between-test pause = skip waiting, start next now
            self._stop_transition_timer()
            self._append_log("skip pause, starting next")
            self._start_next_in_queue()

    def _on_abort_clicked(self) -> None:
        self._abort_protocol = True
        self._append_log("ABORT requested (entire protocol)")
        if self._worker is not None:
            self._worker.cancel()
        self._stop_transition_timer()
        # If we were in transition pause, call complete directly
        if self._worker is None:
            self._on_protocol_complete()

    def _fire_manual(self, response: str) -> None:
        if self._worker is not None:
            self._worker.manual_reaction(response)

    def _fire_manual_random(self) -> None:
        if self._worker is not None:
            self._worker.manual_random()

    # ── worker signals ─────────────────────────────────────────────────────
    def _on_daq_frame_encoders(self, fr) -> None:
        """Update encoder bars + live rep counter from DAQ stream. Honours
        both the config hardware flag AND the per-session enc-active state
        so a "비활성" bar never animates with live data."""
        if self._enc1_bar.isAvailable():
            self._enc1_bar.set_value(fr.enc1_mm)
        if self._enc2_bar.isAvailable():
            self._enc2_bar.set_value(fr.enc2_mm)
        # Push enc1 into rep counter (enc1 is the one available channel)
        if self._rep_counter is not None:
            n = self._rep_counter.push(fr.enc1_mm)
            self._dashboard.set_rep_count(n)

    def _on_state(self, st: RecorderState) -> None:
        # Phase-aware live display: only the "recording" phase gets the
        # full raw signal through the dashboard. During wait / countdown /
        # done / idle / inter_set_rest phases the dashboard suppresses
        # <10kg readings so the trainer doesn't see scrolling noise.
        self._dashboard.set_recording(st.phase == "recording")

        kg = self._active_subject.weight_kg if self._active_subject else 0.0
        # Use the running item's stance, not the UI value. For non-balance
        # tests stance is effectively "two" regardless.
        opts = self._current_opts or {}
        test = opts.get("test", "")
        stance = opts.get("stance", "two") if test in (
            "balance_eo", "balance_ec") else "two"
        self._overlay.update_from_state(st, subject_kg=kg, stance=stance)
        # Multi-set strength: refresh button enables + indicator label
        # whenever the state ticks. (Cheap — just label text + setEnabled.)
        self._refresh_strength_controls(st)
        # V6 — forward the positional cue (cognitive_reaction) to the
        # camera preview. Recorder clears the cue when the stim banner
        # expires; passing None in either coord tells CameraView to hide
        # the ring. No-op for tests that never set the fields.
        self._camera_view.set_positional_cue(
            getattr(st, "cog_target_x_norm", None),
            getattr(st, "cog_target_y_norm", None),
            getattr(st, "cog_target_label",  None),
        )

    def _on_finished(self, result: dict) -> None:
        # Realtime overlay no longer needed — kill the live worker
        self._stop_pose_live()
        meta = result.get("meta", {}) or {}
        fell_off = bool(meta.get("fell_off_detected", False))
        self._append_log(
            f"finished: {result.get('session_name') or '?'}"
            f"{'  [FELL_OFF]' if fell_off else ''}"
        )
        sess_dir = result.get("session_dir", "?")
        cancelled = bool(result.get("cancelled", False))

        # Record in DB
        db_row = None
        if self._active_subject is not None and sess_dir and sess_dir != "?":
            try:
                db_row = self._record_session_to_db(result)
                self._append_log(f"DB session saved: {db_row.id}")
            except Exception as e:
                self._append_log(f"DB save error: {e}")

        # Auto-analyze unless the test was cancelled / fell-off / no DAQ data
        if (not cancelled and not fell_off
                and int(result.get("n_daq_samples", 0)) > 0
                and db_row is not None):
            self._queue_auto_analysis(db_row.id, sess_dir,
                                      meta.get("test"))

        self._protocol_results.append(result)
        self._worker = None
        self.session_completed.emit(result)

        # F3: continue to next regardless of per-test cancel status, UNLESS
        # the whole protocol was explicitly aborted.
        has_more = (self._protocol_idx + 1 < len(self._protocol_queue))
        if self._abort_protocol or not has_more:
            self._refresh_controls()
            self._on_protocol_complete()
            return

        # Advance + schedule next with a visible countdown.
        self._protocol_idx += 1
        nxt_opts = self._protocol_queue[self._protocol_idx]
        self._append_log(
            f"next in {BETWEEN_TEST_PAUSE_S:.0f}s: "
            f"{self._summarize_opts(nxt_opts)}")
        self._refresh_controls()
        self._begin_transition_pause()

    def _begin_transition_pause(self) -> None:
        """Between-test countdown — shows header + overlay with prev/next."""
        import time as _time
        self._transition_end_ns = _time.monotonic_ns() + int(
            BETWEEN_TEST_PAUSE_S * 1e9)
        if self._transition_timer is not None:
            self._transition_timer.stop()
        self._transition_timer = QTimer(self)
        self._transition_timer.setInterval(200)
        self._transition_timer.timeout.connect(self._on_transition_tick)
        self._transition_timer.start()
        self._on_transition_tick()   # paint immediately

    def _on_transition_tick(self) -> None:
        import time as _time
        rem = max(0.0, (self._transition_end_ns - _time.monotonic_ns()) / 1e9)
        self._update_protocol_header(
            transitioning=True, transition_remaining_s=rem)
        # Overlay shows prev/next summary so the subject knows what happened
        # and what's next — no more "select subject" idle message.
        prev_ko = self._ko_name_for_opts(
            self._protocol_queue[self._protocol_idx - 1]
            if self._protocol_idx > 0 else None
        )
        next_ko = self._ko_name_for_opts(
            self._protocol_queue[self._protocol_idx]
            if self._protocol_idx < len(self._protocol_queue) else None
        )
        self._overlay.render_transition(prev_ko, next_ko, rem)
        if rem <= 0.0:
            self._stop_transition_timer()
            self._start_next_in_queue()

    @staticmethod
    def _ko_name_for_opts(opts: Optional[dict]) -> str:
        if not opts:
            return "—"
        test = opts.get("test", "?")
        name = TEST_KO.get(test, test)
        extras = []
        if test in ("balance_eo", "balance_ec"):
            stance = opts.get("stance", "two")
            extras.append(
                {"two": "양발", "left": "좌측발", "right": "우측발"}
                .get(stance, stance)
            )
        dur = opts.get("duration_s", 0.0)
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{name}{suffix} · {dur:.0f}s"

    def _stop_transition_timer(self) -> None:
        if self._transition_timer is not None:
            self._transition_timer.stop()
            self._transition_timer = None

    def _on_protocol_complete(self) -> None:
        self._stop_transition_timer()
        self._protocol_header.hide_header()
        if not self._protocol_results:
            self._refresh_controls()
            return
        lines = []
        any_cancel = False
        for r in self._protocol_results:
            meta = r.get("meta", {}) or {}
            cancelled = bool(r.get("cancelled", False))
            fell = bool(meta.get("fell_off_detected", False))
            any_cancel = any_cancel or cancelled
            if fell:
                status = "⚠ 이탈"
            elif cancelled:
                status = "✗ 취소"
            else:
                status = "✓ 완료"
            lines.append(
                f"{status}  {meta.get('test', '?')}  "
                f"{r.get('n_daq_samples', 0)} samples  →  "
                f"{r.get('session_name', '?')}"
            )
        title = ("프로토콜 종료 (일부 취소됨)"
                 if any_cancel else "프로토콜 완료")
        QMessageBox.information(self, title, "\n".join(lines))
        # Reset for next run
        self._protocol_queue = []
        self._protocol_idx = 0
        self._protocol_results = []
        self._abort_protocol = False
        self._current_opts = None
        self._refresh_controls()

    # ── realtime pose overlay ──────────────────────────────────────────────
    def _start_pose_live(self, cam_id: str, complexity: int) -> None:
        if self._pose_live_worker is not None:
            self._stop_pose_live()
        w = PoseLiveWorker(cam_id=cam_id, complexity=complexity, parent=self)
        self._pose_live_worker = w
        w.log_message.connect(self._append_log)
        w.pose_overlay.connect(self._camera_view.on_pose_overlay)
        # Feed every camera_frame from RecordWorker into the live worker;
        # it filters internally to the selected cam_id.
        if self._worker is not None:
            self._worker.camera_frame.connect(
                lambda cid, bgr, idx, t_ns: w.feed_frame(cid, bgr))
        w.start()

    def _stop_pose_live(self) -> None:
        w = self._pose_live_worker
        if w is None:
            return
        try:
            w.stop()
            w.wait(1500)
        except Exception:
            pass
        self._pose_live_worker = None
        self._camera_view.clear_overlay()

    # ── auto-analysis ──────────────────────────────────────────────────────
    def _queue_auto_analysis(self, session_id: str,
                             session_dir: str, test_type: Optional[str]) -> None:
        """Kick off background Force analysis. If auto-pose is enabled, chain
        Pose processing + re-analyze after Force analysis completes."""
        if not test_type:
            return
        want_pose = bool((self._current_opts or {}).get("_auto_pose", False))
        worker = AnalysisWorker(session_dir, test_type=test_type, parent=self)
        self._analysis_workers[session_id] = worker

        def _on_started(sd: str) -> None:
            self._append_log(f"analysis started: {Path(sd).name}")
            try:
                db_models.update_session_status(session_id, "analyzing")
            except Exception as e:
                self._append_log(f"DB status update error: {e}")

        def _on_done(sd: str, res: dict) -> None:
            ok = res.get("error") is None
            status = "analyzed" if ok else "analysis_failed"
            try:
                db_models.update_session_status(session_id, status)
            except Exception as e:
                self._append_log(f"DB status update error: {e}")
            msg = (f"analysis OK: {Path(sd).name}" if ok
                   else f"analysis FAILED: {Path(sd).name} — {res.get('error')}")
            self._append_log(msg)
            self._analysis_workers.pop(session_id, None)
            # Chain 2D pose processing if the user opted in + force succeeded
            if ok and want_pose:
                self._queue_pose_processing(session_id, sd, test_type)

        worker.started_ok.connect(_on_started)
        worker.log_message.connect(self._append_log)
        worker.finished_ok.connect(_on_done)
        worker.start()

    def _queue_pose_processing(self, session_id: str,
                               session_dir: str, test_type: str) -> None:
        """Run MediaPipe BlazePose on every camera in the session, then
        re-analyze so joint angles are baked into result.json. The
        complexity setting comes from the options panel (user selectable)."""
        complexity = int((self._current_opts or {}).get(
            "_post_complexity", config.POSE_POSTRECORD_COMPLEXITY))
        pose = PoseWorker(session_dir, complexity=complexity, parent=self)
        self._pose_workers[session_id] = pose
        self._append_log(f"pose started: {Path(session_dir).name}")
        try:
            db_models.update_session_status(session_id, "analyzing")
        except Exception:
            pass

        def _on_pose_progress(cam: str, i: int, total: int) -> None:
            if i == total or i % max(1, total // 5) == 0:
                self._append_log(f"  [pose {cam}] {i}/{total}")

        def _on_pose_all_done(sd: str, success: bool, err: str) -> None:
            self._pose_workers.pop(session_id, None)
            if not success:
                self._append_log(f"pose FAILED: {err}")
                try:
                    db_models.update_session_status(session_id, "analyzed")
                except Exception:
                    pass
                return
            self._append_log(f"pose OK: {Path(sd).name} — re-analyzing")
            # Re-run force analysis so pose angles are merged into result.json
            rerun = AnalysisWorker(sd, test_type=test_type, parent=self)

            def _on_rerun_done(sd2: str, res: dict) -> None:
                new_status = ("analyzed_full" if res.get("error") is None
                              else "analysis_failed")
                try:
                    db_models.update_session_status(session_id, new_status)
                except Exception:
                    pass
                self._append_log(f"re-analysis status: {new_status}")

            rerun.log_message.connect(self._append_log)
            rerun.finished_ok.connect(_on_rerun_done)
            rerun.start()
            # Keep a ref so it isn't GC'd mid-run
            self._analysis_workers[f"{session_id}:rerun"] = rerun

        pose.progress.connect(_on_pose_progress)
        pose.log_message.connect(self._append_log)
        pose.all_done.connect(_on_pose_all_done)
        pose.start()

    def _record_session_to_db(self, result: dict) -> DbSession:
        """Insert a sessions-table row for the just-finished session."""
        meta = result.get("meta", {}) or {}
        cancelled = bool(result.get("cancelled", False))
        options = {k: meta.get(k) for k in (
            "duration_s", "stance", "vision", "reaction_trigger",
            "reaction_responses", "encoder_prompt", "smart_wait",
            "subject_kg", "wait_duration_s",
            # free_exercise
            "exercise_name", "load_kg", "use_bodyweight_load",
            # encoder usage (non-balance tests)
            "uses_encoder",
        ) if meta.get(k) is not None}
        subj = self._active_subject
        sess = DbSession.new(
            subject_id=subj.id,
            test_type=meta.get("test", self._options.current_test()),
            duration_s=float(meta.get("duration_s", 0.0) or 0.0),
            options_json=DbSession.encode_options(options),
            session_dir=result.get("session_dir"),
            status="cancelled" if cancelled else "recorded",
            trainer=subj.trainer,
        )
        return db_models.create_session(sess)

    # ── utilities ──────────────────────────────────────────────────────────
    def _append_log(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _update_protocol_header(self, *, transitioning: bool = False,
                                transition_remaining_s: float = 0.0) -> None:
        if len(self._protocol_queue) > 1:
            self._protocol_header.show_queue(
                self._protocol_queue, self._protocol_idx,
                transitioning=transitioning,
                transition_remaining_s=transition_remaining_s,
            )
        else:
            self._protocol_header.hide_header()

    def _refresh_controls(self) -> None:
        running = (self._worker is not None and self._worker.isRunning())
        in_transition = (self._transition_timer is not None)
        active = running or in_transition or self._starting
        can_start = (self._active_subject is not None) and (not active)

        self._btn_start.setEnabled(can_start)
        self._btn_stop.setEnabled(active)
        # Abort only matters in protocol mode with queue > 1
        protocol_active = active and len(self._protocol_queue) > 1
        self._btn_abort.setEnabled(protocol_active)
        self._options.setEnabled(not active)
        # Reaction's directional buttons (left/right/jump) only apply to
        # the classic "reaction" test; cognitive_reaction uses positional
        # cues so a random-pick (Space) is enough.
        for b in (self._btn_left, self._btn_right, self._btn_jump):
            b.setEnabled(running and self._options.current_test() == "reaction")
        # Random-pick button + Space shortcut work for both reaction and
        # cognitive_reaction (in either auto or manual trigger).
        self._btn_rand.setEnabled(
            running and self._options.current_test()
            in ("reaction", "cognitive_reaction"))
        # VRT cue button — enabled while recording a squat test.
        self._btn_vrt_cue.setEnabled(
            running and self._options.current_test()
            in ("squat", "overhead_squat"))

        # Context-aware disabled tooltip for 측정 시작 — explains *why*
        # the button is grey instead of silently refusing clicks.
        if can_start:
            self._btn_start.setToolTip("측정을 시작합니다 (단축키 F5)")
        elif self._active_subject is None:
            self._btn_start.setToolTip(
                "먼저 Subjects 탭에서 측정 대상을 선택하세요.")
        elif active:
            self._btn_start.setToolTip(
                "이미 측정이 진행 중입니다 — 끝난 뒤 다시 시도하세요.")
