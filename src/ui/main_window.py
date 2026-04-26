"""
Biomech MoCap main window.

Top row (4 columns):
    [ Cam 0 ]  [ Cam 1 ]  [ Cam 2 ]  [ 3D Skeleton ]

Bottom row (tabbed):
    Live / Calibration / Analysis / Playback
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget, QFileDialog,
)

import config
from src.ui.widgets.camera_tile import CameraTile
from src.ui.widgets.skeleton3d import Skeleton3DView
from src.ui.widgets.force_widgets import VGRFPlot, COPTrajectory, EncoderBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{config.APP_TITLE}  v{config.APP_VERSION}")
        self.resize(1680, 900)

        self._capture_running = False
        self._session_dir: Path | None = None

        self._build_ui()
        self._wire_timers()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Toolbar-like control row
        root.addLayout(self._build_control_row())

        # Top row: 3 cameras + 3D skeleton (each column ~equal width)
        top = QSplitter(Qt.Orientation.Horizontal)
        self._tiles: list[CameraTile] = []
        for cam in config.CAMERAS:
            tile = CameraTile(cam["id"], cam["index"], cam["label"])
            self._tiles.append(tile)
            top.addWidget(tile)

        self._skel3d = Skeleton3DView()
        top.addWidget(self._skel3d)
        top.setSizes([config.TILE_WIDTH] * 4)
        root.addWidget(top, stretch=2)

        # Bottom row: tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_live_tab(),        "Live")
        self._tabs.addTab(self._build_calibration_tab(), "Calibration")
        self._tabs.addTab(self._build_analysis_tab(),    "Analysis")
        self._tabs.addTab(self._build_playback_tab(),    "Playback (TDMS)")
        root.addWidget(self._tabs, stretch=3)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Generate ChArUco → Calibrate → Record → Analyze.")

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._btn_start = QPushButton("▶ Start Preview")
        self._btn_start.clicked.connect(self._toggle_preview)
        row.addWidget(self._btn_start)

        self._btn_record = QPushButton("● Record Session")
        self._btn_record.setEnabled(False)
        self._btn_record.clicked.connect(self._toggle_record)
        row.addWidget(self._btn_record)

        row.addSpacing(20)

        self._btn_session_dir = QPushButton("📁 Session Dir")
        self._btn_session_dir.clicked.connect(self._choose_session_dir)
        row.addWidget(self._btn_session_dir)

        self._lbl_session = QLabel(str(config.SESSIONS_DIR))
        self._lbl_session.setStyleSheet("color:#888; font-size:10px;")
        row.addWidget(self._lbl_session)

        row.addStretch()

        self._lbl_mode = QLabel("Mode: Online preview")
        self._lbl_mode.setStyleSheet("color:#90caf9;")
        row.addWidget(self._lbl_mode)

        return row

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        # Left: vGRF plot
        self._vgrf = VGRFPlot()
        lay.addWidget(self._vgrf, stretch=3)

        # Middle: CoP trajectory
        self._cop = COPTrajectory()
        lay.addWidget(self._cop, stretch=2)

        # Right: encoder bars
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)
        self._enc = EncoderBar()
        right_lay.addWidget(self._enc)
        right_lay.addStretch()
        lay.addWidget(right, stretch=1)

        return w

    def _build_calibration_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Calibration Workflow  (skeleton + force plate)")
        title.setStyleSheet("color:#90caf9; font-weight:bold; font-size:14px;")
        lay.addWidget(title)

        steps = QLabel(
            "This project uses SKELETON-BASED calibration - no ChArUco board.\n"
            "The force plate (558 x 432 mm) provides the world scale via CoP.\n\n"
            "Run these three commands in order (terminal):\n\n"
            "  1. Record a 60s synchronized session (3 cams + DAQ):\n"
            "     python scripts/record_calibration_session.py\n\n"
            "  2. Estimate camera intrinsics + extrinsics from 2D poses:\n"
            "     python scripts/calibrate_from_poses.py --session <name>\n\n"
            "  3. Align to world frame using force plate CoP:\n"
            "     python scripts/align_world_from_force.py --session <name>\n\n"
            "Output: data/calibration/world_frame.npz\n\n"
            "See docs/CALIBRATION_WORKFLOW.md for full details."
        )
        steps.setStyleSheet("color:#CCC; font-family:Consolas,monospace;")
        lay.addWidget(steps)
        lay.addStretch()
        return w

    def _build_analysis_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Biomech Analysis")
        title.setStyleSheet("color:#CE93D8; font-weight:bold; font-size:14px;")
        lay.addWidget(title)

        info = QLabel(
            "Target analyses (user-specified):\n"
            "  • Static balance / posture  (single-leg, double-leg, Romberg)\n"
            "  • Weight Bearing Asymmetry (WBA)\n"
            "  • Counter Movement Jump — height, RFD, impulse\n"
            "  • Overhead squat form assessment\n"
            "  • Squat / lift technique (bar path via encoders)\n"
            "  • Reaction time tests (visual/auditory)\n"
            "  • Proprioception tests (position reproduction)\n\n"
            "Each analysis runs on a recorded session (.tdms + .mp4 × 3 + 3D .trc)."
        )
        info.setStyleSheet("color:#CCC;")
        lay.addWidget(info)
        lay.addStretch()
        return w

    def _build_playback_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        btn = QPushButton("Open TDMS file…")
        btn.clicked.connect(self._open_tdms)
        lay.addWidget(btn)

        self._lbl_playback = QLabel("No file loaded.")
        self._lbl_playback.setStyleSheet("color:#888;")
        lay.addWidget(self._lbl_playback)
        lay.addStretch()
        return w

    # ── timers / wiring ───────────────────────────────────────────────────────
    def _wire_timers(self):
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(config.PLOT_UPDATE_MS)
        self._ui_timer.timeout.connect(self._refresh_plots)

    # ── actions ───────────────────────────────────────────────────────────────
    def _toggle_preview(self):
        if not self._capture_running:
            ok_any = False
            for tile in self._tiles:
                if tile.start():
                    ok_any = True
            self._capture_running = True
            self._btn_start.setText("■ Stop Preview")
            self._btn_record.setEnabled(ok_any)
            self._ui_timer.start()
            self.statusBar().showMessage("Camera preview running.")
        else:
            for tile in self._tiles:
                tile.stop()
            self._capture_running = False
            self._btn_start.setText("▶ Start Preview")
            self._btn_record.setEnabled(False)
            self._ui_timer.stop()
            self.statusBar().showMessage("Preview stopped.")

    def _toggle_record(self):
        # TODO: wire to src.io.recorder once implemented
        self.statusBar().showMessage(
            "Recording module not yet wired. See roadmap in README."
        )

    def _refresh_plots(self):
        self._vgrf.refresh()
        self._cop.refresh()

    def _choose_session_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Session Directory", str(config.SESSIONS_DIR)
        )
        if d:
            self._session_dir = Path(d)
            self._lbl_session.setText(str(self._session_dir))

    def _open_tdms(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open TDMS file", str(config.SESSIONS_DIR), "TDMS (*.tdms)"
        )
        if f:
            self._lbl_playback.setText(f"Loaded: {f}  (viewer integration pending)")

    def closeEvent(self, event):
        if self._capture_running:
            for tile in self._tiles:
                tile.stop()
        super().closeEvent(event)
