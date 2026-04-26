"""
EncoderTimelineWidget — full-session encoder (bar/rod) position plot.

Overlays enc1 (left, blue) and enc2 (right, orange) on a single time-axis
so left/right bar travel can be compared at a glance. Shares the draggable
``seek_requested(t_s)`` contract with ForceTimelineWidget so both plots can
drive the same PlaybackController.

When the channel is not applicable (balance tests, or uses_encoder=False),
call ``set_available(False)`` — the widget shows a "비활성" banner over the
plot area and drops the data traces. Layout stays put, matching the
Measure-tab encoder-bar behaviour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedLayout

import config
from src.analysis.common import ForceSession, load_force_session
from src.ui.widgets.replay_colors import HOVER_LINE_COLOR


# Match ForceDashboard colors so L/R stays visually consistent across the
# Measure and Replay tabs.
_ENC1_COLOR = "#4FC3F7"   # left, blue
_ENC2_COLOR = "#FF8A65"   # right, orange


class EncoderTimelineWidget(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._force: Optional[ForceSession] = None
        self._available: bool = True
        self._departure_items: list[pg.LinearRegionItem] = []

        pg.setConfigOption("background", "#111")
        pg.setConfigOption("foreground", "#CCC")

        # Stacked layout: index 0 = live plot, index 1 = "비활성" placeholder
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(2, 2, 2, 2)

        # ── Plot page ───────────────────────────────────────────────────────
        plot_page = QWidget()
        pl = QVBoxLayout(plot_page)
        pl.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setLabel("left",   "Encoder (mm)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(-10, 10))

        self._c_enc1 = self._plot.plot(
            pen=pg.mkPen(_ENC1_COLOR, width=2), name="L (enc1)")
        self._c_enc2 = self._plot.plot(
            pen=pg.mkPen(_ENC2_COLOR, width=2), name="R (enc2)")

        # Draggable time cursor
        self._cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen("#FFEB3B", width=2),
            hoverPen=pg.mkPen("#FFF59D", width=3),
        )
        self._plot.addItem(self._cursor)
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)

        # Hover crosshair + readout
        self._hover_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen(HOVER_LINE_COLOR, width=1,
                         style=Qt.PenStyle.DashLine),
        )
        self._hover_line.setVisible(False)
        self._plot.addItem(self._hover_line, ignoreBounds=True)
        self._hover_label = pg.TextItem(anchor=(0, 0), color=(230, 230, 230))
        self._hover_label.setVisible(False)
        self._plot.addItem(self._hover_label, ignoreBounds=True)

        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        pl.addWidget(self._plot)
        self._stack.addWidget(plot_page)       # index 0

        # ── Placeholder page (for balance / uses_encoder=False) ────────────
        ph = QWidget()
        phl = QVBoxLayout(ph)
        phl.setContentsMargins(0, 0, 0, 0)
        self._placeholder_label = QLabel("엔코더 비활성")
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pf = QFont(); pf.setPointSize(12); pf.setBold(True)
        self._placeholder_label.setFont(pf)
        self._placeholder_label.setStyleSheet(
            "QLabel { color:#888; background:#1a1a1a; "
            "border:1px dashed #555; padding:20px; }"
        )
        phl.addWidget(self._placeholder_label)
        self._stack.addWidget(ph)              # index 1

        # Start on plot page
        self._stack.setCurrentIndex(0)

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, session_dir: str | Path) -> bool:
        try:
            self._force = load_force_session(Path(session_dir))
        except Exception:
            self._force = None
            return False
        t = self._force.t_s
        self._c_enc1.setData(t, self._force.enc1)
        # enc2 only if hardware flag allows — otherwise omit trace entirely
        # so the placeholder noise doesn't pollute the plot range.
        if bool(getattr(config, "ENCODER2_AVAILABLE", False)):
            self._c_enc2.setData(t, self._force.enc2)
        else:
            self._c_enc2.setData([], [])
        self._plot.setXRange(float(t[0]), float(t[-1]), padding=0)
        self._cursor.setValue(0.0)
        return True

    def unload(self) -> None:
        self._force = None
        self._c_enc1.setData([], [])
        self._c_enc2.setData([], [])
        self._cursor.setValue(0.0)
        self.set_departures([])

    def set_departures(self, intervals: list[tuple[float, float]]) -> None:
        """Mirror of ForceTimelineWidget.set_departures so all
        timeseries widgets in the right column show the same off-plate
        bands for visual cross-correlation. Phase U3-3."""
        for item in self._departure_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._departure_items.clear()
        for t0, t1 in intervals:
            region = pg.LinearRegionItem(
                values=(float(t0), float(t1)),
                brush=pg.mkBrush(255, 152, 0, 50),
                pen=pg.mkPen(255, 152, 0, 130, width=1),
                movable=False,
            )
            region.setZValue(-10)
            self._plot.addItem(region)
            self._departure_items.append(region)

    def duration_s(self) -> float:
        if self._force is None or len(self._force.t_s) == 0:
            return 0.0
        return float(self._force.t_s[-1] - self._force.t_s[0])

    def set_cursor(self, t_s: float) -> None:
        self._cursor.blockSignals(True)
        self._cursor.setValue(float(t_s))
        self._cursor.blockSignals(False)

    def set_available(self, ok: bool, reason: str = "엔코더 비활성") -> None:
        """Toggle between the live plot and the "비활성" placeholder.

        Placeholder is shown when the test doesn't use the encoder
        (balance) or the operator unchecked "엔코더 사용" at record time.
        """
        self._available = bool(ok)
        if self._available:
            self._stack.setCurrentIndex(0)
        else:
            self._placeholder_label.setText(str(reason))
            self._stack.setCurrentIndex(1)

    def is_available(self) -> bool:
        return self._available

    # ── internals ──────────────────────────────────────────────────────────
    def _on_cursor_moved(self) -> None:
        if not self._available:
            return
        t = float(self._cursor.value())
        self.seek_requested.emit(t)

    def _on_mouse_clicked(self, event) -> None:
        if self._force is None or not self._available:
            return
        try:
            mouse_pt = event.scenePos()
            view = self._plot.plotItem.vb.mapSceneToView(mouse_pt)
            t = float(view.x())
        except Exception:
            return
        t = max(float(self._force.t_s[0]),
                min(float(self._force.t_s[-1]), t))
        self.set_cursor(t)
        self.seek_requested.emit(t)

    def _on_mouse_moved(self, scene_pos) -> None:
        if self._force is None or not self._available:
            self._hover_line.setVisible(False)
            self._hover_label.setVisible(False)
            return
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            self._hover_line.setVisible(False)
            self._hover_label.setVisible(False)
            return
        try:
            vp = self._plot.plotItem.vb.mapSceneToView(scene_pos)
        except Exception:
            return
        t = float(vp.x())
        t = max(float(self._force.t_s[0]),
                min(float(self._force.t_s[-1]), t))
        idx = int(np.clip(np.searchsorted(self._force.t_s, t),
                          0, len(self._force.t_s) - 1))
        self._hover_line.setPos(t)
        self._hover_line.setVisible(True)
        enc1 = float(self._force.enc1[idx])
        parts = [
            f"t = {t:5.2f}s",
            f"<span style='color:{_ENC1_COLOR};'>L</span> {enc1:7.1f} mm",
        ]
        if bool(getattr(config, "ENCODER2_AVAILABLE", False)):
            enc2 = float(self._force.enc2[idx])
            parts.append(
                f"<span style='color:{_ENC2_COLOR};'>R</span> {enc2:7.1f} mm")
        self._hover_label.setHtml(
            f"<div style='background:rgba(20,20,20,200); padding:3px 6px;'>"
            f"{'&nbsp;&nbsp;'.join(parts)}</div>"
        )
        vr = self._plot.plotItem.vb.viewRange()
        self._hover_label.setPos(vr[0][0], vr[1][1])
        self._hover_label.setVisible(True)
