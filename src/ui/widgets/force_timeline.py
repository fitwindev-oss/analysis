"""
ForceTimelineWidget — full-session force plot with draggable time cursor.

Plots vGRF (total / Board1 / Board2) over the whole recording. A vertical
``InfiniteLine`` represents the playback cursor; dragging it emits
``seek_requested(t_s)`` so the PlaybackController can snap to that time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from src.analysis.common import ForceSession, load_force_session
from src.ui.widgets.replay_colors import HOVER_LINE_COLOR


class ForceTimelineWidget(QWidget):
    seek_requested = pyqtSignal(float)   # when user drags/clicks cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._force: Optional[ForceSession] = None
        # Off-plate region overlays — re-created whenever set_departures
        # is called so we can shed them on unload without leaking.
        self._departure_items: list[pg.LinearRegionItem] = []

        pg.setConfigOption("background", "#111")
        pg.setConfigOption("foreground", "#CCC")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setLabel("left",   "Force (N)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(-10, 10))

        self._c_total = self._plot.plot(
            pen=pg.mkPen("#A5D6A7", width=2), name="Total")
        self._c_b1 = self._plot.plot(
            pen=pg.mkPen("#4FC3F7", width=1), name="Board1 (L)")
        self._c_b2 = self._plot.plot(
            pen=pg.mkPen("#FF8A65", width=1), name="Board2 (R)")

        # Draggable time cursor
        self._cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen("#FFEB3B", width=2),
            hoverPen=pg.mkPen("#FFF59D", width=3),
        )
        self._plot.addItem(self._cursor)
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)

        # Hover crosshair + value label (not draggable)
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

        # Click on plot → seek
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        lay.addWidget(self._plot)

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, session_dir: str | Path) -> bool:
        try:
            self._force = load_force_session(Path(session_dir))
        except Exception as e:
            self._force = None
            return False
        t = self._force.t_s
        self._c_total.setData(t, self._force.total)
        self._c_b1.setData(t, self._force.b1_total)
        self._c_b2.setData(t, self._force.b2_total)
        self._plot.setXRange(float(t[0]), float(t[-1]), padding=0)
        self._cursor.setValue(0.0)
        return True

    def unload(self) -> None:
        self._force = None
        self._c_total.setData([], [])
        self._c_b1.setData([], [])
        self._c_b2.setData([], [])
        self._cursor.setValue(0.0)
        self.set_departures([])

    def set_departures(self, intervals: list[tuple[float, float]]) -> None:
        """Draw translucent regions over off-plate intervals.

        Each ``(t0, t1)`` becomes a ``LinearRegionItem`` rendered behind
        the force lines (zValue −10) so the data plot stays readable.
        Calling with ``[]`` clears any previously drawn regions.
        Phase U3-3: visual encoding of the events.csv data.
        """
        # Clear previous regions
        for item in self._departure_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._departure_items.clear()
        # Add new ones — muted orange ≈ 25% alpha keeps the band visible
        # without overwhelming the green/blue/orange force traces.
        for t0, t1 in intervals:
            region = pg.LinearRegionItem(
                values=(float(t0), float(t1)),
                brush=pg.mkBrush(255, 152, 0, 50),       # 오렌지 ≈ 20% alpha
                pen=pg.mkPen(255, 152, 0, 130, width=1),
                movable=False,
            )
            region.setZValue(-10)        # behind force lines
            self._plot.addItem(region)
            self._departure_items.append(region)

    def duration_s(self) -> float:
        if self._force is None or len(self._force.t_s) == 0:
            return 0.0
        return float(self._force.t_s[-1] - self._force.t_s[0])

    def set_cursor(self, t_s: float) -> None:
        # Avoid re-emitting sigPositionChanged during external updates
        self._cursor.blockSignals(True)
        self._cursor.setValue(float(t_s))
        self._cursor.blockSignals(False)

    # ── internals ──────────────────────────────────────────────────────────
    def _on_cursor_moved(self) -> None:
        t = float(self._cursor.value())
        self.seek_requested.emit(t)

    def _on_mouse_clicked(self, event) -> None:
        # pyqtgraph scene click → map to plot coords
        if self._force is None:
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
        if self._force is None:
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
        total = float(self._force.total[idx])
        b1    = float(self._force.b1_total[idx])
        b2    = float(self._force.b2_total[idx])
        self._hover_label.setHtml(
            f"<div style='background:rgba(20,20,20,200); padding:3px 6px;'>"
            f"t = {t:5.2f}s &nbsp; "
            f"<span style='color:#A5D6A7;'>Total</span> {total:6.0f} N &nbsp; "
            f"<span style='color:#4FC3F7;'>B1</span> {b1:5.0f} N &nbsp; "
            f"<span style='color:#FF8A65;'>B2</span> {b2:5.0f} N"
            f"</div>"
        )
        vr = self._plot.plotItem.vb.viewRange()
        self._hover_label.setPos(vr[0][0], vr[1][1])
        self._hover_label.setVisible(True)
