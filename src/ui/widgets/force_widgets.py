"""
Force-related widgets for the bottom row: vGRF plot, COP trajectory, encoders.

These live below the top row (cameras + 3D skeleton).
"""
from __future__ import annotations

import collections
from typing import Optional
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

import config


class VGRFPlot(QWidget):
    """Live vGRF trace for Board1, Board2, Total (N)."""

    def __init__(self, window_s: float = 10.0, parent=None):
        super().__init__(parent)
        self._window_s = window_s
        self._capacity = int(window_s * config.SAMPLE_RATE_HZ)
        self._t  = collections.deque(maxlen=self._capacity)
        self._b1 = collections.deque(maxlen=self._capacity)
        self._b2 = collections.deque(maxlen=self._capacity)
        self._tot = collections.deque(maxlen=self._capacity)
        self._t0: float | None = None

        pg.setConfigOption("background", "#111")
        pg.setConfigOption("foreground", "#CCC")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "vGRF (N)")
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.addLegend(offset=(-10, 10))
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._c1 = self._plot.plot(pen=pg.mkPen("#4FC3F7", width=2), name="Board1 (L)")
        self._c2 = self._plot.plot(pen=pg.mkPen("#FF8A65", width=2), name="Board2 (R)")
        self._ct = self._plot.plot(pen=pg.mkPen("#A5D6A7", width=2), name="Total")
        layout.addWidget(self._plot)

    def push(self, ts: float, b1_n: float, b2_n: float):
        if self._t0 is None:
            self._t0 = ts
        self._t.append(ts - self._t0)
        self._b1.append(b1_n)
        self._b2.append(b2_n)
        self._tot.append(b1_n + b2_n)

    def refresh(self):
        if not self._t:
            return
        t = np.fromiter(self._t, dtype=np.float64)
        self._c1.setData(t, np.fromiter(self._b1, dtype=np.float64))
        self._c2.setData(t, np.fromiter(self._b2, dtype=np.float64))
        self._ct.setData(t, np.fromiter(self._tot, dtype=np.float64))


class COPTrajectory(QWidget):
    """2D CoP trajectory on the force plate footprint (X-Y in mm).

    Minimum widget size (T3): the plate footprint is 558 × 432 mm
    (aspect ≈ 1.29:1). With ``setAspectLocked(True)`` enabled, the
    plot will crop one axis if the widget aspect doesn't match. We
    enforce a minimum 380×300 (aspect 1.27, slightly more square so
    a small label margin always fits) so both Board1 (blue) and
    Board2 (orange) outlines stay fully on-screen no matter how
    narrow the parent splitter pane gets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 300)
        self._xs: collections.deque = collections.deque(maxlen=2000)
        self._ys: collections.deque = collections.deque(maxlen=2000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        # Hard-clamp axes so noisy out-of-plate samples can't force the
        # viewport to autoRange into [-4000, 4000] (makes the board
        # outlines collapse to a speck in the corner + creates the
        # illusion that L/R are swapped).
        # Padding of 50 mm (U2-2) keeps a comfortable visual margin
        # around the outlines and gives the aspect-locked viewport
        # enough slack so neither board outline gets cropped when the
        # widget aspect doesn't match the plate aspect (1.29:1).
        pad = 50.0
        self._plot.setLimits(
            xMin=-pad, xMax=config.PLATE_TOTAL_WIDTH_MM + pad,
            yMin=-pad, yMax=config.PLATE_TOTAL_HEIGHT_MM + pad,
        )
        self._plot.enableAutoRange(axis='xy', enable=False)
        self._plot.setXRange(-pad, config.PLATE_TOTAL_WIDTH_MM + pad,
                              padding=0)
        self._plot.setYRange(-pad, config.PLATE_TOTAL_HEIGHT_MM + pad,
                              padding=0)
        self._plot.setLabel("left", "Y (mm)")
        self._plot.setLabel("bottom", "X (mm)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        # Board outlines
        for (x0, y0), color in [
            (config.BOARD1_ORIGIN_MM, "#4FC3F7"),
            (config.BOARD2_ORIGIN_MM, "#FF8A65"),
        ]:
            w, h = config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM
            rect = pg.PlotDataItem(
                x=[x0, x0+w, x0+w, x0, x0],
                y=[y0, y0, y0+h, y0+h, y0],
                pen=pg.mkPen(color, width=1.5),
            )
            self._plot.addItem(rect)

        self._trail = self._plot.plot(
            pen=pg.mkPen("#CE93D8", width=1.5),
            symbol=None,
        )
        self._point = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("#FFEB3B"), pen=pg.mkPen("#FFC107", width=2),
        )
        self._plot.addItem(self._point)
        # Faded "last-valid" marker — drawn at the most recent CoP
        # position whenever the live sample is being filter-rejected
        # (jump flight, noise spike). Tells the trainer "we lost the
        # signal here — subject left the plate" without a confusing
        # gap or wild trail (U2-4).
        self._frozen_point = pg.ScatterPlotItem(
            size=12,
            brush=pg.mkBrush(255, 235, 59, 80),    # faded yellow ≈ 30% alpha
            pen=pg.mkPen(255, 193, 7, 120, width=2,
                          style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._frozen_point)
        self._frozen_xy: Optional[tuple[float, float]] = None
        self._is_frozen: bool = False

        layout.addWidget(self._plot)

    def push(self, cop_x_mm: float, cop_y_mm: float):
        """Append a valid CoP sample to the live trail."""
        self._xs.append(cop_x_mm)
        self._ys.append(cop_y_mm)
        # Latest valid → also our potential "freeze" reference for
        # subsequent rejected samples.
        self._frozen_xy = (float(cop_x_mm), float(cop_y_mm))
        self._is_frozen = False

    def hold_at_last_valid(self) -> None:
        """Mark current state as 'in flight / signal dropped' so the
        next refresh draws the faded marker instead of the bright one
        at the last good position."""
        self._is_frozen = True

    def resume(self) -> None:
        """Live signal is good again — bring back the bright marker."""
        self._is_frozen = False

    def refresh(self):
        if not self._xs:
            # Even with no trail history, a faded freeze marker may
            # exist (subject just stepped off without ever pushing a
            # valid CoP). Render whatever we have.
            self._trail.setData([], [])
            if self._is_frozen and self._frozen_xy is not None:
                self._frozen_point.setData(
                    [self._frozen_xy[0]], [self._frozen_xy[1]])
                self._point.setData([], [])
            else:
                self._frozen_point.setData([], [])
                self._point.setData([], [])
            return
        xs = np.fromiter(self._xs, dtype=np.float64)
        ys = np.fromiter(self._ys, dtype=np.float64)
        self._trail.setData(xs, ys)
        # Active marker visibility flips between bright (live) and
        # faded (frozen at last valid during flight / drop).
        if self._is_frozen and self._frozen_xy is not None:
            self._frozen_point.setData(
                [self._frozen_xy[0]], [self._frozen_xy[1]])
            self._point.setData([], [])
        else:
            self._frozen_point.setData([], [])
            self._point.setData([xs[-1]], [ys[-1]])

    def clear(self):
        self._xs.clear()
        self._ys.clear()
        self._frozen_xy = None
        self._is_frozen = False
        self._frozen_point.setData([], [])
        self._point.setData([], [])


class EncoderBar(QWidget):
    """Barbell/dumbbell encoders: live value + velocity, power text readout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._enc1 = QLabel("Enc1  ——— mm")
        self._enc2 = QLabel("Enc2  ——— mm")
        for lbl, color in [(self._enc1, "#F9A825"), (self._enc2, "#F57F17")]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color:{color}; font-weight:bold; font-size:14px; "
                f"background:#1a1a1a; padding:6px; border:1px solid #333;"
            )
            layout.addWidget(lbl)

    def update(self, enc1_mm: float, enc2_mm: float):
        self._enc1.setText(f"Enc1   {enc1_mm:7.1f} mm")
        self._enc2.setText(f"Enc2   {enc2_mm:7.1f} mm")
