"""
DepartureSlider — QSlider that paints small tick marks for off-plate
event start times. Used as the playback slider in ReplayPanel
(Phase U3-3).

The base behaviour is identical to QSlider; the only addition is an
overlay painted in ``paintEvent`` after the default groove/handle. Tick
marks are positioned proportionally based on each event's ``t_start_s``
relative to the session duration.

Keyboard shortcuts ``[`` / ``]`` (wired in ReplayPanel) jump the cursor
to the previous / next departure start.
"""
from __future__ import annotations

from typing import Iterable

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


class DepartureSlider(QSlider):
    """Horizontal slider with departure-event tick overlay."""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        # Each entry is a t_start_s relative to session start.
        self._tick_times_s: list[float] = []
        # Session duration in seconds — needed to map times to slider
        # value-space. Set by ReplayPanel.load_session().
        self._duration_s: float = 0.0

    # ── public API ─────────────────────────────────────────────────────────
    def set_departure_ticks(self,
                            tick_times_s: Iterable[float],
                            duration_s: float) -> None:
        """Configure tick positions + their reference duration.

        The slider value range is unchanged (typically 0..1000); ticks
        are painted at ``t_start / duration_s × max_value`` positions.
        Out-of-range entries are silently dropped.
        """
        self._duration_s = max(0.0, float(duration_s))
        if self._duration_s <= 0:
            self._tick_times_s = []
        else:
            self._tick_times_s = [
                float(t) for t in tick_times_s
                if 0.0 <= float(t) <= self._duration_s
            ]
        self.update()       # repaint

    # ── painting ───────────────────────────────────────────────────────────
    def paintEvent(self, event):
        # Let the base class paint groove + handle first
        super().paintEvent(event)
        if not self._tick_times_s or self._duration_s <= 0:
            return
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        # Get the groove rectangle so ticks line up with it horizontally
        groove_rect: QRect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, self)
        if groove_rect.width() <= 0:
            return
        # Handle width — used to inset tick range so ticks at t=0 / t=end
        # don't get clipped by the handle's edge inflation
        handle_rect: QRect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderHandle, self)
        h_half = handle_rect.width() // 2

        track_left  = groove_rect.left()  + h_half
        track_right = groove_rect.right() - h_half
        track_w = max(1, track_right - track_left)
        track_top    = groove_rect.top()    - 4
        track_bot    = groove_rect.bottom() + 4

        painter = QPainter(self)
        # Same orange as the LinearRegionItem fill so the slider tick
        # visually couples with the timeline bands.
        pen = QPen(QColor(255, 152, 0, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        for t in self._tick_times_s:
            frac = t / self._duration_s
            x = int(track_left + frac * track_w)
            painter.drawLine(x, track_top, x, track_bot)
        painter.end()
