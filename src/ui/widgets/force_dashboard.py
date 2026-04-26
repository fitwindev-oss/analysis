"""
ForceDashboard — trainer's live-force readout panel.

Composition:
    ┌─────────────────────────────────────────────────────────────┐
    │ Total: 912 N (93.0 kg)   Board1: 431 N   Board2: 481 N       │
    ├────────────────────────────┬────────────────────────────────┤
    │  VGRF time-series (10 s)    │  CoP trajectory on plate       │
    └────────────────────────────┴────────────────────────────────┘

Consumes DaqFrame objects via on_daq_frame(). Repainting is throttled by an
internal QTimer to protect the GUI thread from 100 Hz DAQ traffic.
"""
from __future__ import annotations

import collections
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QFrame,
)

import config
from src.capture.daq_reader import DaqFrame
from src.ui.widgets.force_widgets import VGRFPlot, COPTrajectory


G = 9.80665


class _BigReadout(QWidget):
    """Large labelled number with colour-coded state."""

    def __init__(self, title: str, color: str = "#A5D6A7", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        self._title = QLabel(title)
        self._title.setStyleSheet("color:#bbb; font-size:11px;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value = QLabel("— — —")
        f = QFont(); f.setPointSize(18); f.setBold(True)
        self._value.setFont(f)
        self._value.setStyleSheet(f"color:{color};")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._title)
        lay.addWidget(self._value)
        self.setStyleSheet(
            "QWidget { background:#161616; border:1px solid #2a2a2a; }")

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class ForceDashboard(QWidget):
    """Trainer's live force panel. Call on_daq_frame(DaqFrame) per sample.

    ``orientation`` controls the layout:
      - "horizontal" (default): readouts row, then VGRF | CoP side-by-side
      - "vertical":              CoP (top), readouts, VGRF (bottom) — used
                                  in the redesigned Measure-tab right panel
    """

    def __init__(self, parent=None, orientation: str = "horizontal"):
        super().__init__(parent)
        self._latest: Optional[DaqFrame] = None
        self._subject_kg: float = 0.0
        # Live-display noise suppression. When False AND total_n is
        # below the config threshold, VGRF is clamped to 0 and CoP push
        # is skipped so the dashboard shows a clean idle state. The
        # suppression is disabled during active recording so jump-flight
        # signal is never lost.
        self._recording: bool = False
        # CoP real-time noise filter (Phase U2-4) — gates the CoP push
        # by force-context (≥20 % BW) AND velocity (<3 m/s) so jump
        # flight + spike noise don't pollute the trail.
        from src.ui.widgets.cop_filter import CoPFilter
        self._cop_filter = CoPFilter(subject_kg=self._subject_kg)

        # ── Layout ───────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Readouts row (shared layout between orientations)
        readouts = QHBoxLayout()
        readouts.setSpacing(4)
        self._r_total = _BigReadout("Total (N)",  "#A5D6A7")
        self._r_kg    = _BigReadout("Total (kg)", "#FFF176")
        self._r_b1    = _BigReadout("Board1 (N)", "#4FC3F7")
        self._r_b2    = _BigReadout("Board2 (N)", "#FF8A65")
        self._r_bwpct = _BigReadout("BW %",       "#CE93D8")
        # Reps readout — hidden by default; MeasureTab shows it only for
        # free_exercise when the encoder is in use. Value is pushed by
        # set_rep_count(int) from the live RealtimeRepCounter.
        self._r_reps  = _BigReadout("Reps",       "#FFEB3B")
        self._r_reps.setVisible(False)
        for r in (self._r_total, self._r_kg, self._r_b1, self._r_b2,
                  self._r_bwpct, self._r_reps):
            readouts.addWidget(r, stretch=1)

        self._vgrf = VGRFPlot(window_s=10.0)
        self._cop  = COPTrajectory()

        if orientation == "vertical":
            # CoP (top, bigger) — readouts — VGRF (bottom)
            root.addWidget(self._cop, stretch=3)
            root.addLayout(readouts)
            root.addWidget(self._vgrf, stretch=2)
        else:
            root.addLayout(readouts)
            plots = QHBoxLayout()
            plots.setSpacing(4)
            plots.addWidget(self._vgrf, stretch=3)
            plots.addWidget(self._cop,  stretch=2)
            root.addLayout(plots, stretch=1)

        # Refresh timer (~30 fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._repaint)
        self._timer.start(config.PLOT_UPDATE_MS)

    # ── public API ─────────────────────────────────────────────────────────
    def set_subject_weight(self, kg: float) -> None:
        self._subject_kg = max(0.0, float(kg))
        # Re-tune the CoP force gate to this subject's bodyweight.
        self._cop_filter.set_subject_kg(self._subject_kg)

    def set_rep_counter_visible(self, on: bool) -> None:
        """Show/hide the "Reps" readout. MeasureTab enables this only for
        free_exercise sessions with encoder on."""
        self._r_reps.setVisible(bool(on))
        if not on:
            self._r_reps.set_value("— — —")

    def set_rep_count(self, n: int) -> None:
        self._r_reps.set_value(f"{int(n):d}")

    def set_recording(self, on: bool) -> None:
        """When True, live display shows full raw signal (needed so jump
        flight doesn't get clamped to zero). When False, signals below
        ``config.LIVE_DISPLAY_MIN_N`` are suppressed so idle DAQ noise
        doesn't scroll across the plots between sessions."""
        self._recording = bool(on)

    def on_daq_frame(self, fr: DaqFrame) -> None:
        """Called from GUI thread (queued signal from worker)."""
        self._latest = fr
        # Phase-aware noise suppression: when we're NOT recording and the
        # total force is under the idle noise floor, render a clean zero
        # baseline on VGRF and skip the CoP marker. Recording always
        # passes the raw signal so jump-flight / CMJ bottom-out shows.
        threshold = float(getattr(config, "LIVE_DISPLAY_MIN_N", 0.0))
        suppressed = (not self._recording) and (fr.total_n < threshold)

        if suppressed:
            self._vgrf.push(fr.t_wall, 0.0, 0.0)
            # CoP: idle suppression — freeze trail at its last valid
            # position. The faded marker gives the trainer a visual
            # anchor instead of a confusing empty plot.
            self._cop.hold_at_last_valid()
        else:
            self._vgrf.push(fr.t_wall, fr.b1_total_n, fr.b2_total_n)
            cx, cy = fr.cop_world_mm()
            # U2-4: route through CoPFilter — gates flight-phase + spike
            # noise during active recording. Returns False when the
            # current sample is in jump flight (force < 20 % BW) or a
            # velocity outlier (>3 m/s implied jump). On rejection we
            # display the faded "last-valid" marker instead of pushing
            # a wild value into the trail.
            if not np.isnan(cx) and self._cop_filter.accept(
                    fr.total_n, cx, cy):
                self._cop.push(cx, cy)
                self._cop.resume()
            else:
                # Filter rejected — keep the trail intact and show
                # the faded marker at last-valid position.
                self._cop.hold_at_last_valid()

    def reset(self) -> None:
        self._latest = None
        self._cop.clear()
        # CoP filter — drop the last-valid reference so a stale
        # position from a prior session can't influence the next one.
        self._cop_filter.reset()
        # Rebuild VGRFPlot buffers — simplest is clearing via internal deques
        self._vgrf._t.clear(); self._vgrf._b1.clear()
        self._vgrf._b2.clear(); self._vgrf._tot.clear()
        self._vgrf._t0 = None
        self._r_total.set_value("— — —")
        self._r_kg.set_value("— — —")
        self._r_b1.set_value("— — —")
        self._r_b2.set_value("— — —")
        self._r_bwpct.set_value("— — —")
        self._r_reps.set_value("— — —")

    # ── internals ──────────────────────────────────────────────────────────
    def _repaint(self) -> None:
        # Readouts (use latest sample, not buffered — zero display lag)
        fr = self._latest
        if fr is not None:
            threshold = float(getattr(config, "LIVE_DISPLAY_MIN_N", 0.0))
            suppressed = (not self._recording) and (fr.total_n < threshold)
            if suppressed:
                # Idle state — show placeholders instead of noise numbers.
                self._r_total.set_value("— — —")
                self._r_kg.set_value("— — —")
                self._r_b1.set_value("— — —")
                self._r_b2.set_value("— — —")
                self._r_bwpct.set_value("— — —")
            else:
                total = fr.total_n
                b1 = fr.b1_total_n
                b2 = fr.b2_total_n
                kg = total / G if total > 0 else 0.0
                self._r_total.set_value(f"{total:7.1f}")
                self._r_kg.set_value(f"{kg:6.1f}")
                self._r_b1.set_value(f"{b1:6.1f}")
                self._r_b2.set_value(f"{b2:6.1f}")
                if self._subject_kg > 0:
                    pct = 100.0 * kg / self._subject_kg
                    self._r_bwpct.set_value(f"{pct:5.1f}")
                else:
                    self._r_bwpct.set_value("— — —")

        # Plots (buffer-driven)
        self._vgrf.refresh()
        self._cop.refresh()
