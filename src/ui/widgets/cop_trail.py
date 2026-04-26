"""
CopTrailWidget — plate outline + CoP trail up to the playback cursor.

Reads force.cop_x / cop_y (100 Hz) from a ForceSession. For a given
``set_time(t)``, redraws:
  - the full path up to t as a faint trail
  - the last ~1.5 s as a brighter lead-in
  - current CoP point as a filled marker
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout

import config
from src.analysis.common import ForceSession, load_force_session
from src.ui.widgets.replay_colors import HOVER_LINE_COLOR


class CopTrailWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Same min-size + aspect padding as Measure-tab COPTrajectory
        # (T3) so both views render the plate identically on tiny
        # report-viewer panes.
        self.setMinimumSize(380, 300)
        self._force: Optional[ForceSession] = None

        pg.setConfigOption("background", "#111")
        pg.setConfigOption("foreground", "#CCC")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        # U2-2: 50 mm padding (was 30) on the limits gives the
        # aspect-locked viewport room to breathe when the parent's
        # widget aspect doesn't exactly match the plate aspect (1.29).
        # Result: board outlines always fit within the viewport even
        # at non-ideal widget shapes.
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
        self._plot.setLabel("left",   "Y (mm)")
        self._plot.setLabel("bottom", "X (mm)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        # Board outlines
        for (x0, y0), color in [
            (config.BOARD1_ORIGIN_MM, "#4FC3F7"),
            (config.BOARD2_ORIGIN_MM, "#FF8A65"),
        ]:
            w, h = config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM
            self._plot.addItem(pg.PlotDataItem(
                x=[x0, x0 + w, x0 + w, x0, x0],
                y=[y0, y0, y0 + h, y0 + h, y0],
                pen=pg.mkPen(color, width=1.5)))

        # Trail and marker
        self._trail_full = self._plot.plot(
            pen=pg.mkPen("#4A148C", width=1), name="trail")       # faded
        self._trail_lead = self._plot.plot(
            pen=pg.mkPen("#CE93D8", width=2), name="recent")      # bright
        self._marker = pg.ScatterPlotItem(
            size=12,
            brush=pg.mkBrush("#FFEB3B"),
            pen=pg.mkPen("#FFC107", width=2))
        self._plot.addItem(self._marker)
        # U2-6: Faded marker for replay's jump-flight / spike-rejected
        # frames. Same visual language as the live ForceDashboard so
        # both views look consistent.
        self._frozen_marker = pg.ScatterPlotItem(
            size=14,
            brush=pg.mkBrush(255, 235, 59, 80),       # 30 % alpha
            pen=pg.mkPen(255, 193, 7, 120, width=2,
                          style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._frozen_marker)
        self._last_valid_xy: Optional[np.ndarray] = None

        # Hover
        self._hover_marker = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush(None),
            pen=pg.mkPen(HOVER_LINE_COLOR, width=2))
        self._hover_marker.setVisible(False)
        self._plot.addItem(self._hover_marker)
        self._hover_label = pg.TextItem(anchor=(0, 0), color=(230, 230, 230))
        self._hover_label.setVisible(False)
        self._plot.addItem(self._hover_label, ignoreBounds=True)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        lay.addWidget(self._plot)

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, session_dir: str | Path) -> bool:
        try:
            self._force = load_force_session(Path(session_dir))
        except Exception:
            self._force = None
            return False

        # U2-6: Apply the same CoP noise filter the live ForceDashboard
        # uses, so jump-flight + velocity-spike samples are masked out
        # in replay too. Recorder writes raw CoP (only the 5 N gate) so
        # this offline pass is necessary to match the live experience.
        try:
            from src.ui.widgets.cop_filter import filter_offline
            import json
            subject_kg = 0.0
            meta_p = Path(session_dir) / "session.json"
            if meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                subject_kg = float(meta.get("subject_kg") or 0.0)
            self._force.cop_x, self._force.cop_y = filter_offline(
                self._force.total, self._force.cop_x, self._force.cop_y,
                subject_kg=subject_kg,
                sample_rate_hz=float(self._force.fs or 100.0),
            )
        except Exception:
            # Filter is best-effort — fall back to raw CoP on any
            # unexpected error so replay never breaks.
            pass

        # Cache last-valid (cx, cy) per sample index for fast lookup
        # in set_time() — the faded marker shown during invalid frames.
        self._last_valid_xy = self._build_last_valid_lookup(
            self._force.cop_x, self._force.cop_y)

        self.set_time(0.0)
        return True

    @staticmethod
    def _build_last_valid_lookup(cop_x: np.ndarray, cop_y: np.ndarray):
        """Per-sample 'most recent valid CoP' table.

        For each index i, stores the (cx, cy) of the latest j ≤ i
        whose CoP is not NaN. Drives the faded freeze marker so even
        during a long flight phase the operator can still see *where
        the subject left the plate*.
        """
        n = len(cop_x)
        out = np.full((n, 2), np.nan, dtype=np.float64)
        last = (np.nan, np.nan)
        for i in range(n):
            if not (np.isnan(cop_x[i]) or np.isnan(cop_y[i])):
                last = (float(cop_x[i]), float(cop_y[i]))
            out[i, 0] = last[0]
            out[i, 1] = last[1]
        return out

    def unload(self) -> None:
        self._force = None
        self._last_valid_xy = None
        self._trail_full.setData([], [])
        self._trail_lead.setData([], [])
        self._marker.setData([], [])
        if hasattr(self, "_frozen_marker"):
            self._frozen_marker.setData([], [])

    def set_time(self, t_s: float) -> None:
        if self._force is None:
            return
        t = self._force.t_s
        idx = int(np.clip(np.searchsorted(t, float(t_s)), 0, len(t) - 1))
        # Full trail up to cursor, with NaN-masked CoP dropped
        x_full = self._force.cop_x[:idx + 1]
        y_full = self._force.cop_y[:idx + 1]
        mask_full = ~(np.isnan(x_full) | np.isnan(y_full))
        self._trail_full.setData(x_full[mask_full], y_full[mask_full])
        # Recent 1.5 s (brighter)
        lead_start = max(0, idx - 150)   # ~1.5s at 100Hz
        x_lead = self._force.cop_x[lead_start:idx + 1]
        y_lead = self._force.cop_y[lead_start:idx + 1]
        mask_lead = ~(np.isnan(x_lead) | np.isnan(y_lead))
        self._trail_lead.setData(x_lead[mask_lead], y_lead[mask_lead])
        # Current marker — bright yellow when this sample passed both
        # filter gates; faded freeze marker at the most recent valid
        # position when the current sample was rejected (jump flight,
        # noise spike). U2-6/U2-7.
        cx = self._force.cop_x[idx]
        cy = self._force.cop_y[idx]
        if not (np.isnan(cx) or np.isnan(cy)):
            self._marker.setData([cx], [cy])
            self._frozen_marker.setData([], [])
        else:
            self._marker.setData([], [])
            # Pull the last-valid lookup we built at load() time.
            if (self._last_valid_xy is not None
                    and idx < len(self._last_valid_xy)):
                fx = self._last_valid_xy[idx, 0]
                fy = self._last_valid_xy[idx, 1]
                if not (np.isnan(fx) or np.isnan(fy)):
                    self._frozen_marker.setData([fx], [fy])
                else:
                    self._frozen_marker.setData([], [])
            else:
                self._frozen_marker.setData([], [])

    # ── hover ──────────────────────────────────────────────────────────────
    def _on_mouse_moved(self, scene_pos) -> None:
        if self._force is None:
            self._hover_marker.setVisible(False)
            self._hover_label.setVisible(False)
            return
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            self._hover_marker.setVisible(False)
            self._hover_label.setVisible(False)
            return
        try:
            vp = self._plot.plotItem.vb.mapSceneToView(scene_pos)
        except Exception:
            return
        mx, my = float(vp.x()), float(vp.y())
        # Nearest CoP sample (within the trail-so-far is ambiguous; use full)
        xs = self._force.cop_x
        ys = self._force.cop_y
        mask = (~np.isnan(xs)) & (~np.isnan(ys))
        if not mask.any():
            return
        xs_v, ys_v = xs[mask], ys[mask]
        ts_v = self._force.t_s[mask]
        dx = xs_v - mx
        dy = ys_v - my
        i = int(np.argmin(dx * dx + dy * dy))
        dist = float(np.sqrt(dx[i] * dx[i] + dy[i] * dy[i]))
        if dist > 60.0:     # 60 mm radius
            self._hover_marker.setVisible(False)
            self._hover_label.setVisible(False)
            return
        self._hover_marker.setData([float(xs_v[i])], [float(ys_v[i])])
        self._hover_marker.setVisible(True)
        self._hover_label.setHtml(
            f"<div style='background:rgba(20,20,20,200); padding:3px 6px;'>"
            f"t = {float(ts_v[i]):5.2f}s &nbsp; "
            f"CoP ({float(xs_v[i]):.0f}, {float(ys_v[i]):.0f}) mm"
            f"</div>"
        )
        vr = self._plot.plotItem.vb.viewRange()
        self._hover_label.setPos(vr[0][0], vr[1][1])
        self._hover_label.setVisible(True)
