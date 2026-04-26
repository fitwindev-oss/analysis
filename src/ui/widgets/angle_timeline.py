"""
AngleTimelineStack — 1~3 stacked subplots showing selected joint angles.

Each slot is driven independently by ``set_angle(slot, angle_name | None)``.
Selecting ``None`` hides the corresponding subplot. The X axis is
force-timeline seconds (same as ForceTimelineWidget), and a shared
vertical cursor indicates playback position. A separate dashed crosshair
+ top-right hover label shows the value at the mouse position.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from src.analysis.pose2d import (
    Pose2DSeries, load_video_timestamps, get_record_start_wall_s,
)
from src.ui.widgets.replay_colors import ANGLE_COLORS, HOVER_LINE_COLOR


N_SLOTS = 3


class AngleTimelineStack(QWidget):
    """Vertical stack of up to 3 angle subplots. All share a playback cursor."""

    seek_requested = pyqtSignal(float)   # emitted when user clicks on any subplot

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pose: Optional[Pose2DSeries] = None
        self._t_s: Optional[np.ndarray] = None     # per-frame force-time (sec)
        self._slot_angles: list[Optional[str]] = [None] * N_SLOTS
        # Departure intervals applied to every slot's plot. Stored
        # at the stack level so set_departures touches one source of
        # truth and re-broadcasts on any slot toggle.
        self._departure_intervals: list[tuple[float, float]] = []
        # Per-slot region item lists, parallel to self._plots.
        self._departure_items: list[list[pg.LinearRegionItem]] = [
            [] for _ in range(N_SLOTS)
        ]

        pg.setConfigOption("background", "#111")
        pg.setConfigOption("foreground", "#CCC")

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        # Pre-create 3 plots, hide those without a selection
        self._plots:   list[pg.PlotWidget] = []
        self._curves:  list[pg.PlotDataItem] = []
        self._cursors: list[pg.InfiniteLine] = []
        self._hover_lines: list[pg.InfiniteLine] = []
        self._hover_labels: list[pg.TextItem] = []
        for i in range(N_SLOTS):
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setLabel("left", "°")
            plot.setLabel("bottom", "t (s)")
            plot.setMinimumHeight(110)
            plot.setVisible(False)
            curve = plot.plot(pen=pg.mkPen(ANGLE_COLORS[i], width=2))

            cursor = pg.InfiniteLine(
                pos=0.0, angle=90, movable=True,
                pen=pg.mkPen("#FFEB3B", width=2),
                hoverPen=pg.mkPen("#FFF59D", width=3),
            )
            plot.addItem(cursor)

            hover_line = pg.InfiniteLine(
                pos=0.0, angle=90, movable=False,
                pen=pg.mkPen(HOVER_LINE_COLOR, width=1, style=Qt.PenStyle.DashLine),
            )
            hover_line.setVisible(False)
            plot.addItem(hover_line, ignoreBounds=True)

            hover_label = pg.TextItem(anchor=(0, 0), color=(230, 230, 230))
            hover_label.setVisible(False)
            plot.addItem(hover_label, ignoreBounds=True)

            # Close over slot index
            idx = i

            def _make_moved(i=idx):
                def _fn():
                    self.seek_requested.emit(float(self._cursors[i].value()))
                return _fn
            cursor.sigPositionChanged.connect(_make_moved())

            def _make_clicked(i=idx):
                def _fn(event):
                    try:
                        pt = event.scenePos()
                        vp = self._plots[i].plotItem.vb.mapSceneToView(pt)
                        t = float(vp.x())
                    except Exception:
                        return
                    self.set_cursor(t)
                    self.seek_requested.emit(t)
                return _fn
            plot.scene().sigMouseClicked.connect(_make_clicked())

            def _make_hover(i=idx):
                def _fn(scene_pos):
                    self._on_hover(i, scene_pos)
                return _fn
            plot.scene().sigMouseMoved.connect(_make_hover())

            root.addWidget(plot)
            self._plots.append(plot)
            self._curves.append(curve)
            self._cursors.append(cursor)
            self._hover_lines.append(hover_line)
            self._hover_labels.append(hover_label)

        # Minimum height is recomputed per visible slot count so that
        # toggling slots on/off actually grows/shrinks the widget.
        self._update_min_height()

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, session_dir: str | Path,
             cam_id: str, pose: Pose2DSeries) -> None:
        """Attach a pose series so slots can be populated.

        Computes the per-frame force-time axis from the camera's
        timestamps.csv + session.json's record_start_wall_s.
        """
        self._pose = pose
        walls = load_video_timestamps(session_dir, cam_id)
        rec = get_record_start_wall_s(session_dir)
        if walls is not None and rec is not None and len(walls) == len(pose):
            self._t_s = (walls - rec).astype(np.float64)
        else:
            # Fallback to fps-based pseudo-time
            self._t_s = np.arange(len(pose), dtype=np.float64) / max(pose.fps, 1.0)
        # Re-apply any already-set slots to refresh
        for i, name in enumerate(self._slot_angles):
            self._refresh_slot(i, name)

    def clear(self) -> None:
        self._pose = None
        self._t_s = None
        for i in range(N_SLOTS):
            self._refresh_slot(i, None)
        self._slot_angles = [None] * N_SLOTS
        self.set_departures([])

    def set_departures(self, intervals: list[tuple[float, float]]) -> None:
        """Apply off-plate region overlays to every slot's plot.

        The intervals are stored at the stack level so that re-rendering
        a slot (set_angle) automatically re-applies the bands; otherwise
        toggling a previously hidden slot would lose the bands.
        Phase U3-3.
        """
        self._departure_intervals = [
            (float(a), float(b)) for (a, b) in intervals
        ]
        for slot in range(N_SLOTS):
            self._apply_departures_to_slot(slot)

    def _apply_departures_to_slot(self, slot: int) -> None:
        """Re-create LinearRegionItems on slot's plot from current intervals."""
        plot = self._plots[slot]
        # Clear previous regions on this slot
        for item in self._departure_items[slot]:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._departure_items[slot].clear()
        # Add fresh ones
        for (t0, t1) in self._departure_intervals:
            region = pg.LinearRegionItem(
                values=(t0, t1),
                brush=pg.mkBrush(255, 152, 0, 50),
                pen=pg.mkPen(255, 152, 0, 130, width=1),
                movable=False,
            )
            region.setZValue(-10)
            plot.addItem(region)
            self._departure_items[slot].append(region)

    def set_angle(self, slot: int, angle_name: Optional[str]) -> None:
        """Select (or clear with ``None``) the angle for slot [0..N_SLOTS)."""
        if not (0 <= slot < N_SLOTS):
            return
        self._slot_angles[slot] = angle_name
        self._refresh_slot(slot, angle_name)

    def set_cursor(self, t_s: float) -> None:
        for c in self._cursors:
            c.blockSignals(True)
            c.setValue(float(t_s))
            c.blockSignals(False)

    # ── internals ──────────────────────────────────────────────────────────
    def _refresh_slot(self, slot: int, angle_name: Optional[str]) -> None:
        plot = self._plots[slot]
        curve = self._curves[slot]
        if angle_name is None or self._pose is None or self._t_s is None:
            curve.setData([], [])
            plot.setVisible(False)
            self._update_min_height()
            return
        names = list(self._pose.angle_names)
        if angle_name not in names:
            curve.setData([], [])
            plot.setVisible(False)
            self._update_min_height()
            return
        j = names.index(angle_name)
        col = self._pose.angles[:, j]
        # Plot only finite values — pyqtgraph handles NaN gaps via connect="finite"
        curve.setData(
            x=self._t_s,
            y=col,
            pen=pg.mkPen(ANGLE_COLORS[slot], width=2),
            connect="finite",
        )
        plot.setTitle(f"<span style='color:{ANGLE_COLORS[slot]};"
                      f"font-weight:bold'>{angle_name}</span>",
                      color="#ddd", size="10pt")
        # Auto-range to visible data
        finite = col[~np.isnan(col)]
        if len(finite) > 1:
            lo, hi = float(finite.min()), float(finite.max())
            pad = max(2.0, 0.08 * (hi - lo))
            plot.setYRange(lo - pad, hi + pad, padding=0)
        if len(self._t_s) > 1:
            plot.setXRange(float(self._t_s[0]), float(self._t_s[-1]), padding=0)
        plot.setVisible(True)
        # Re-apply any departure bands — they may have been lost when
        # the slot was previously hidden / cleared.
        self._apply_departures_to_slot(slot)
        self._update_min_height()

    def _update_min_height(self) -> None:
        visible = sum(1 for p in self._plots if p.isVisible())
        if visible == 0:
            self.setMinimumHeight(8)
        else:
            # Matches each plot's own minimumHeight + small padding
            self.setMinimumHeight(110 * visible + 6)

    # ── hover ──────────────────────────────────────────────────────────────
    def _on_hover(self, slot: int, scene_pos) -> None:
        plot = self._plots[slot]
        curve = self._curves[slot]
        hover_line = self._hover_lines[slot]
        hover_label = self._hover_labels[slot]
        if not plot.isVisible() or self._pose is None or self._t_s is None:
            hover_line.setVisible(False)
            hover_label.setVisible(False)
            return
        if not plot.sceneBoundingRect().contains(scene_pos):
            hover_line.setVisible(False)
            hover_label.setVisible(False)
            return
        try:
            view_pt = plot.plotItem.vb.mapSceneToView(scene_pos)
        except Exception:
            return
        t = float(view_pt.x())
        # Find nearest frame
        idx = int(np.clip(np.searchsorted(self._t_s, t), 0, len(self._t_s) - 1))
        name = self._slot_angles[slot]
        names = list(self._pose.angle_names)
        if name not in names:
            return
        j = names.index(name)
        val = float(self._pose.angles[idx, j]) if not np.isnan(
            self._pose.angles[idx, j]) else None
        hover_line.setPos(t)
        hover_line.setVisible(True)
        val_txt = "—" if val is None else f"{val:6.1f}°"
        hover_label.setHtml(
            f"<div style='background:rgba(20,20,20,200); padding:3px 6px;'>"
            f"t = {t:5.2f}s &nbsp; "
            f"<span style='color:{ANGLE_COLORS[slot]};font-weight:bold'>"
            f"{name}</span> = {val_txt}"
            f"</div>"
        )
        # Anchor to top-left of view
        view_range = plot.plotItem.vb.viewRange()
        hover_label.setPos(view_range[0][0], view_range[1][1])
        hover_label.setVisible(True)
