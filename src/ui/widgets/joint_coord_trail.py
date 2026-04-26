"""
JointCoordTrail — single 2D pixel-space chart showing up to 2 joint paths.

For each selected MediaPipe-33 joint slot, renders the pixel-coordinate
trail up to the current playback cursor, plus a marker at the cursor.
Y axis is image-down (origin top-left) to match the displayed video.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from src.analysis.pose2d import Pose2DSeries
from src.pose.mediapipe_backend import MP33
from src.ui.widgets.replay_colors import COORD_COLORS, HOVER_LINE_COLOR


N_SLOTS = 2

# Subset of the 33-landmark set useful for biomech trajectory study.
COORD_CHOICES: list[str] = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
    "left_foot_index","right_foot_index",
]


class JointCoordTrail(QWidget):
    """2D pixel-space path chart for up to N_SLOTS joints."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pose: Optional[Pose2DSeries] = None
        self._image_size: Optional[tuple[int, int]] = None
        self._slot_joints: list[Optional[str]] = [None] * N_SLOTS
        self._current_idx: int = 0

        pg.setConfigOption("background", "#111")
        pg.setConfigOption("foreground", "#CCC")

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        # Y-down to match image coords; caller sets actual range on load()
        self._plot.invertY(True)
        self._plot.setLabel("left",   "Y (px, down)")
        self._plot.setLabel("bottom", "X (px)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        root.addWidget(self._plot)

        # Per-slot trail + marker items (created up-front)
        self._trails:   list[pg.PlotDataItem] = []
        self._markers:  list[pg.ScatterPlotItem] = []
        for i in range(N_SLOTS):
            trail = self._plot.plot(
                pen=pg.mkPen(COORD_COLORS[i], width=2), name=f"slot{i+1}")
            marker = pg.ScatterPlotItem(
                size=10,
                brush=pg.mkBrush(COORD_COLORS[i]),
                pen=pg.mkPen("white", width=1))
            self._plot.addItem(marker)
            self._trails.append(trail)
            self._markers.append(marker)

        # Hover crosshair + label (single, shared by both slots)
        self._hover_marker = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush(None),
            pen=pg.mkPen(HOVER_LINE_COLOR, width=2))
        self._plot.addItem(self._hover_marker)
        self._hover_marker.setVisible(False)
        self._hover_label = pg.TextItem(anchor=(0, 0), color=(230, 230, 230))
        self._plot.addItem(self._hover_label, ignoreBounds=True)
        self._hover_label.setVisible(False)
        self._plot.scene().sigMouseMoved.connect(self._on_hover)

    # ── public API ─────────────────────────────────────────────────────────
    def load(self, pose: Pose2DSeries) -> None:
        self._pose = pose
        w, h = pose.image_size
        self._image_size = (int(w), int(h))
        self._plot.setXRange(0, int(w), padding=0)
        self._plot.setYRange(0, int(h), padding=0)
        self.set_frame_index(0)
        # Re-draw trails with any already-selected slots
        for i, name in enumerate(self._slot_joints):
            self._redraw_slot(i)

    def clear(self) -> None:
        self._pose = None
        self._image_size = None
        for t in self._trails:
            t.setData([], [])
        for m in self._markers:
            m.setData([], [])
        self._slot_joints = [None] * N_SLOTS
        self._current_idx = 0

    def set_joint(self, slot: int, joint_name: Optional[str]) -> None:
        if not (0 <= slot < N_SLOTS):
            return
        self._slot_joints[slot] = joint_name
        self._redraw_slot(slot)

    def set_frame_index(self, idx: int) -> None:
        """Called per playback tick. Updates trails (0..idx) and markers."""
        if self._pose is None:
            return
        self._current_idx = int(np.clip(idx, 0, len(self._pose) - 1))
        for slot in range(N_SLOTS):
            self._redraw_slot(slot)

    # ── internals ──────────────────────────────────────────────────────────
    def _redraw_slot(self, slot: int) -> None:
        name = self._slot_joints[slot]
        trail = self._trails[slot]
        marker = self._markers[slot]
        if (name is None or self._pose is None
                or name not in MP33):
            trail.setData([], [])
            marker.setData([], [])
            return
        j = MP33[name]
        xs = self._pose.kpts_mp33[:self._current_idx + 1, j, 0]
        ys = self._pose.kpts_mp33[:self._current_idx + 1, j, 1]
        vis = self._pose.vis_mp33[:self._current_idx + 1, j]
        mask = (~np.isnan(xs)) & (~np.isnan(ys)) & (vis >= 0.3)
        if mask.any():
            trail.setData(xs[mask], ys[mask],
                          pen=pg.mkPen(COORD_COLORS[slot], width=2))
            # Current marker = last valid sample up to current_idx
            last_valid = int(np.flatnonzero(mask)[-1])
            marker.setData([xs[last_valid]], [ys[last_valid]])
        else:
            trail.setData([], [])
            marker.setData([], [])

    # ── hover ──────────────────────────────────────────────────────────────
    def _on_hover(self, scene_pos) -> None:
        if self._pose is None:
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

        # Find nearest trail point across both slots
        best = None
        best_slot = None
        for slot in range(N_SLOTS):
            name = self._slot_joints[slot]
            if name is None or name not in MP33:
                continue
            j = MP33[name]
            xs = self._pose.kpts_mp33[:self._current_idx + 1, j, 0]
            ys = self._pose.kpts_mp33[:self._current_idx + 1, j, 1]
            vis = self._pose.vis_mp33[:self._current_idx + 1, j]
            mask = (~np.isnan(xs)) & (~np.isnan(ys)) & (vis >= 0.3)
            if not mask.any():
                continue
            dx = xs[mask] - mx
            dy = ys[mask] - my
            d2 = dx * dx + dy * dy
            i_rel = int(np.argmin(d2))
            i_abs = int(np.flatnonzero(mask)[i_rel])
            dist = float(np.sqrt(d2[i_rel]))
            if best is None or dist < best[0]:
                best = (dist, i_abs, float(xs[mask][i_rel]),
                        float(ys[mask][i_rel]), name)
                best_slot = slot
        if best is None or best[0] > 80:     # 80 px search radius
            self._hover_marker.setVisible(False)
            self._hover_label.setVisible(False)
            return
        _, i_abs, x, y, name = best
        self._hover_marker.setData([x], [y])
        self._hover_marker.setVisible(True)
        self._hover_label.setHtml(
            f"<div style='background:rgba(20,20,20,200); padding:3px 6px;'>"
            f"frame {i_abs} &nbsp; "
            f"<span style='color:{COORD_COLORS[best_slot]};font-weight:bold'>"
            f"{name}</span> ({x:.0f}, {y:.0f}) px"
            f"</div>"
        )
        vr = self._plot.plotItem.vb.viewRange()
        self._hover_label.setPos(vr[0][0], vr[1][0])
        self._hover_label.setVisible(True)
