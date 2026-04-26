"""3D skeleton viewer using pyqtgraph OpenGL.

Shows:
  - force plate footprint on the ground plane (X=0..558 mm, Y=0..432 mm)
  - world coordinate axes at origin
  - joint markers + bone lines when 3D keypoints are fed via `update_skeleton`.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel

import config

# COCO-17 bone pairs (index into KPT_NAMES)
COCO_BONES = [
    (5, 7),   (7, 9),     # left arm
    (6, 8),   (8, 10),    # right arm
    (5, 6),               # shoulders
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (11, 12),             # hips
    (5, 11),  (6, 12),    # torso
    (0, 5),   (0, 6),     # head → shoulders
]


class Skeleton3DView(QWidget):
    """Right-most tile of the top row. Placeholder grid until 3D data arrives."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gl_available = False
        self._gl_view = None
        self._scatter = None
        self._bone_lines = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        header = QLabel("3D Skeleton")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "color:#CE93D8; font-weight:bold; font-size:11px; padding:2px;"
        )
        layout.addWidget(header)

        try:
            import pyqtgraph.opengl as gl
            self._gl_view = gl.GLViewWidget()
            self._gl_view.setBackgroundColor(10, 10, 15)
            self._gl_view.setMinimumSize(config.TILE_WIDTH, config.TILE_HEIGHT)
            self._gl_view.opts["distance"] = 2500  # mm
            self._gl_view.opts["elevation"] = 20
            self._gl_view.opts["azimuth"] = -75
            layout.addWidget(self._gl_view, stretch=1)
            self._init_scene(gl)
            self._gl_available = True
        except Exception as e:
            fallback = QLabel(
                f"3D view unavailable (pyqtgraph.opengl).\n\n"
                f"Install: pip install PyOpenGL pyqtgraph\n\nReason: {e}"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet(
                "background:#111; border:1px solid #333; color:#888;"
            )
            fallback.setWordWrap(True)
            fallback.setMinimumSize(config.TILE_WIDTH, config.TILE_HEIGHT)
            layout.addWidget(fallback, stretch=1)

        self._status = QLabel("Waiting for 3D keypoints…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color:#777; font-size:10px;")
        layout.addWidget(self._status)

    # ── scene setup ──────────────────────────────────────────────────────────
    def _init_scene(self, gl):
        # Ground grid
        grid = gl.GLGridItem()
        grid.setSize(x=2000, y=2000)
        grid.setSpacing(x=100, y=100)
        grid.setColor((100, 100, 100, 100))
        self._gl_view.addItem(grid)

        # World axes at origin (0, 0, 0)
        axis_len = 200
        for direction, color in [
            ((axis_len, 0, 0), (1, 0, 0, 1)),     # X red
            ((0, axis_len, 0), (0, 1, 0, 1)),     # Y green
            ((0, 0, axis_len), (0, 0, 1, 1)),     # Z blue
        ]:
            pts = np.array([[0, 0, 0], list(direction)], dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=color, width=3, antialias=True)
            self._gl_view.addItem(line)

        # Force plate footprint on ground (Z=0) — Board1 + Board2
        self._add_plate_outline(gl,
            *config.BOARD1_ORIGIN_MM, config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM,
            color=(0.31, 0.76, 0.97, 1.0))
        self._add_plate_outline(gl,
            *config.BOARD2_ORIGIN_MM, config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM,
            color=(1.0, 0.54, 0.40, 1.0))

        # Pre-allocate scatter + bones for skeleton updates
        n = len(config.KPT_NAMES)
        empty = np.full((n, 3), np.nan, dtype=np.float32)
        self._scatter = gl.GLScatterPlotItem(
            pos=empty, color=(1, 1, 1, 1), size=8,
            pxMode=True,
        )
        self._gl_view.addItem(self._scatter)

        for (a, b) in COCO_BONES:
            line = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
                color=(0.9, 0.9, 0.3, 1.0), width=2, antialias=True,
            )
            self._bone_lines.append(line)
            self._gl_view.addItem(line)

    def _add_plate_outline(self, gl, x0: float, y0: float, w: float, h: float,
                           color: tuple):
        import pyqtgraph.opengl as glmod
        corners = np.array([
            [x0,     y0,     0],
            [x0 + w, y0,     0],
            [x0 + w, y0 + h, 0],
            [x0,     y0 + h, 0],
            [x0,     y0,     0],
        ], dtype=np.float32)
        line = glmod.GLLinePlotItem(pos=corners, color=color, width=2, antialias=True)
        self._gl_view.addItem(line)

    # ── public API ───────────────────────────────────────────────────────────
    def update_skeleton(self, kpts_3d: np.ndarray) -> None:
        """
        kpts_3d: (17, 3) in world mm. NaN for missing joints.
        """
        if not self._gl_available or self._scatter is None:
            return
        if kpts_3d.shape != (17, 3):
            return
        # Scatter: drop NaN rows to avoid rendering them
        valid = ~np.isnan(kpts_3d).any(axis=1)
        self._scatter.setData(pos=kpts_3d[valid].astype(np.float32))

        # Bones
        for idx, (a, b) in enumerate(COCO_BONES):
            if not (valid[a] and valid[b]):
                empty = np.full((2, 3), np.nan, dtype=np.float32)
                self._bone_lines[idx].setData(pos=empty)
                continue
            seg = np.stack([kpts_3d[a], kpts_3d[b]]).astype(np.float32)
            self._bone_lines[idx].setData(pos=seg)

        n_valid = int(valid.sum())
        self._status.setText(f"Tracking {n_valid}/17 joints")
        self._status.setStyleSheet("color:#66BB6A; font-size:10px;")

    def clear(self):
        if self._gl_available and self._scatter is not None:
            empty = np.full((17, 3), np.nan, dtype=np.float32)
            self._scatter.setData(pos=empty)
            for line in self._bone_lines:
                line.setData(pos=np.full((2, 3), np.nan, dtype=np.float32))
        self._status.setText("Waiting for 3D keypoints…")
        self._status.setStyleSheet("color:#777; font-size:10px;")
