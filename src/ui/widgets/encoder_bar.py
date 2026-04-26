"""
EncoderBar — vertical progress-bar widget showing a single linear-encoder
position in millimetres.

Typical placement: two instances flanking the video tile, one per DAQ
encoder channel. Fixed display range (configurable; default 0–2000 mm);
values outside the range are clipped for drawing but shown unclipped in
the label.

When the underlying hardware is marked unavailable via
``config.ENCODER1_AVAILABLE`` / ``config.ENCODER2_AVAILABLE``, create the
widget with ``available=False`` and it draws a "비활성" placeholder so
the layout stays consistent.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen, QFont
from PyQt6.QtWidgets import QWidget


class EncoderBar(QWidget):
    def __init__(self, label: str = "enc", max_mm: float = 2000.0,
                 available: bool = True, parent=None):
        super().__init__(parent)
        self._label   = str(label)
        self._max_mm  = float(max_mm)
        self._value_mm: Optional[float] = 0.0
        self._available = bool(available)
        # Min height bumped 180 → 200 to accommodate header (20) +
        # bar (>=120) + footer (36) + breathing room. Width 56 keeps
        # the value text "1234 mm" centered without horizontal clip.
        self.setMinimumSize(QSize(56, 200))
        self.setStyleSheet("background:transparent;")

    # ── public API ─────────────────────────────────────────────────────────
    def set_value(self, mm: Optional[float]) -> None:
        self._value_mm = None if mm is None else float(mm)
        self.update()

    def set_available(self, ok: bool) -> None:
        self._available = bool(ok)
        # Clear stale readings so the placeholder doesn't flash the last
        # live value before repainting.
        if not self._available:
            self._value_mm = None
        self.update()

    def is_available(self) -> bool:
        return bool(self._available)

    # Qt-style casing for consumer callsites
    isAvailable = is_available

    def set_max(self, max_mm: float) -> None:
        self._max_mm = max(1.0, float(max_mm))
        self.update()

    def set_label(self, text: str) -> None:
        self._label = str(text)
        self.update()

    # ── paint ──────────────────────────────────────────────────────────────
    def sizeHint(self) -> QSize:
        return QSize(54, 240)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width(); h = self.height()

        # Background panel
        p.fillRect(self.rect(), QColor("#161616"))
        p.setPen(QPen(QColor("#333"), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        # Header label area (top ~18 px)
        header_h = 20
        p.setPen(QColor("#90caf9"))
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(0, 0, w, header_h,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                    self._label)

        # Footer label area (bottom ~36 px) — value or "비활성".
        # U2-3: bumped from 28 px because at 100%+ DPI scaling, 10pt
        # bold value text was getting clipped at the top edge.
        footer_h = 36
        bar_top = header_h + 2
        bar_bottom = h - footer_h - 4
        bar_height = max(10, bar_bottom - bar_top)

        # Unavailable path
        if not self._available:
            # Grey diagonal hatching + 비활성 label
            p.setPen(QPen(QColor("#555"), 1, Qt.PenStyle.DashLine))
            p.drawRect(4, bar_top, w - 8, bar_height)
            p.setPen(QColor("#888"))
            f2 = QFont(); f2.setPointSize(8); p.setFont(f2)
            p.drawText(0, bar_top, w, bar_height,
                        Qt.AlignmentFlag.AlignCenter, "비활성")
            # Footer
            p.setPen(QColor("#666"))
            f3 = QFont(); f3.setPointSize(8); p.setFont(f3)
            p.drawText(0, h - footer_h, w, footer_h,
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                        "— mm")
            return

        # Normal: scale + fill
        # Scale ticks (every 500 mm if max≤2500, else every 1000)
        step = 500 if self._max_mm <= 2500 else 1000
        p.setPen(QPen(QColor("#2a2a2a"), 1))
        for mm in range(0, int(self._max_mm) + 1, step):
            y = bar_bottom - int((mm / self._max_mm) * bar_height)
            p.drawLine(4, y, w - 4, y)
            p.setPen(QColor("#555"))
            f_tick = QFont(); f_tick.setPointSize(7); p.setFont(f_tick)
            p.drawText(2, y - 7, w - 4, 10,
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        f"{mm}")
            p.setPen(QPen(QColor("#2a2a2a"), 1))

        # Filled value area
        if self._value_mm is not None:
            v = max(0.0, min(self._max_mm, float(self._value_mm)))
            fill_h = int((v / self._max_mm) * bar_height)
            # Gradient: green (low) → yellow (high)
            grad = QLinearGradient(0, bar_bottom, 0, bar_top)
            grad.setColorAt(0.0, QColor("#4CAF50"))
            grad.setColorAt(0.6, QColor("#FFC107"))
            grad.setColorAt(1.0, QColor("#FF7043"))
            p.fillRect(6, bar_bottom - fill_h, w - 12, fill_h, grad)
            # Current-position marker line
            p.setPen(QPen(QColor("#FFEB3B"), 2))
            y_marker = bar_bottom - fill_h
            p.drawLine(4, y_marker, w - 4, y_marker)

            # Footer: numeric value
            p.setPen(QColor("#fff"))
            f4 = QFont(); f4.setPointSize(10); f4.setBold(True); p.setFont(f4)
            p.drawText(0, h - footer_h, w, footer_h,
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                        f"{self._value_mm:.0f} mm")
        else:
            p.setPen(QColor("#666"))
            f5 = QFont(); f5.setPointSize(9); p.setFont(f5)
            p.drawText(0, h - footer_h, w, footer_h,
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                        "— mm")
