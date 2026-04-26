"""
StabilityOverlay — phase-aware status panel for the Measure tab.

Adapts its layout to the current RecorderState:
  - wait       → detailed stability readout + progress bar
  - countdown  → big countdown number
  - recording  → big REC indicator + elapsed/total + prompt + stim banner
  - done       → green check
  - cancelled  → red X
  - idle       → prompt message
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame,
)

from src.capture.session_recorder import RecorderState


G = 9.80665


class StabilityOverlay(QWidget):
    """Single panel that re-skins itself per recorder phase."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # T2: bump minimum height so the banner has room to wrap or
        # show ellipsis when the parent gets narrow. The previous
        # 100 px had to share with two detail rows + progress bar.
        self.setMinimumHeight(72)
        self.setStyleSheet(
            "QWidget { background:#1e1e1e; border:1px solid #333; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(3)

        # Banner (big status text). Word-wrap enabled (T2) so a long
        # message ("DAQ 영점 보정 중 — 플레이트에서 내려와 계세요 (4.5 s)")
        # gracefully takes 2 lines instead of being clipped on narrow
        # screens. Tooltip carries the full text on hover.
        self._banner = QLabel("대기 중 — 측정 대상 선택 후 시작 버튼을 누르세요.")
        self._banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._banner.setWordWrap(True)
        bf = QFont(); bf.setPointSize(15); bf.setBold(True)
        self._banner.setFont(bf)
        root.addWidget(self._banner)

        # Detail row (varies by phase). Single-line; if the parent is
        # narrow the QLabel auto-elides via stylesheet text-overflow,
        # but we also keep tooltips synced with the full text via
        # ``_set_label_with_tooltip`` below.
        self._detail_row = QHBoxLayout()
        self._detail_row.setSpacing(16)
        self._detail_total = QLabel("")
        self._detail_boards = QLabel("")
        self._detail_std   = QLabel("")
        for lbl in (self._detail_total, self._detail_boards, self._detail_std):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#bbb; font-size:11px;")
            lbl.setWordWrap(False)
            # Min width 0 lets the label compress; tooltip restores info.
            lbl.setMinimumWidth(0)
            self._detail_row.addWidget(lbl, stretch=1)
        root.addLayout(self._detail_row)

        # Progress bar (for stability hold / recording duration)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(18)
        root.addWidget(self._progress)

    # ── helpers ────────────────────────────────────────────────────────────
    def _set_text_with_tip(self, label: QLabel, text: str) -> None:
        """Set text + matching tooltip so the user can hover for the
        full message even when the label is too narrow to display it."""
        label.setText(text)
        label.setToolTip(text or "")

    # ── public API ──────────────────────────────────────────────────────────
    def update_from_state(self, st: RecorderState, subject_kg: float = 0.0,
                          stance: str = "two") -> None:
        if st.phase == "wait":
            self._render_wait(st, subject_kg, stance)
        elif st.phase == "countdown":
            self._render_countdown(st)
        elif st.phase == "recording":
            self._render_recording(st)
        elif st.phase == "done":
            self._render_done()
        elif st.phase == "cancelled":
            self._render_cancelled()
        else:   # idle
            self._render_idle()

    def reset(self) -> None:
        self._render_idle()

    def render_transition(self, prev_test_ko: str, next_test_ko: str,
                          remaining_s: float) -> None:
        """Between-test pause screen. Shown instead of the idle message so
        the subject + trainer know what just happened and what's next."""
        self._banner.setText(
            f"✓ 조금 전 측정 저장됨: {prev_test_ko}"
        )
        self._banner.setStyleSheet("color:#A5D6A7;")
        self._detail_total.setText(f"다음: {next_test_ko}")
        self._detail_total.setStyleSheet("color:#FFD54F; font-size:13px;")
        self._detail_boards.setText(
            f"{remaining_s:.1f} s 후 시작 — ⚠ 플레이트에서 내려와 계세요"
        )
        self._detail_boards.setStyleSheet("color:#FFA726; font-size:13px;")
        self._detail_std.setText("")
        frac = max(0.0, min(1.0, 1.0 - remaining_s / 8.0))
        self._progress.setValue(int(frac * 1000))
        self._progress.setFormat(f"다음 테스트까지 {remaining_s:.1f} s")
        self.setStyleSheet(
            "QWidget { background:#1a1300; border:1px solid #FF9800; }")

    # ── phase renderers ────────────────────────────────────────────────────
    def _render_idle(self) -> None:
        self._banner.setText("대기 중 — 측정 대상 선택 후 시작 버튼을 누르세요.")
        self._banner.setStyleSheet("color:#ddd;")
        self._detail_total.setText("")
        self._detail_boards.setText("")
        self._detail_std.setText("")
        self._progress.setValue(0)
        self._progress.setFormat("")
        self.setStyleSheet(
            "QWidget { background:#1e1e1e; border:1px solid #333; }")

    def _render_wait(self, st: RecorderState, subject_kg: float, stance: str) -> None:
        # During the explicit zero-cal window, show a dedicated countdown.
        # Once zeroing is False the banner stays on stability feedback even
        # if wait is briefly None between DAQ frames — no more flicker.
        if st.zeroing:
            rem = st.zero_cal_remaining_s
            self._banner.setText(
                f"DAQ 영점 보정 중 — 플레이트에서 내려와 계세요  ({rem:.1f} s)"
            )
            self._banner.setStyleSheet("color:#FFD54F;")
            self._detail_total.setText("플레이트 외부 대기")
            self._detail_boards.setText("")
            self._detail_std.setText("")
            frac = 1.0 - min(1.0, rem / 5.5)
            self._progress.setValue(int(frac * 1000))
            self._progress.setFormat(f"영점 보정 {frac*100:.0f}%")
            self.setStyleSheet(
                "QWidget { background:#261f00; border:1px solid #8d6e63; }")
            return
        wait = st.wait
        if wait is None:
            # Past zero-cal but DAQ frame not yet observed — keep a neutral
            # holding message rather than flashing back to the zero-cal banner.
            self._banner.setText("DAQ 신호 대기 중 ...")
            self._banner.setStyleSheet("color:#FFA726;")
            self._detail_total.setText("")
            self._detail_boards.setText("")
            self._detail_std.setText("")
            self._progress.setValue(0)
            self._progress.setFormat("")
            self.setStyleSheet(
                "QWidget { background:#1e1e1e; border:1px solid #444; }")
            return

        # Active stability feedback
        stance_hint = {
            "two":   "양발을 보드 위에",
            "left":  "좌측 발만 (Board1)",
            "right": "우측 발만 (Board2)",
        }.get(stance, "양발")

        status_map = {
            "STEP_ON":     (f"플레이트에 올라가세요 — {stance_hint}", "#EF5350", "#3a0000"),
            "STABILIZING": ("안정화 중 — 움직이지 마세요",         "#FFA726", "#2c1d00"),
            "READY":       ("READY — 시작합니다!",                  "#66BB6A", "#0d3311"),
            "TIMEOUT":     ("시간 초과",                            "#EF5350", "#3a0000"),
        }
        text, text_col, bg_col = status_map.get(
            wait.status, ("...", "#ddd", "#1e1e1e"))
        self._banner.setText(text)
        self._banner.setStyleSheet(f"color:{text_col};")

        # Detail lines — vary with stance
        if stance == "two":
            target = 0.70 * subject_kg * G if subject_kg > 0 else 0
            self._detail_total.setText(
                f"Total {wait.total_n:.0f} N  (목표 ≥ {target:.0f} N)")
            self._detail_boards.setText(
                f"B1 {wait.b1_n:.0f} N  ·  B2 {wait.b2_n:.0f} N")
        elif stance == "left":
            need = 0.60 * subject_kg * G if subject_kg > 0 else 0
            self._detail_total.setText(
                f"B1 (좌) {wait.b1_n:.0f} N  (목표 ≥ {need:.0f} N)")
            self._detail_boards.setText(
                f"B2 (우) {wait.b2_n:.0f} N  (< 100 N)")
        else:   # right
            need = 0.60 * subject_kg * G if subject_kg > 0 else 0
            self._detail_total.setText(
                f"B2 (우) {wait.b2_n:.0f} N  (목표 ≥ {need:.0f} N)")
            self._detail_boards.setText(
                f"B1 (좌) {wait.b1_n:.0f} N  (< 100 N)")
        self._detail_std.setText(
            f"안정성 σ = {wait.std_total_n:.1f} N  (< 50)")

        # Progress bar = stable-hold progress
        frac = min(wait.stable_progress_s /
                   max(wait.stable_target_s, 1e-6), 1.0)
        self._progress.setValue(int(frac * 1000))
        self._progress.setFormat(
            f"Stable {wait.stable_progress_s:.1f} / "
            f"{wait.stable_target_s:.1f} s  |  총 대기 {wait.wait_elapsed_s:.0f} s")
        self.setStyleSheet(
            f"QWidget {{ background:{bg_col}; border:1px solid #444; }}")

    def _render_countdown(self, st: RecorderState) -> None:
        rem = max(0.0, st.total_s - st.elapsed_s)  # reusing total_s as countdown target not ideal
        # For countdown, elapsed_s is phase-elapsed. We don't know countdown_s
        # from state — but callers render with big text either way.
        self._banner.setText(f"카운트다운 ... {st.elapsed_s:.1f} s")
        self._banner.setStyleSheet("color:#FFEB3B;")
        self._detail_total.setText("")
        self._detail_boards.setText("")
        self._detail_std.setText("")
        self._progress.setValue(0)
        self._progress.setFormat("")
        self.setStyleSheet(
            "QWidget { background:#1e1e1e; border:1px solid #FFEB3B; }")

    def _render_recording(self, st: RecorderState) -> None:
        banner_parts = [f"● REC  {st.elapsed_s:.1f} / {st.total_s:.0f} s"]
        if st.prompt:
            banner_parts.append(st.prompt)
        text = "   ".join(banner_parts)
        if st.stim_banner:
            text = f"{text}   ⚡ {st.stim_banner}"
        self._banner.setText(text)
        self._banner.setStyleSheet("color:#FF5252;")
        self._detail_total.setText("")
        self._detail_boards.setText(
            f"자극 수: {st.n_stim_fired}" if st.n_stim_fired > 0 else "")
        self._detail_std.setText("")
        frac = min(st.elapsed_s / max(st.total_s, 1e-6), 1.0)
        self._progress.setValue(int(frac * 1000))
        self._progress.setFormat(f"{st.elapsed_s:.1f} / {st.total_s:.0f} s")
        self.setStyleSheet(
            "QWidget { background:#3a0000; border:1px solid #B71C1C; }")

    def _render_done(self) -> None:
        self._banner.setText("✓ 측정 완료")
        self._banner.setStyleSheet("color:#66BB6A;")
        self._detail_total.setText("")
        self._detail_boards.setText("")
        self._detail_std.setText("")
        self._progress.setValue(1000)
        self._progress.setFormat("완료")
        self.setStyleSheet(
            "QWidget { background:#0d3311; border:1px solid #2E7D32; }")

    def _render_cancelled(self) -> None:
        self._banner.setText("✗ 측정 취소됨")
        self._banner.setStyleSheet("color:#EF5350;")
        self._detail_total.setText("")
        self._detail_boards.setText("")
        self._detail_std.setText("")
        self._progress.setValue(0)
        self._progress.setFormat("취소됨")
        self.setStyleSheet(
            "QWidget { background:#1e1e1e; border:1px solid #EF5350; }")
