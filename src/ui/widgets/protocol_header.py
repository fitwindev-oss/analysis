"""
ProtocolHeader — prominent progress banner for protocol (checklist) mode.

Shows at the top of the live column whenever a protocol queue is active:

    ┌───────────────────────────────────────────────────────────────────┐
    │ 프로토콜 진행  [2/3]                          다음: 좌측 발 밸런스 5s │
    │ ██████████████████████░░░░░░░░░░░░                                 │
    │ 현재: 🔴 CMJ · 10 s                   📢 양발을 양쪽 보드에 올리세요    │
    └───────────────────────────────────────────────────────────────────┘

Hidden in single-test mode or when queue is empty.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
)


# Human-friendly names + stance instruction
TEST_KO = {
    "balance_eo":     "눈 뜨고 밸런스",
    "balance_ec":     "눈 감고 밸런스",
    "cmj":            "CMJ 점프",
    "squat":          "스쿼트",
    "overhead_squat": "오버헤드 스쿼트",
    "encoder":        "엔코더 (바)",
    "reaction":       "반응 시간",
    "proprio":        "고유감각",
}


def _describe(opts: dict) -> str:
    """Short one-line descriptor for a queue item."""
    test = opts.get("test", "?")
    name = TEST_KO.get(test, test)
    dur = opts.get("duration_s", 0.0)
    extras = []
    if test in ("balance_eo", "balance_ec"):
        stance = opts.get("stance", "two")
        extras.append({"two": "양발", "left": "좌측발", "right": "우측발"}.get(stance, stance))
    if test == "reaction":
        extras.append(f"{opts.get('n_stimuli', 0)}회")
        extras.append("수동" if opts.get("trigger") == "manual" else "자동")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{name}{suffix}  ·  {dur:.0f} s"


def _stance_instruction(opts: dict) -> str:
    """Big actionable instruction telling subject what to do."""
    test = opts.get("test", "?")
    if test in ("balance_eo", "balance_ec"):
        stance = opts.get("stance", "two")
        if stance == "left":
            return "⬅  좌측 발만 Board1 (왼쪽 보드) 위에"
        if stance == "right":
            return "➡  우측 발만 Board2 (오른쪽 보드) 위에"
        return "⬇  양발을 양쪽 보드 위에"
    if test == "cmj":
        return "⬇  양발 기립 → 신호 후 최대한 높이 점프"
    if test == "squat":
        return "⬇  양발 기립 → 준비 되면 스쿼트 반복"
    if test == "overhead_squat":
        return "⬇  바/봉을 머리 위로 들고 스쿼트"
    if test == "encoder":
        prompt = opts.get("encoder_prompt") or ""
        return f"🏋  {prompt}" if prompt else "🏋  프롬프트에 따른 동작 수행"
    if test == "reaction":
        return "⚡  신호를 보고 즉시 반응"
    if test == "proprio":
        return "🎯  목표 자세 확인 후 재현"
    return "—"


class ProtocolHeader(QWidget):
    """Header strip shown at top of live column during a protocol run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(92)
        self.setStyleSheet(
            "QWidget { background:#102a16; border:2px solid #2E7D32; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(4)

        # Row 1: title + next
        top = QHBoxLayout()
        self._title = QLabel("프로토콜 진행  [ — / — ]")
        tf = QFont(); tf.setPointSize(13); tf.setBold(True)
        self._title.setFont(tf)
        self._title.setStyleSheet("color:#A5D6A7;")
        top.addWidget(self._title)
        top.addStretch(1)
        self._next = QLabel("")
        self._next.setStyleSheet("color:#bbb; font-size:12px;")
        top.addWidget(self._next)
        root.addLayout(top)

        # Row 2: progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(12)
        self._progress.setStyleSheet(
            "QProgressBar { background:#1a1a1a; border:1px solid #333; } "
            "QProgressBar::chunk { background:#4CAF50; }")
        root.addWidget(self._progress)

        # Row 3: current test + stance instruction
        bot = QHBoxLayout()
        self._current = QLabel("현재: —")
        cf = QFont(); cf.setPointSize(12); cf.setBold(True)
        self._current.setFont(cf)
        self._current.setStyleSheet("color:#fff;")
        bot.addWidget(self._current)
        bot.addStretch(1)
        self._instruction = QLabel("")
        inf = QFont(); inf.setPointSize(14); inf.setBold(True)
        self._instruction.setFont(inf)
        self._instruction.setStyleSheet("color:#FFEB3B;")
        bot.addWidget(self._instruction)
        root.addLayout(bot)

        self.hide()

    # ── public API ─────────────────────────────────────────────────────────
    def show_queue(self, queue: list[dict], idx: int,
                   transitioning: bool = False,
                   transition_remaining_s: float = 0.0) -> None:
        """Update header to reflect current queue state.

        transitioning=True during the inter-test pause — shows a big
        countdown + "step off plate" message.
        """
        n = len(queue)
        if n == 0:
            self.hide()
            return
        shown_idx = min(idx, n - 1)
        cur = queue[shown_idx]
        nxt = queue[idx + 1] if (idx + 1 < n) else None

        self._title.setText(f"프로토콜 진행  [{idx + 1} / {n}]")
        self._progress.setValue(int(1000 * (idx) / n))

        if transitioning:
            # Next test is about to start — encourage subject to step off
            self._current.setText("⚠  다음 테스트 준비 중 — 플레이트에서 내려오세요")
            self._current.setStyleSheet("color:#FFA726;")
            self._instruction.setText(f"{transition_remaining_s:.0f} s 후 시작")
            self._instruction.setStyleSheet("color:#FFEB3B;")
            if nxt is not None:
                self._next.setText(f"다음: {_describe(nxt)}")
            else:
                self._next.setText("")
            self.setStyleSheet(
                "QWidget { background:#2a1a00; border:2px solid #FF9800; }")
        else:
            self._current.setText(f"현재: {_describe(cur)}")
            self._current.setStyleSheet("color:#fff;")
            self._instruction.setText(_stance_instruction(cur))
            self._instruction.setStyleSheet("color:#FFEB3B;")
            if nxt is not None:
                self._next.setText(f"다음: {_describe(nxt)}")
            else:
                self._next.setText("(마지막)")
            self.setStyleSheet(
                "QWidget { background:#102a16; border:2px solid #2E7D32; }")

        self.show()

    def hide_header(self) -> None:
        self.hide()
