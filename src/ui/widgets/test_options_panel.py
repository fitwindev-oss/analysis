"""
Per-test options panel for the Measure tab.

Swaps its visible widgets based on the currently selected test. Emits
options() -> dict compatible with RecorderConfig.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QComboBox,
    QDoubleSpinBox, QSpinBox, QLineEdit, QCheckBox, QLabel, QFrame,
    QGroupBox, QButtonGroup, QRadioButton,
)

import config


# Test keys -> (Korean label, default duration_s)
TESTS_KO: list[tuple[str, str, float]] = [
    ("balance_eo",     "밸런스 (눈 뜨고)",     30.0),
    ("balance_ec",     "밸런스 (눈 감고)",     30.0),
    ("cmj",            "CMJ (점프)",           10.0),
    ("squat",          "스쿼트",               30.0),
    ("overhead_squat", "오버헤드 스쿼트",      30.0),
    ("reaction",       "반응 시간",            60.0),
    ("proprio",        "고유감각",             60.0),
    ("free_exercise",  "자유 운동 측정",       60.0),
]
DURATION_BY_TEST: dict[str, float] = {k: d for k, _, d in TESTS_KO}


class TestOptionsPanel(QWidget):
    """Dynamic options form. Call options() to read the current values."""

    test_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._on_test_changed(self._combo.currentIndex())

    # ── public ─────────────────────────────────────────────────────────────
    def current_test(self) -> str:
        return self._combo.currentData()

    def options(self) -> dict:
        """Return a dict of kwargs suitable for RecorderConfig(**dict).

        Note: the `_auto_pose` key is NOT a RecorderConfig field; callers
        must pop it before constructing the config. It's carried here so the
        Measure tab can read it in the same place as the recording options.
        """
        test = self.current_test()
        is_balance = test in ("balance_eo", "balance_ec")
        opts: dict = {
            "test":           test,
            "duration_s":     self._duration.value(),
            "use_smart_wait": self._smart_wait.isChecked(),
            "wait_timeout_s": self._wait_timeout.value(),
            "countdown_s":    self._countdown.value(),
            # For balance tests the encoder flag is meaningless; force True
            # so the RecorderConfig default holds and meta serializes None.
            "uses_encoder":   True if is_balance else self._uses_enc.isChecked(),
            "_auto_pose":         self._auto_pose.isChecked(),
            "_post_complexity":   int(self._post_complexity.currentData() or 1),
            "_live_pose":         self._live_pose.isChecked(),
            "_live_complexity":   int(self._live_complexity.currentData() or 0),
            "_live_cam_id":       str(self._live_cam_combo.currentData()
                                      or config.POSE_REALTIME_CAM_ID),
        }
        if test in ("balance_eo", "balance_ec"):
            opts["stance"] = self._stance_key()
        if test == "reaction":
            opts["n_stimuli"]    = self._n_stim.value()
            opts["stim_min_gap"] = self._stim_min.value()
            opts["stim_max_gap"] = self._stim_max.value()
            opts["trigger"]      = "manual" if self._trigger_manual.isChecked() else "auto"
            opts["responses"]    = self._responses_str()
        if test == "free_exercise":
            opts["exercise_name"]       = self._free_name.text().strip() or None
            opts["load_kg"]             = float(self._free_load.value())
            opts["use_bodyweight_load"] = self._free_use_bw.isChecked()
        return opts

    # ── UI build ───────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Test selector
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("테스트 항목"))
        self._combo = QComboBox()
        for key, ko, _ in TESTS_KO:
            self._combo.addItem(ko, key)
        self._combo.currentIndexChanged.connect(self._on_test_changed)
        top.addWidget(self._combo, stretch=1)
        root.addLayout(top)

        # Common options (always visible)
        common = QFormLayout()
        common.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._duration = QDoubleSpinBox()
        self._duration.setRange(1.0, 600.0)
        self._duration.setSuffix(" s")
        self._duration.setDecimals(1)
        common.addRow("측정 시간", self._duration)

        self._smart_wait = QCheckBox("스마트 대기 (안정된 자세 감지 후 자동 시작)")
        self._smart_wait.setChecked(True)
        self._smart_wait.toggled.connect(self._on_smart_toggled)
        common.addRow("", self._smart_wait)

        self._wait_timeout = QDoubleSpinBox()
        self._wait_timeout.setRange(10.0, 300.0)
        self._wait_timeout.setValue(60.0)
        self._wait_timeout.setSuffix(" s")
        common.addRow("대기 타임아웃", self._wait_timeout)

        self._countdown = QDoubleSpinBox()
        self._countdown.setRange(1.0, 30.0)
        self._countdown.setValue(5.0)
        self._countdown.setSuffix(" s")
        self._countdown.setEnabled(False)   # only used when smart-wait off
        common.addRow("고정 카운트다운", self._countdown)

        # Encoder usage — whether this session attaches the linear encoder
        # (bar/rod). Hidden for balance tests (no bar involved).
        self._uses_enc = QCheckBox("엔코더 사용 (바/봉 연결된 경우 체크)")
        self._uses_enc.setChecked(True)
        self._uses_enc.setToolTip(
            "바벨·덤벨·봉에 엔코더를 부착해 측정할 때 체크하세요. "
            "체크 해제 시 리플레이/리포트에서 엔코더 시계열이 '비활성'으로 "
            "표시됩니다. 밸런스 테스트에서는 자동으로 숨겨집니다.")
        common.addRow("", self._uses_enc)

        self._auto_pose = QCheckBox("📷 녹화 후 2D 포즈 자동 처리 (MediaPipe)")
        self._auto_pose.setChecked(False)
        self._auto_pose.setToolTip(
            "녹화된 mp4에서 MediaPipe BlazePose로 33개 관절을 추정하고, "
            "분석 리포트에 12개 관절 각도(무릎·고관절·발목·어깨·팔꿈치×L/R + "
            "몸통/목 기울기)를 포함시킵니다. CPU 사용, 세션당 10~30초 추가.\n"
            "실패해도 Force 분석 결과는 보존됩니다."
        )
        common.addRow("", self._auto_pose)

        # Post-record complexity (Lite / Full / Heavy)
        self._post_complexity = QComboBox()
        self._post_complexity.addItem("Lite (0) — 가장 빠름",   0)
        self._post_complexity.addItem("Full (1) — 기본 (권장)", 1)
        self._post_complexity.addItem("Heavy (2) — 가장 정확",  2)
        self._post_complexity.setCurrentIndex(
            {0: 0, 1: 1, 2: 2}.get(config.POSE_POSTRECORD_COMPLEXITY, 1))
        self._post_complexity.setToolTip(
            "녹화 후 포즈 처리에 사용할 MediaPipe 모델 복잡도. "
            "Heavy는 CPU 사용 시 분 단위로 오래 걸릴 수 있습니다.")
        common.addRow("  └ 모델 (post-record)", self._post_complexity)

        # Realtime overlay options
        self._live_pose = QCheckBox(
            "🔴 녹화 중 2D 포즈 실시간 오버레이 (1 카메라, CPU)")
        self._live_pose.setChecked(bool(config.POSE_REALTIME_ENABLED))
        self._live_pose.setToolTip(
            "녹화 중 선택한 1 카메라에 BlazePose 결과를 skeleton으로 오버레이합니다. "
            "CPU 사용량이 높으면 Lite 모델을 권장합니다.")
        common.addRow("", self._live_pose)

        live_row = QHBoxLayout()
        self._live_complexity = QComboBox()
        self._live_complexity.addItem("Lite (0)",  0)
        self._live_complexity.addItem("Full (1)",  1)
        self._live_complexity.addItem("Heavy (2)", 2)
        self._live_complexity.setCurrentIndex(
            {0: 0, 1: 1, 2: 2}.get(config.POSE_REALTIME_COMPLEXITY, 0))
        self._live_cam_combo = QComboBox()
        # Populated lazily from config.CAMERAS (post app-startup probe)
        for cam in config.CAMERAS:
            self._live_cam_combo.addItem(
                f"{cam['id']}  {cam.get('label', '')}", cam["id"])
        # Pre-select default if available
        idx = self._live_cam_combo.findData(config.POSE_REALTIME_CAM_ID)
        if idx >= 0:
            self._live_cam_combo.setCurrentIndex(idx)
        live_row.addWidget(QLabel("모델"))
        live_row.addWidget(self._live_complexity, stretch=1)
        live_row.addWidget(QLabel("대상 카메라"))
        live_row.addWidget(self._live_cam_combo, stretch=1)
        common.addRow("  └ 실시간", self._wrap_layout(live_row))
        root.addLayout(common)

        # Divider
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); root.addWidget(sep)

        # ── Test-specific groups ────────────────────────────────────────────
        # Balance — stance selector
        self._balance_box = QGroupBox("밸런스 옵션")
        bl = QFormLayout(self._balance_box)
        self._stance_group = QButtonGroup(self)
        self._stance_two   = QRadioButton("양발")
        self._stance_left  = QRadioButton("좌측 (Board1)")
        self._stance_right = QRadioButton("우측 (Board2)")
        self._stance_two.setChecked(True)
        row = QHBoxLayout()
        for btn in (self._stance_two, self._stance_left, self._stance_right):
            self._stance_group.addButton(btn)
            row.addWidget(btn)
        row.addStretch(1)
        bl.addRow("스탠스", self._wrap_layout(row))
        root.addWidget(self._balance_box)

        # Reaction — stimulus schedule + trigger + responses
        self._reaction_box = QGroupBox("반응 시간 옵션")
        rl = QFormLayout(self._reaction_box)
        self._n_stim = QSpinBox()
        self._n_stim.setRange(1, 100); self._n_stim.setValue(10)
        rl.addRow("자극 횟수", self._n_stim)
        self._stim_min = QDoubleSpinBox()
        self._stim_min.setRange(0.5, 30.0); self._stim_min.setValue(2.0)
        self._stim_min.setSuffix(" s")
        rl.addRow("자극 간 최소 간격", self._stim_min)
        self._stim_max = QDoubleSpinBox()
        self._stim_max.setRange(0.5, 30.0); self._stim_max.setValue(5.0)
        self._stim_max.setSuffix(" s")
        rl.addRow("자극 간 최대 간격", self._stim_max)

        trig_row = QHBoxLayout()
        self._trigger_auto   = QRadioButton("자동 (사전 스케줄)")
        self._trigger_manual = QRadioButton("수동 (운영자 키 입력)")
        self._trigger_auto.setChecked(True)
        self._trigger_group = QButtonGroup(self)
        for b in (self._trigger_auto, self._trigger_manual):
            self._trigger_group.addButton(b)
            trig_row.addWidget(b)
        trig_row.addStretch(1)
        rl.addRow("트리거", self._wrap_layout(trig_row))

        resp_row = QHBoxLayout()
        self._resp_left  = QCheckBox("좌측 이동")
        self._resp_right = QCheckBox("우측 이동")
        self._resp_jump  = QCheckBox("점프")
        for b in (self._resp_left, self._resp_right, self._resp_jump):
            b.setChecked(True)
            resp_row.addWidget(b)
        resp_row.addStretch(1)
        rl.addRow("응답 유형", self._wrap_layout(resp_row))
        root.addWidget(self._reaction_box)

        # Free exercise — exercise name + external load + bodyweight override
        self._free_box = QGroupBox("자유 운동 옵션")
        fl = QFormLayout(self._free_box)
        self._free_name = QLineEdit()
        self._free_name.setPlaceholderText("예: 데드리프트, 벤치프레스, 푸쉬업")
        fl.addRow("운동 이름", self._free_name)
        self._free_load = QDoubleSpinBox()
        self._free_load.setRange(0.0, 500.0)
        self._free_load.setDecimals(1)
        self._free_load.setSingleStep(1.0)
        self._free_load.setSuffix(" kg")
        fl.addRow("외부 하중", self._free_load)
        self._free_use_bw = QCheckBox("자기 하중 추가 (외부 하중에 체중 가산)")
        self._free_use_bw.setToolTip(
            "체크하면 피험자의 체중(subject.weight_kg)이 위의 '외부 하중'에 "
            "더해져 최종 하중으로 사용됩니다. 예) 외부 20 kg + 체중 93 kg = "
            "113 kg. 외부 하중이 0이면 자중 단독(푸쉬업·풀업 등)으로 동작."
        )
        fl.addRow("", self._free_use_bw)
        root.addWidget(self._free_box)

        root.addStretch(1)

    def _wrap_layout(self, lay) -> QWidget:
        w = QWidget(); w.setLayout(lay); return w

    # ── dynamic visibility ─────────────────────────────────────────────────
    def _on_test_changed(self, _idx: int) -> None:
        test = self.current_test()
        self._duration.setValue(DURATION_BY_TEST.get(test, 30.0))
        is_balance = test in ("balance_eo", "balance_ec")
        self._balance_box.setVisible(is_balance)
        self._reaction_box.setVisible(test == "reaction")
        # Encoder-usage checkbox is not applicable to balance tests.
        self._uses_enc.setVisible(not is_balance)
        self._free_box.setVisible(test == "free_exercise")
        self.test_changed.emit(test)

    def _on_smart_toggled(self, on: bool) -> None:
        self._countdown.setEnabled(not on)
        self._wait_timeout.setEnabled(on)


    # ── helpers ────────────────────────────────────────────────────────────
    def _stance_key(self) -> str:
        if self._stance_left.isChecked():  return "left"
        if self._stance_right.isChecked(): return "right"
        return "two"

    def _responses_str(self) -> str:
        picked = []
        if self._resp_left.isChecked():  picked.append("left_shift")
        if self._resp_right.isChecked(): picked.append("right_shift")
        if self._resp_jump.isChecked():  picked.append("jump")
        if len(picked) == 3:
            return "random"
        return ",".join(picked) if picked else "random"
