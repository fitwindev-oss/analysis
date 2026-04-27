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
# strength_3lift is a multi-set, operator-driven test — duration_s is
# unused (the recorder ignores it for that test) so we set it to 0
# and hide the duration row from the UI.
TESTS_KO: list[tuple[str, str, float]] = [
    ("balance_eo",     "밸런스 (눈 뜨고)",     30.0),
    ("balance_ec",     "밸런스 (눈 감고)",     30.0),
    ("cmj",            "CMJ (반동 점프)",      10.0),
    ("sj",             "SJ (스쿼트 점프, 반동 없음)", 10.0),
    ("squat",          "스쿼트",               30.0),
    ("overhead_squat", "오버헤드 스쿼트",      30.0),
    ("reaction",            "반응 시간",                       60.0),
    # V6 — visual + cognitive reaction with positional cues + skeleton
    # tracking. The screen flashes a target spot (N/E/S/W or N/NE/.../NW);
    # subject must reach it with the configured body part. Pose pipeline
    # extracts RT (stim → motion onset), MT (onset → arrival) and the
    # spatial accuracy of the reach.
    ("cognitive_reaction",  "시각/인지 반응 (위치 큐 + 스켈레톤)", 60.0),
    ("proprio",             "고유감각",                        60.0),
    ("free_exercise",  "자유 운동 측정",       60.0),
    ("strength_3lift", "전신 근력 (3대 운동)", 0.0),
]
DURATION_BY_TEST: dict[str, float] = {k: d for k, _, d in TESTS_KO}

# Per-exercise pretty labels for the strength_3lift dropdown.
STRENGTH_EXERCISE_LABELS: list[tuple[str, str]] = [
    ("bench_press", "벤치프레스 (Bench Press) — 상체"),
    ("back_squat",  "백스쿼트 (Back Squat) — 하체"),
    ("deadlift",    "데드리프트 (Deadlift) — 전신"),
]


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
        if test == "cognitive_reaction":
            # V6 — reuses n_stimuli / stim_min_gap / stim_max_gap from the
            # cognitive group so timing semantics are identical to ``reaction``.
            opts["n_stimuli"]    = self._cog_n_stim.value()
            opts["stim_min_gap"] = self._cog_stim_min.value()
            opts["stim_max_gap"] = self._cog_stim_max.value()
            opts["trigger"]      = "manual" if self._cog_trigger_manual.isChecked() else "auto"
            opts["react_track_body_part"] = (
                self._cog_body_part.currentData() or "right_hand")
            opts["react_n_positions"] = (
                8 if self._cog_pos_8.isChecked() else 4)
        if test == "free_exercise":
            opts["exercise_name"]       = self._free_name.text().strip() or None
            opts["load_kg"]             = float(self._free_load.value())
            opts["use_bodyweight_load"] = self._free_use_bw.isChecked()
        if test == "strength_3lift":
            opts["exercise"]    = self._strength_exercise.currentData()
            opts["n_sets"]      = int(self._strength_n_sets.value())
            opts["target_reps"] = int(self._strength_reps.value())
            opts["load_kg"]     = float(self._strength_load.value())
            opts["rest_s"]      = float(self._strength_rest.value())
            opts["warmup_set"]  = self._strength_warmup.isChecked()
            opts["use_bodyweight_load"] = self._strength_use_bw.isChecked()
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

        # ── Cognitive reaction (Phase V6-UI) ────────────────────────────────
        # Visual+cognitive RT with on-screen positional cue + skeleton-based
        # reach tracking. The "응답 유형" concept doesn't apply (every cue
        # IS its own response — a position to reach), so the form is
        # narrower: body part, n_positions, stim count + gaps + trigger.
        self._cognitive_box = QGroupBox("시각/인지 반응 옵션 (V6)")
        cl = QFormLayout(self._cognitive_box)

        self._cog_body_part = QComboBox()
        self._cog_body_part.addItem("오른손 (Right Hand)", "right_hand")
        self._cog_body_part.addItem("왼손 (Left Hand)",    "left_hand")
        self._cog_body_part.addItem("오른발 (Right Foot)", "right_foot")
        self._cog_body_part.addItem("왼발 (Left Foot)",    "left_foot")
        self._cog_body_part.setToolTip(
            "위치 큐를 향해 이동시킬 신체 부위. MediaPipe BlazePose의 33개 "
            "키포인트 중 손목/검지(손) 또는 발끝/발목(발)을 추적합니다.")
        cl.addRow("추적 부위", self._cog_body_part)

        pos_row = QHBoxLayout()
        self._cog_pos_4 = QRadioButton("4 방향 (N/E/S/W)")
        self._cog_pos_8 = QRadioButton("8 방향 (N/NE/E/SE/S/SW/W/NW)")
        self._cog_pos_4.setChecked(True)
        self._cog_pos_group = QButtonGroup(self)
        for b in (self._cog_pos_4, self._cog_pos_8):
            self._cog_pos_group.addButton(b)
            pos_row.addWidget(b)
        pos_row.addStretch(1)
        cl.addRow("위치 수", self._wrap_layout(pos_row))

        self._cog_n_stim = QSpinBox()
        self._cog_n_stim.setRange(1, 100); self._cog_n_stim.setValue(10)
        cl.addRow("자극 횟수", self._cog_n_stim)
        self._cog_stim_min = QDoubleSpinBox()
        self._cog_stim_min.setRange(0.5, 30.0); self._cog_stim_min.setValue(2.5)
        self._cog_stim_min.setSuffix(" s")
        cl.addRow("자극 간 최소 간격", self._cog_stim_min)
        self._cog_stim_max = QDoubleSpinBox()
        self._cog_stim_max.setRange(0.5, 30.0); self._cog_stim_max.setValue(5.0)
        self._cog_stim_max.setSuffix(" s")
        cl.addRow("자극 간 최대 간격", self._cog_stim_max)

        cog_trig_row = QHBoxLayout()
        self._cog_trigger_auto   = QRadioButton("자동 (사전 스케줄)")
        self._cog_trigger_manual = QRadioButton("수동 (운영자 키 입력)")
        self._cog_trigger_auto.setChecked(True)
        self._cog_trigger_group = QButtonGroup(self)
        for b in (self._cog_trigger_auto, self._cog_trigger_manual):
            self._cog_trigger_group.addButton(b)
            cog_trig_row.addWidget(b)
        cog_trig_row.addStretch(1)
        cl.addRow("트리거", self._wrap_layout(cog_trig_row))

        root.addWidget(self._cognitive_box)

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

        # ── Strength 3-lift (Phase V1-E) ────────────────────────────────────
        # Multi-set protocol: bench_press / back_squat / deadlift × N sets
        # of target_reps reps with `rest_s` inter-set rest. Set ends are
        # operator-driven via the "세트 종료" button in MeasureTab; rest
        # auto-transitions. See SessionRecorder.end_set / pause_rest etc.
        self._strength_box = QGroupBox("3대 운동 옵션")
        sl = QFormLayout(self._strength_box)

        self._strength_exercise = QComboBox()
        for key, ko in STRENGTH_EXERCISE_LABELS:
            self._strength_exercise.addItem(ko, key)
        sl.addRow("운동", self._strength_exercise)

        self._strength_n_sets = QSpinBox()
        self._strength_n_sets.setRange(3, 5)
        self._strength_n_sets.setValue(3)
        self._strength_n_sets.setToolTip(
            "총 측정 세트 수 (3-5). 1세트 워밍업 + 2-4 본세트가 일반적입니다.")
        sl.addRow("세트 수", self._strength_n_sets)

        self._strength_reps = QSpinBox()
        self._strength_reps.setRange(1, 30)
        self._strength_reps.setValue(12)
        self._strength_reps.setToolTip(
            "목표 반복 횟수 (안내용). 10-12회를 권장합니다 — 1RM 추정 신뢰도가 "
            "가장 높은 구간입니다. 실패 직전까지 반복하세요.")
        sl.addRow("목표 반복 횟수", self._strength_reps)

        self._strength_load = QDoubleSpinBox()
        self._strength_load.setRange(0.0, 500.0)
        self._strength_load.setDecimals(1)
        self._strength_load.setSingleStep(2.5)
        self._strength_load.setValue(40.0)
        self._strength_load.setSuffix(" kg")
        self._strength_load.setToolTip(
            "바벨 외부 하중 (자체 중량). 모든 세트는 같은 하중으로 진행합니다.")
        sl.addRow("작업 하중", self._strength_load)

        self._strength_rest = QDoubleSpinBox()
        self._strength_rest.setRange(1.0, 600.0)
        self._strength_rest.setDecimals(1)
        self._strength_rest.setSingleStep(5.0)
        self._strength_rest.setValue(30.0)
        self._strength_rest.setSuffix(" s")
        self._strength_rest.setToolTip(
            "세트 사이 휴식 시간. 30초가 ATP-PCr 회복 반감기 기준 표준값입니다.")
        sl.addRow("세트간 휴식", self._strength_rest)

        self._strength_warmup = QCheckBox("1세트는 웜업 (1RM 추정에서 제외)")
        self._strength_warmup.setChecked(True)
        self._strength_warmup.setToolTip(
            "체크하면 첫 번째 세트는 웜업 세트로 간주되어 1RM 추정 계산에서 "
            "제외됩니다. 기록은 그대로 보존됩니다.")
        sl.addRow("", self._strength_warmup)

        # V1.5 — bodyweight contribution flag for trainees who can't
        # add external load (women, beginners, elderly with bodyweight
        # squats; bench presses don't lift the body so the factor is 0).
        self._strength_use_bw = QCheckBox(
            "자체중 가산 (빈 봉 / 자체중 운동 시)")
        self._strength_use_bw.setChecked(False)
        self._strength_use_bw.setToolTip(
            "체크하면 1RM 산출 시 작업하중 = 외부하중 + α × 체중으로 보정됩니다.\n"
            "  • 벤치프레스: α = 0.0 (체중이 봉에 안 실림)\n"
            "  • 백스쿼트:   α = 0.85 (하퇴/발 외 신체가 같이 들림)\n"
            "  • 데드리프트: α = 0.10 (신체 COM 소폭 상승)\n"
            "외부 하중 없이 자체중만 가능한 피험자(여성/고령자)에게 의미 있는 "
            "1RM 추정값을 만들기 위한 옵션입니다.")
        sl.addRow("", self._strength_use_bw)

        root.addWidget(self._strength_box)

        root.addStretch(1)

    def _wrap_layout(self, lay) -> QWidget:
        w = QWidget(); w.setLayout(lay); return w

    # ── dynamic visibility ─────────────────────────────────────────────────
    def _on_test_changed(self, _idx: int) -> None:
        test = self.current_test()
        self._duration.setValue(DURATION_BY_TEST.get(test, 30.0))
        is_balance = test in ("balance_eo", "balance_ec")
        is_strength = (test == "strength_3lift")
        self._balance_box.setVisible(is_balance)
        self._reaction_box.setVisible(test == "reaction")
        # V6 — cognitive reaction has its own option group
        self._cognitive_box.setVisible(test == "cognitive_reaction")
        # Encoder-usage checkbox is not applicable to balance tests.
        # cognitive_reaction also doesn't involve a bar/rod.
        self._uses_enc.setVisible(not is_balance and test != "cognitive_reaction")
        self._free_box.setVisible(test == "free_exercise")
        self._strength_box.setVisible(is_strength)
        # strength_3lift is operator-driven (multi-set); the global
        # duration_s is unused. Hide the duration row to avoid confusion.
        self._duration.setEnabled(not is_strength)
        self._duration.setVisible(not is_strength)
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
