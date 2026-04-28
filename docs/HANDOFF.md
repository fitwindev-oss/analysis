# 📘 biomech-mocap 인수인계 자료
**스냅샷 태그**: `v.01.2026.04.28_testend`
**테스트 현황**: 332 / 332 통과 (17 테스트 파일)
**대상 독자**: 본 코드베이스를 이어받아 개발할 차기 엔지니어

---

## 1. 전체 시스템 개요

### 1.1 시스템 구성

FITWIN 모션캡처 측정·분석 통합 데스크톱 앱. 다음 4가지 하드웨어를 동시 캡처하고 임상 리포트를 생성합니다:

| 하드웨어 | 데이터 | 주파수 | 위치 |
|---|---|---|---|
| 트윈 포스플레이트 (4 load-cell × 2) | 8 corner force (N) | 100 Hz | NI-DAQ |
| 선형 인코더 ×2 | 위치(mm) | 100 Hz | NI-DAQ (동일 task) |
| 카메라 ×N (Logitech StreamCam 등) | mp4 + 프레임별 wall-clock 타임스탬프 | 30~60 fps | OpenCV / DSHOW |
| MediaPipe BlazePose | 33 keypoint (x_px, y_px, vis) + 12 angle | 카메라 fps | post-record (mp4 → npz) + live overlay |

### 1.2 핵심 기술 스택

- **언어**: Python 3.11 (Windows)
- **GUI**: PyQt6 + pyqtgraph
- **캡처**: nidaqmx (DAQ) + opencv-python (camera)
- **포즈**: mediapipe (Tasks API, BlazePose Lite/Full/Heavy)
- **분석**: numpy + scipy.signal + pandas
- **리포트**: matplotlib (chart) + reportlab (PDF) + openpyxl (Excel)
- **DB**: SQLite (sessions, subjects, session_metrics 캐시)

### 1.3 디렉터리 구조

```
biomech-mocap/
├── config.py                  # 모든 상수 (DAQ rate, cameras, thresholds)
├── src/
│   ├── capture/               # 하드웨어 IO + recorder state machine
│   │   ├── session_recorder.py   ★ 핵심 state machine (1500+ LOC)
│   │   ├── camera_worker.py      # multiprocessing 카메라 캡처
│   │   ├── daq_reader.py         # NI-DAQ 8 force + 2 encoder 채널
│   │   ├── wait_for_stance.py    # 측정 시작 전 stance 안정 검출기
│   │   ├── cop_state.py          # on/off-plate 분류 (20N 임계)
│   │   └── departure_events.py   # off-plate 이벤트 streaming tracker
│   ├── analysis/              # 측정 후 분석 파이프라인
│   │   ├── dispatcher.py         # test_type → analyzer 라우팅
│   │   ├── common.py             # ForceSession, butter_lowpass 등 공통
│   │   ├── pose2d.py             # MP33 / 12 angle / window_summary
│   │   ├── balance.py            # 균형 (이동속도, 95% 타원, RMS)
│   │   ├── cmj.py                # CMJ jump height + power + RFD
│   │   ├── squat.py              # squat reps + CMC + L/R impulse + CoP safety (V5)
│   │   ├── encoder.py            # 인코더 기반 rep counter
│   │   ├── reaction.py           # 자극→FORCE 반응 시간
│   │   ├── proprio.py            # 고유감각 trial별 절대/일정/가변 오차
│   │   ├── free_exercise.py      # 외부 하중 자유 운동
│   │   ├── strength_norms.py     ★ V1: StrengthLevel-style 6 norm tables
│   │   ├── one_rm.py             ★ V1-C: Epley/Brzycki/Lombardi 앙상블
│   │   ├── strength_3lift.py     ★ V1-F: bench/squat/deadlift 다중 세트
│   │   ├── multiset_recovery.py  ★ V2: ATP-PCr 회복지수 (FI/PDS)
│   │   ├── composite_strength.py ★ V3: 다중세션 합성 등급
│   │   ├── ssc.py                ★ V4: CMJ vs SJ Stretch-Shortening
│   │   ├── cognitive_reaction.py ★ V6: 시각/인지 반응 + V6-G CRI
│   │   ├── csv_export.py         ★ Notepad-readable export (NEW)
│   │   └── excel_export.py       # 4시트 xlsx export
│   ├── reports/               # HTML / PDF 리포트 빌더
│   │   ├── report_builder.py     ★ test_type → 카드 + 섹션 라우팅
│   │   ├── charts.py             # 모든 matplotlib 차트 (1700+ LOC)
│   │   ├── palette.py            # FITWIN 색상 상수
│   │   ├── fonts.py              # 한글 폰트 자동 등록
│   │   ├── html_renderer.py      # QTextBrowser용 HTML
│   │   ├── pdf_renderer.py       # reportlab PDF
│   │   ├── key_metrics.py        # session_metrics 캐시 추출기
│   │   ├── norms.py              # 정상범위 lookup + 등급화
│   │   └── sections/             # 검사별 리포트 섹션
│   ├── pose/
│   │   └── mediapipe_backend.py  # MPPoseDetector (Tasks API wrapper)
│   ├── ui/
│   │   ├── app_window.py         # 메인 윈도우 (3 탭)
│   │   ├── tabs/
│   │   │   ├── subjects_tab.py   # 피험자 등록/관리
│   │   │   ├── measure_tab.py    ★ 측정 탭 (1300+ LOC)
│   │   │   └── reports_tab.py    # 리포트 + 리플레이 라우터
│   │   ├── widgets/
│   │   │   ├── camera_view.py    ★ 라이브 카메라 + 포즈 + V6 큐 + V6-G HUD
│   │   │   ├── video_player.py   ★ 리플레이 비디오 + V6 큐 + V6-G HUD
│   │   │   ├── replay_panel.py   # 동기 재생 (force + cop + encoder + angle)
│   │   │   ├── cognitive_hud.py  ★ V6-G: progress bar / grade burst / counters
│   │   │   ├── test_options_panel.py # 테스트 선택 + 옵션 UI
│   │   │   ├── force_dashboard.py # 측정 중 실시간 force 게이지
│   │   │   └── ...               # cop_trail, force_timeline 등
│   │   └── workers/              # QThread 백그라운드 작업
│   │       ├── record_worker.py
│   │       ├── analysis_worker.py
│   │       ├── pose_worker.py    # mp4 → poses_*.npz
│   │       ├── pose_live_worker.py # 실시간 스켈레톤 오버레이
│   │       ├── excel_export_worker.py
│   │       └── csv_export_worker.py # NEW
│   └── db/
│       ├── models.py             # Subject, Session, SessionMetricsRow
│       └── migrations/
├── data/
│   ├── sessions/                 # 세션 폴더 (test_..._<timestamp>_<subj>)
│   ├── subjects.db               # SQLite
│   └── pose_models/              # MediaPipe .task 파일 캐시
├── tests/                        # 332 단위 테스트
└── docs/                         # 본 문서 + CALIBRATION_WORKFLOW + ...
```

---

## 2. 데이터 흐름 다이어그램 (전체)

```
┌────────────────────────────────────────────────────────────────────┐
│  [측정 단계]                                                          │
│                                                                      │
│  GUI:  MeasureTab → TestOptionsPanel → 옵션 dict                     │
│            │                                                          │
│            ▼                                                          │
│        RecorderConfig(test, subject_*, n_stimuli, ...)                │
│            │                                                          │
│            ▼                                                          │
│        RecordWorker (QThread)                                         │
│            │                                                          │
│            ▼                                                          │
│        SessionRecorder.run()                                          │
│           ├─ DaqReader (별도 thread, 100Hz force+enc)                │
│           ├─ MultiCameraCapture (별도 process, mp4 + timestamps.csv) │
│           ├─ StabilityDetector (smart-wait stance)                    │
│           └─ DepartureEventTracker (cognitive_reaction에선 비활성)    │
│            │                                                          │
│            ▼ on_state callback                                        │
│        RecorderState (phase, prompts, V6 cog_*, V6-G hud_*)           │
│            │                                                          │
│            ▼                                                          │
│  GUI:  CameraView 렌더 + ForceDashboard + StabilityOverlay            │
│                                                                      │
│  Output: <session>/forces.csv, <cam>.mp4, <cam>.timestamps.csv,      │
│          stimulus_log.csv, events.csv, session.json                   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  [분석 단계 1차: force만]                                              │
│                                                                      │
│  AnalysisWorker → dispatcher.analyze_session(test_type)              │
│                       │                                                │
│                       ├─ balance_eo/ec → analyze_balance              │
│                       ├─ cmj/sj → analyze_cmj                         │
│                       ├─ squat/overhead_squat → analyze_squat         │
│                       ├─ strength_3lift → analyze_strength_3lift      │
│                       ├─ reaction → analyze_reaction                  │
│                       ├─ cognitive_reaction → analyze_cognitive_reaction │
│                       └─ ... (proprio/encoder/free_exercise)          │
│                       │                                                │
│                       ▼                                                │
│                  result.json (1차)                                     │
│                                                                      │
│  if _auto_pose=True:                                                  │
│  [Pose 처리 + 분석 단계 2차]                                            │
│  PoseWorker → mp4 → MediaPipe BlazePose → poses_<cam>.npz            │
│                       │                                                │
│                       ▼ 끝나면 자동 재실행                              │
│  AnalysisWorker → dispatcher.analyze_session                          │
│                       │                                                │
│                       ▼                                                │
│                  result.json (2차, pose 포함)                          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  [리포트/리플레이/내보내기]                                              │
│                                                                      │
│  ReportsTab → ReportViewer                                            │
│      ├─ build_trainer_report / build_subject_report                   │
│      │     ├─ CoverPage / ExecutiveSummary / Header                   │
│      │     ├─ SummaryCards (test별 카드 빌더)                         │
│      │     ├─ <test>ChartsSection                                     │
│      │     └─ DetailSection / PoseAnglesSection / HistorySection      │
│      ├─ ReplayPanel → VideoPlayerWidget + force/cop/encoder graphs    │
│      ├─ ExcelExportWorker → excel_export.py → xlsx                    │
│      ├─ CsvExportWorker → csv_export.py → 2× csv (NEW)                │
│      └─ PdfExportWorker → pdf_renderer.py → pdf                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 측정 모드 (Test Types)

`TestOptionsPanel.TESTS_KO`(테스트 선택 드롭다운)에 등록된 검사:

| key | Korean | duration_s 기본 | 분석기 |
|---|---|---|---|
| `balance_eo` | 밸런스 (눈 뜨고) | 30 | balance.py |
| `balance_ec` | 밸런스 (눈 감고) | 30 | balance.py |
| `cmj` | CMJ (반동 점프) | 10 | cmj.py |
| `sj` | SJ (반동 없음) | 10 | cmj.py (재사용, V4) |
| `squat` | 스쿼트 | 30 | squat.py |
| `overhead_squat` | 오버헤드 스쿼트 | 30 | squat.py |
| `reaction` | 반응 시간 | 60 | reaction.py |
| **`cognitive_reaction`** | **시각/인지 반응 (V6)** | 60 | cognitive_reaction.py |
| `proprio` | 고유감각 | 60 | proprio.py |
| `free_exercise` | 자유 운동 | 60 | free_exercise.py |
| **`strength_3lift`** | **전신 근력 3대 운동 (V1)** | (multi-set) | strength_3lift.py |

---

## 4. Phase별 구현 상세

각 phase의 **목적 / 임상 근거 / 핵심 코드 / 데이터 / 테스트 / 의존성**을 1:1로 정리합니다.

### V1 — 1RM 등급 평가 기반 (커밋 `909e3fe`)

#### 목적
StrengthLevel 같은 외부 사이트와 동일한 방식으로, **연령 × 성별 × 체중 × 운동** 별 1RM 정상범위를 제공하고 7-band 등급화.

#### 임상 근거
StrengthLevel 통계 + ACSM 1RM 추정 가이드라인:
- Epley: `1RM = load × (1 + reps/30)`
- Brzycki: `1RM = load × 36/(37 - reps)`
- Lombardi: `1RM = load × reps^0.10`
- 앙상블 (3가지 평균) — Reps 1-12 범위에서 신뢰도 높음, 12+ 시 "신뢰 불가" 표시

#### 등급 cutoff (1=엘리트 → 7=초보)
```python
# src/analysis/strength_norms.py
GRADE_LABELS_KO = {
    1: "엘리트", 2: "고급", 3: "중상급",
    4: "중급",   5: "초중급", 6: "초보", 7: "신참",
}
GRADE_PERCENT = [99, 90, 75, 50, 25, 10, 0]   # population percentile
```

#### 6개 norm 테이블 (각 body weight 구간별)
- `BENCH_PRESS_M`, `BENCH_PRESS_F`
- `BACK_SQUAT_M`, `BACK_SQUAT_F`
- `DEADLIFT_M`, `DEADLIFT_F`

각 테이블: `{ bw_kg: [grade1_kg, grade2_kg, ..., grade7_kg] }` 보간으로 임의 체중 처리.

#### 코드 위치
- `src/analysis/strength_norms.py` (~250 LOC)
- `src/analysis/one_rm.py` (~120 LOC)
- 테스트: `tests/test_strength_norms.py` (35) + `tests/test_one_rm.py` (19)

### V1-D — 다중세트 state machine (커밋 `30595ee`)

#### 목적
operator-driven 멀티세트 검사. recording → inter_set_rest → recording 사이클을 N번 반복.

#### state machine
```
idle → wait → countdown → recording (set 1)
                             │ end_set()
                             ▼
                        inter_set_rest (rest_s 카운트다운)
                             │ pause_rest / resume_rest / skip_rest
                             ▼
                        recording (set 2) → ... → done
```

#### 외부 트리거 (GUI 버튼)
- `recorder.end_set()` — 세트 종료
- `recorder.pause_rest()` / `resume_rest()` — 휴식 일시정지/재개
- `recorder.skip_rest()` — 휴식 건너뛰기
- `recorder.end_session()` — 세션 종료 (cancel과 다름 — 데이터 보존)

### V1.5 — 자체중 보정 (커밋 `589fd96`)

#### 운동별 BW 가산 계수
```python
EXERCISE_BW_FACTOR = {
    "bench_press": 0.0,   # 체중이 봉에 안 실림
    "back_squat":  0.85,  # 하퇴/발 외 신체가 같이 들림
    "deadlift":    0.10,  # 신체 COM 소폭 상승
}
effective_load = load_kg + (factor × subject_kg if use_bodyweight_load else 0)
```

#### 의의
빈 봉만 들어도 1RM 추산 가능. 여성/고령자 대상 의미 있는 데이터 산출.

#### 신뢰도 체계
- `excellent`: load_kg ≥ 70% effective, reps 1-3
- `high`: 60-70%, reps 4-6
- `medium`: 50-60%, reps 7-12
- `low`: 30-50%
- **`unreliable`**: reps > 12 또는 load < 30% effective → 리포트에 빨강 표시

### V1-bugfix — Subject 컨텍스트 흐름 (커밋 `f49aa7e`)

#### 문제
session.json에 subject_sex / birthdate가 없어서 strength_3lift 등급 lookup 시 NULL 반환.

#### 수정
`RecorderConfig`에 `subject_sex`, `subject_birthdate`, `subject_height_cm` 추가 → MeasureTab이 DB Subject 행에서 읽어와 RecorderConfig에 주입 → recorder가 session.json에 저장 → 분석기가 거기서 직접 읽음.

DB 폴백도 유지 (session.json에 없으면 subjects.db를 subject_id로 조회).

### V2 — ATP-PCr 회복지수 (커밋 `1370cc7`)

#### 목적
다중세트 검사에서 세트별 power 감소율로 ATP-PCr 시스템 회복 능력 측정.

#### 임상 근거
근섬유 타입 추정 (Bompa, Periodization for Sports):
- **Type II 우세 (속근)**: 1세트에 power 폭발, 후속 세트에서 큰 감소 (피로↑)
- **Type I 우세 (지근)**: 세트 간 power 비교적 일정 (피로 저항↑)

#### 산출식
```python
FI = (Set1_power - SetN_power) / Set1_power × 100        # Fatigue Index %
PDS = (1 - sum(set_powers) / (Set1 × N)) × 100            # Performance Decrement Score
```

#### 등급
| FI 범위 | 의미 | fiber_tendency |
|---|---|---|
| < 5% | 매우 우수 | type_I (지근) |
| 5-15% | 우수 | mixed |
| 15-30% | 보통 | mixed |
| 30-45% | 미흡 | type_II_dominant |
| > 45% | 부족 | type_II (속근, 빠른 피로) |

#### 코드
- `src/analysis/multiset_recovery.py` (~320 LOC)
- 테스트: `tests/test_multiset_recovery.py` (27)

### V3 — 다중세션 합성 등급 + 신체 다이어그램 (커밋 `0981d3e`, `ca6f039`)

#### 목적
사용자의 **여러 세션**을 모아 부위별 가중평균 등급 산출.

#### 부위 매핑
```python
EXERCISE_REGION = {
    "bench_press": "upper",   # 상체
    "back_squat":  "lower",   # 하체
    "deadlift":    "total",   # 전신
}
REGION_WEIGHTS = {"upper": 0.30, "lower": 0.40, "total": 0.30}
```

#### 합성 점수
```python
composite_score = Σ(region_grade_score × region_weight) / Σ(region_weight)
# region_grade_score: 1등급=100, 7등급=0의 선형 매핑
```

#### 신체 다이어그램 (V3-vis)
matplotlib polygon 기반의 **앞/뒤 인체 실루엣**, 부위별 등급 색칠. `src/reports/charts.py:make_body_strength_diagram` (~600 LOC).

> 사용자께서 향후 자체 이미지로 교체 예정 (현재는 polygon 기반).

### V4 — SSC (Stretch-Shortening Cycle) 분석 (커밋 `4131c16`)

#### 목적
CMJ vs SJ 비교로 **신장-단축 사이클(SSC) 활용도** 측정. 폭발력 vs 탄성 능력 진단.

#### 산출식
```python
EUR (Eccentric Utilization Ratio) = h_CMJ / h_SJ
SSC% (contribution) = (h_CMJ - h_SJ) / h_SJ × 100
```

#### 해석 분기
- **EUR > 1.10**: SSC 우수 (탄성 활용 양호)
- **EUR 1.0-1.10**: 보통
- **EUR < 1.0**: SSC 비효율 (CMJ가 SJ보다 안 좋음 → 동작 패턴 문제)

#### 듀얼 해석
점프 높이 절대값에 따라 **strength_focus** (높이 부족 → 근력 보강) 또는 **elastic_focus** (높이는 OK but EUR 낮음 → 폭발력 훈련) 분기 메시지.

#### 코드
- `src/analysis/ssc.py` (~280 LOC)
- 테스트: `tests/test_ssc.py` (26)
- 차트: `make_ssc_jump_comparison`, `make_ssc_grade_band`

### V5 — Squat 정밀 동작 보강 (커밋 `f7b24af`)

#### 추가된 메트릭
1. **CoP 안전 경로 등급**: AP 드리프트 + ML 드리프트 최대값 기준 5단계
2. **L/R 좌우 비대칭** (per-rep, eccentric/concentric 분리): 5%/10% 임계로 caution/warning

#### CoP 안전 등급 (3-band table)
```python
_COP_SAFETY_BANDS = [
    # (등급, max_AP_rear, max_AP_fwd, max_ML)
    (1, -25, +5,  25),    # 안전
    (2, -40, +15, 40),    # 양호
    (3, -55, +25, 55),    # 보통
    # 4 / 5는 밴드 초과 + warning 분기
]
```

#### Warning 분기
- `forward_lean`: AP 드리프트 +25mm 초과 (앞으로 쏠림 → 무릎 부하)
- `rearfoot_excessive`: AP -55mm 초과 (뒤로 쏠림)
- `lateral_drift`: ML 70mm 초과 (좌우 흔들림)

각 warning은 5등급 (extreme) 도달 시 더 진한 빨강.

#### L/R 비대칭
```python
ASYM_CAUTION_PCT = 5.0    # 노란 경고
ASYM_WARNING_PCT = 10.0   # 빨간 경고
```
phase 별로 (eccentric / concentric) 따로 산출 — 하강에서만 비대칭이거나 상승에서만 비대칭일 수 있음.

#### 코드
- `src/analysis/squat.py`에 모든 함수 추가
- 테스트: `tests/test_squat_v5.py` (23)
- 섹션: `src/reports/sections/squat_precision.py`

---

## 5. **V6 — 시각/인지 반응 검사 (가장 큰 작업)**

총 7개 phase + 다수의 fix 거쳐 완성. 각 단계 정리:

### V6 (백엔드, 커밋 `e119bd7`)

#### 목적
**시지각·인지·운동 통합 반응** 검사. 화면에 위치 큐가 LED처럼 표시되고, 피험자는 지정된 신체 부위를 그 위치로 이동. 분석기가 RT/MT/공간정확도 산출.

#### 핵심 데이터 구조

```python
# src/capture/session_recorder.py
COGNITIVE_REACTION_POSITIONS_4 = [
    ("pos_N", 0.50, 0.20),  # 화면 normalized 좌표 (top-left=0,0)
    ("pos_E", 0.85, 0.50),
    ("pos_S", 0.50, 0.80),
    ("pos_W", 0.15, 0.50),
]
COGNITIVE_REACTION_POSITIONS_8 = [...]  # 8 cardinal+ordinal directions
```

#### 분석기 — `_per_cam_metrics` 알고리즘 (V6-fix2 강화 후)

```python
1. resolve_pose_frame(t_stim_s) → i_stim
   (timestamps.csv 우선, 없으면 fps 폴백)
2. baseline window (자극 직전 0.4초)
   pre_speed = norm(diff(pre_kpts))
   thr = max(min_motion_speed_px,
             min(bl_mean + onset_sigma * bl_std,
                 30.0))   # 상한 캡 (이전 trial 잔여 모션 방지)
3. post window (자극 후 max_response_s = 2.5초)
   post_speed = norm(diff(post_kpts))
   visibility 게이트: vis ≥ 0.5
4. Motion onset = first frame where post_speed ≥ thr
5. PROXIMITY FALLBACK (V6-fix2):
   if onset_rel is None:
       below_radius = dists ≤ hit_radius_px (= 0.12 × diag)
       if below_radius.any():
           onset_rel = first_hit_frame
           failure_reason = "ok_proximity_hit"
6. End of reach = closest approach AFTER onset
7. RT = onset_rel / fps × 1000 ms
   MT = (end_rel - onset_rel) / fps × 1000 ms
   err_norm = closest_dist / image_diagonal
   hit = err_norm ≤ hit_tolerance_norm (0.12)
```

#### Body part → MP33 매핑 (live + offline 일치 필수)
```python
BODY_PART_TO_KEYPOINTS = {
    "right_hand": ["right_wrist", "right_index"],
    "left_hand":  ["left_wrist",  "left_index"],
    "right_foot": ["right_foot_index", "right_ankle"],
    "left_foot":  ["left_foot_index",  "left_ankle"],
}
```

### V6-UI (커밋 `5f10fb7`)

#### TestOptionsPanel 추가
- 검사 드롭다운에 "시각/인지 반응 (위치 큐 + 스켈레톤)" 추가
- 옵션 그룹: 추적 부위 (4종), 위치 수 (4 or 8), 자극 횟수, 간격, 자동/수동

#### CameraView LED 큐 + 추적 점
- `_draw_positional_cue` — FITWIN 그린 (BGR `(0, 245, 170)`) 헤일로 + 흰 코어 + 펄스 (60-tick 사이클)
- 하단 라벨 (drop-shadow)
- normalized 좌표 → 픽셀 변환 (image_size 비율 기준)

#### 라이브 hit 검출 + 체크 효과 (커밋 `5c62abb`)
```python
def _evaluate_hit(self, w, h):
    # 추적 keypoint를 normalized로 변환
    err_norm = sqrt(dx_px² + dy_px²) / diag
    return err_norm ≤ 0.12 (hit_tolerance_norm)
```
**hit 시 시각 효과**:
- 헤일로 → 시안-노랑 (`(80, 255, 255)`)
- 코어 → 그린 + ✓ 체크마크
- 펄스 빠르게 (1초 주기)
- 라벨 앞에 "✓ "
- 추적 손목 dot도 그린 9px 링으로

**Sticky latch**: 12-frame (~0.4s) hold — 손이 잠깐 흔들려도 hit 깜빡임 방지.

### V6-fix 시리즈

| 커밋 | 문제 | 수정 |
|---|---|---|
| `4b403f8` | smart-wait이 손-뻗기 검사에서 stance 못 잡아 영원히 wait | smart-wait skip (V6-G5에서 다시 통일) |
| `b24fdc5` | DAQ zero-cal 미완료 상태로 녹화 시작 → 첫 0.5s 노이즈 + auto_pose 기본 비활성으로 분석 0건 | `_zero_cal_only` 모드 (wait phase로 zero-cal 5.5초 보장) + `_auto_pose=True` 강제 잠금 |
| `c77607e` | live ✓는 잡혔는데 분석기는 무응답으로 처리 | proximity fallback + visibility 게이트 + threshold cap + per-trial `failure_reason` 진단 |
| `ae00483` | 시각/인지 검사에 ⚠ 이탈 banner 잘못 표시 | recorder + ReplayPanel에서 cognitive_reaction은 departure 추적 비활성 + replay LED + ✓ overlay 추가 |

### V6-G — 게임화 점수 시스템 (커밋 `ec254e4`, `95f798f`, `819a3d8`)

#### G1 — Cognitive Reaction Index (CRI)

**시행별 등급**:
| 등급 | RT 조건 (hit) | weight | 색상 |
|---|---|---|---|
| GREAT | ≤ 350ms | 1.00 | 골드 |
| GOOD | ≤ 500ms | 0.75 | 그린 |
| NORMAL | ≤ 750ms | 0.50 | 옐로 |
| BAD | > 750ms hit | 0.20 | 빨강 |
| MISS | no_hit | 0.00 | 다크빨강 |

**합성 점수**:
```python
MS = mean(grade_weight) × 100             # 속도+정확도
AS = (n_hit / n_trials) × 100             # 정확도
CS = max(0, 1 - CV_RT) × 100              # 일관성
CRI = 0.50×MS + 0.30×AS + 0.20×CS         # 종합
```

**가중치 근거** (Posner 1980 / Salthouse 1996 / Bunce 2008):
- 50% MS: 임상 RT 검사 표준 (속도 + 정확도 결합 단일 지표)
- 30% AS: 인지-운동 통합 정확도
- 20% CS: 일관성 (CV > 0.3 → 신경학 의심)

**종합 등급 cutoff**:
```python
CRI_GRADE_BANDS = [
    (85.0, "A", "매우 우수"),
    (70.0, "B", "우수"),
    (55.0, "C", "보통"),
    (40.0, "D", "미흡"),
    (0.0,  "E", "부족"),
]
```

#### G2 — HUD 인프라 (`src/ui/widgets/cognitive_hud.py`)

**Pure BGR 그리기 함수** (Qt 의존성 없음 → 라이브 + 리플레이 둘 다 동일하게 사용):
- `draw_progress_bar(bgr, n_done, n_total, cri_live)` — 상단
- `draw_grade_message(bgr, grade, rt_ms, age_frames)` — 중앙 페이드
- `draw_grade_counters(bgr, grade_counts)` — 하단 칩
- `draw_full_hud(...)` — 모두 합친 wrapper

**상수**:
```python
GRADE_BGR = {"great":(0,215,255), "good":(50,230,50), ...}
GRADE_TEXT = {"great":"GREAT!", ...}
GRADE_MSG_HOLD_FRAMES = 24  # 0.8초 @ 30Hz
```

#### G3 — 라이브 HUD 통합

**RecorderState 확장**:
```python
cog_progress_done: int           # 발사된 자극 수
cog_progress_total: int          # = cfg.n_stimuli
cog_recent_grade: Optional[str]  # 직전 시행 등급
cog_recent_rt_ms: Optional[float]
cog_recent_grade_t_ns: int
cog_grade_counts: dict
cog_live_cri: float
```

**Hit feedback loop**:
```
CameraView._evaluate_hit (per repaint)
   ↓ raw_hit
CameraView._flush() → cog_hit_state_changed signal (transition만 emit)
   ↓
MeasureTab → RecordWorker.feed_hit_indicator(is_hit, t_ns)
   ↓
SessionRecorder._cog_active_stim["first_hit_t_ns"] = t_ns

(1.5초 cue window 만료 시)
SessionRecorder._tick_recording → _resolve_cog_stim
   ↓
RT = first_hit - stim_t / 1e6 ms
grade = grade_trial(rt, hit)
RecorderState.cog_* 업데이트
```

#### G4 — 리플레이 HUD 통합

`VideoPlayerWidget`이 `result.json["result"]["trials"]`을 읽어, 현재 슬라이더 시각 기준:
- n_done = 시행수 (지금까지 완료된)
- recent_grade = 그 시각이 어느 trial의 burst window (stim+1.5s ~ stim+2.3s) 안인지
- live_cri = `compute_cri(trials_done)`
- 동일한 `draw_full_hud` 호출 → 측정 모드와 픽셀 단위로 동일

#### G5 — 측정 루틴 통일

V6 fix1에서 cognitive_reaction은 smart-wait 우회였는데, V6-G5에서 다른 검사들과 동일하게:
```
zero-cal (5.5s) → wait-for-two-foot-stance → countdown (5s) → recording
```
이탈 추적은 여전히 비활성 (V6 fix 보존) — 측정 시작 직전에만 stance 검증, 시작 후엔 흐름 자유.

#### G6 — 리포트 추가
- 큰 CRI 카드 (점수/100 + 등급 A..E + 라벨)
- 등급별 카운트 chip strip
- MS/AS/CS sub-score 표 행
- failure_reason_counts 진단 블록 (n_valid=0 시 빨간 경고)

#### G7 — End-to-end 통합 테스트
`tests/test_v6g_integration.py` (8) — 라이브 ↔ 분석기 ↔ 리포트 grade 일치 검증.

---

## 6. CSV Export (커밋 `fb5cd36`)

### 목적
메모장에서 바로 열리는 평문 export. Excel과 동일 데이터를 CSV로.

### 핵심 디자인 결정 — 이벤트 컬럼이 시간 옆에

```
t_s, t_wall_s,
event_stim, event_target_x_norm, event_target_y_norm, event_off_plate,
b1_tl_N, ... (force 11), enc1_mm, enc2_mm,
b1_cop_x_mm, ... (CoP 6), on_plate,
nose_x_px, nose_y_px, nose_vis, ... (33×3 keypoint),
nose_vx_px_s, nose_vy_px_s, ... (33×2 velocity),
angle_knee_L_deg, ... (12 angle),
angle_vel_knee_L_deg_s, ... (12 angular velocity)
```

총 **~215 컬럼**.

### 이벤트 컬럼 의미
- `event_stim`: stimulus_log의 response_type / target_label, 발사 시각의 가장 가까운 force sample 행에만 (그 외 빈칸)
- `event_target_x_norm`, `event_target_y_norm`: V6 cognitive_reaction target 좌표 (발사 행만)
- `event_off_plate`: events.csv 기반 0/1 (이탈 구간 전체에 1)
- 별도로 `on_plate`: forces.csv의 sample-by-sample 분류 (20N 임계, raw)

### 출력
- `<session>_timeseries.csv` — 100Hz force 그리드 + 보간된 포즈
- `<session>_pose_native.csv` — 카메라 native fps 포즈 (보간 없음)

UTF-8 BOM (`EF BB BF`) → 메모장 + Excel 둘 다 한글 깨짐 없음.

### Float-quantization safety
`np.arange(N) / fs`로 만든 t_s가 정수 e와 같을 때 FP 에러로 경계 sample 누락되는 문제 → `eps = 1e-9` 허용오차 마스킹으로 해결.

---

## 7. 디자인 결정 사례집

### 7.1 좌표계 일관성

문제: 카메라 회전이 어디서 적용되는가?

해결: `camera_worker.py:_orient` 가 **mp4 저장 직전**에 회전 적용. 따라서:
- 라이브 preview (회전된 frame)
- mp4 (회전된 frame)
- post-record PoseWorker (mp4 직접 read)
- 라이브 PoseLiveWorker (회전된 frame 입력)

**4개 경로 모두 동일한 좌표계** → target_x_norm을 어디서 곱해도 같은 픽셀 위치.

### 7.2 라이브 ↔ 오프라인 일치 (V6 lessons learned)

**원칙**: 임상 결과는 라이브 (✓ 색 변화)와 오프라인 (리포트 카운트)이 **반드시** 일치해야 함.

**구현**:
- 동일한 `hit_tolerance_norm = 0.12` 상수
- 동일한 `BODY_PART_TO_KEYPOINTS` 매핑
- 동일한 `_VIS_THRESH = 0.5` visibility 게이트
- Proximity fallback (분석기): live가 ✓ 잡았던 경우 무조건 hit으로 분류

### 7.3 Test type별 분기 최소화

분석기 / 리포트 빌더가 `test_type`을 한 번만 검사하고 라우팅:

```python
# dispatcher.py
_ANALYZERS = {"balance_eo": analyze_balance_file, ...}

# report_builder.py
_CHARTS_FOR_TEST = {"balance_eo": BalanceChartsSection, ...}
_CARD_BUILDERS = {"balance_eo": _balance_cards, ...}
```

이후 호출 사이트에서는 분기 없음.

### 7.4 1차 + 2차 분석 (post-record pose)

```
녹화 종료 → AnalysisWorker (1차, force만, pose 없음)
   → result.json (pose 필드 NaN)
   → if _auto_pose=True:
        PoseWorker (mp4 → poses_*.npz)
        → AnalysisWorker (2차, pose 포함)
        → result.json 갱신
```

**중요**: 1차 분석에서 에러가 나면 2차로 못 넘어감. cognitive_reaction은 1차에서 FileNotFoundError를 안 내도록 stimulus_log.csv 누락 시 빈 결과를 반환해야 함.

### 7.5 PoseLiveWorker drop-on-full

라이브 포즈는 latest-frame-only mailbox 패턴. 처리량이 카메라 fps 못 따라가면 drop.

**부작용**: 라이브에서 매 frame을 처리하지 못해 **간헐적으로만 hit 검출**. PoseWorker (post-record)는 모든 frame 처리하므로 visibility/motion 분포가 더 보수적 → V6-fix2의 proximity fallback이 두 경로 차이를 메움.

### 7.6 RecorderState — single source of truth

GUI는 `RecorderState`만 보고 그림. 직접 recorder 내부 상태 access X. signal-slot으로 한 방향 전달.

### 7.7 Color palette (FITWIN)
```python
# src/reports/palette.py
STATUS_OK       = "#4CAF50"   # 그린
STATUS_CAUTION  = "#FFB300"   # 주황
STATUS_WARNING  = "#E53935"   # 빨강
HISTORY_LINE    = "#1976D2"   # 파랑 (트렌드)
NORM_BAND_FILL  = "#E8F5E9"   # 부드러운 그린 (정상범위)
TOTAL_COLOR     = "#A5D6A7"   # 합계 라인
```

CameraView 라이브 그리기 BGR:
- LED rest: `(0, 245, 170)` FITWIN green
- LED hit:  `(80, 255, 255)` cyan-yellow + `(50, 230, 50)` green core
- Skeleton edges: `(0, 200, 0)` green
- Joint dots: `(0, 255, 255)` yellow

---

## 8. 테스트 매트릭스

총 **332 테스트 / 17 파일**. 테스트는 모두 `python tests/test_*.py` 직접 실행 가능 (pytest 도 OK).

| 파일 | 테스트 | 커버 |
|---|---|---|
| test_strength_norms.py | 35 | V1 norm tables, grade_1rm, composite_score, EXERCISE_BW_FACTOR |
| test_one_rm.py | 19 | V1-C Epley/Brzycki/Lombardi + 앙상블 + reps 신뢰도 |
| test_strength_3lift_state.py | 23 | V1-D state machine 전이 (recording→rest→recording, end_set, pause/skip) |
| test_strength_3lift_analysis.py | 21 | V1-F 분석기 — 세트 boundaries, best_1rm, 신뢰도 |
| test_multiset_recovery.py | 27 | V2 FI/PDS, fiber_tendency 분기 |
| test_composite_strength.py | 9 | V3 다중세션 합성 — region weights, missing region 처리 |
| test_ssc.py | 26 | V4 EUR/SSC%, dual interpretation 분기 |
| test_squat_v5.py | 23 | V5 CoP safety bands, asymmetry levels, quiet stance 계산 |
| test_recorder_finalize.py | 14 | _finalize 순서, DAQ tail trim, record_end snapshot |
| test_departure_events.py | 17 | DepartureEventTracker 상태 전이 + min_duration 필터 |
| test_cognitive_reaction.py | 26 | V6 분석기 + recorder + smart-wait 통일 |
| test_cognitive_reaction_ui.py | 25 | V6 TestOptionsPanel, CameraView hit detection, draw 함수 |
| test_replay_cognitive_overlay.py | 11 | V6 ReplayPanel/VideoPlayerWidget 큐 + 체크 |
| test_cognitive_grading.py | 20 | V6-G1 grade_trial / compute_cri / CRI letter grades |
| test_cognitive_hud.py | 15 | V6-G2 HUD 그리기 — progress / message / counters |
| test_v6g_integration.py | 8 | V6-G7 e2e — 라이브 ↔ 분석기 grade 일치 검증 |
| test_csv_export.py | 13 | CSV export — BOM, header order, event cols, NaN, pose-native |

### 빠르게 전체 sweep 돌리기
```python
# run_all_tests.py 만들어둘 것 권장
files = ["tests/test_*.py", ...]
for f in files: subprocess.run([sys.executable, f])
```

---

## 9. 향후 개발 가이드

### 9.1 새 검사 추가 시 체크리스트

1. **`session_recorder.py`**:
   - `TEST_PROMPTS`에 추가
   - `RecorderConfig` 새 필드 (필요 시)
   - state machine 분기 (필요 시)
   - `_build_metadata`에 메타 키 저장

2. **`src/analysis/<test>.py`**:
   - `analyze_<test>(force, ...)` + `analyze_<test>_file(session_dir, **kw)`
   - 결과 dataclass + `to_dict()` 메서드

3. **`dispatcher.py`**:
   - `_ANALYZERS`에 등록

4. **리포트**:
   - `src/reports/sections/<test>.py` (HTML + PDF)
   - `src/reports/charts.py`에 차트 함수
   - `report_builder.py` `_CHARTS_FOR_TEST` + `_CARD_BUILDERS`에 등록

5. **GUI**:
   - `TestOptionsPanel.TESTS_KO`에 추가
   - 옵션 그룹 (필요 시)
   - `MeasureTab`에 special handling (필요 시)

6. **테스트** (필수):
   - `tests/test_<test>.py`

### 9.2 V6-G grade 임계 조정 시

`src/analysis/cognitive_reaction.py` 상수만 변경:
```python
RT_GREAT_MS  = 350.0
RT_GOOD_MS   = 500.0
RT_NORMAL_MS = 750.0
GRADE_WEIGHT = {"great": 1.00, ...}
```
`tests/test_cognitive_grading.py`의 boundary 테스트도 같이 갱신 필요.

### 9.3 신체 다이어그램 교체 (V3-vis)

사용자 제공 이미지로 교체 시:
- `src/reports/charts.py:make_body_strength_diagram` 의 `_FRONT_POLYGON` / `_BACK_POLYGON` 좌표를 이미지로 교체
- 부위별 색칠 영역도 이미지 mask로 변환

### 9.4 새 점수 시스템 추가 시 (V6-G CRI 패턴)

V6-G CRI가 좋은 템플릿:
1. `analysis/<test>.py`에 `compute_<index>(trials)` 헬퍼
2. dataclass에 `cri`, `mean_score`, ... 추가
3. `analyze_<test>` 끝에 `compute_<index>(trials)` 호출 결과 분해해서 dataclass 채움
4. `report_builder.py:_<test>_cards`에 lead card로
5. `sections/<test>.py`에 큰 카드 + 등급 chip strip

### 9.5 CSV/Excel export 컬럼 추가

`csv_export.py` + `excel_export.py` 둘 다 `_per_board_cop` / `_pose_to_force_grid` / `_velocity` 헬퍼 공유.

새 컬럼 추가 위치:
- TimeSeries: `_write_timeseries_csv` 의 `cols: dict` 또는 `pose_cols: dict`
- 헤더 순서: `header += ...` 라인에서 그룹 위치 결정

`tests/test_csv_export.py`의 `test_header_order_events_immediately_after_time` 도 갱신.

### 9.6 분석 재실행 (V6-G1 같은 사후 추가 시)

기존 세션은 **데이터(forces.csv, mp4, npz)는 그대로**, 새 코드로 재분석만:

```python
from src.analysis.dispatcher import analyze_session
analyze_session("data/sessions/<name>")   # write_result=True 기본
```

GUI에는 "🔄 재분석" 버튼 (`_btn_analyze`) 이미 있음.

---

## 10. 알려진 이슈 / 향후 작업 (deferred)

### 10.1 V3-vis 신체 실루엣
사용자께서 직접 그린 이미지로 교체 예정. polygon 기반 현재 버전은 임시.

### 10.2 V7 (계획만)
**동적 균형 + COM 제어** — 측정 중 외란 (perturbation)에 대한 회복 능력. 미구현.

### 10.3 V8 (계획만)
**코어 안정성 + 자세 + ROM** — 플랭크 시간, 자세 각도 변화율, 관절 가동범위. 미구현.

### 10.4 8pos 30x cognitive_reaction
auto-trigger 스케줄러가 `t_cur >= duration_s - 1` 조건으로 일찍 끊겨서 30 자극 설정해도 16개만 발사됨. 더 많이 원하면 duration_s 증가 필요. (스케줄러 알고리즘 자체는 정상.)

### 10.5 session_metrics 캐시
V6-G CRI 등 새 필드는 캐시에 들어가지 않음. 리포트는 result.json을 직접 읽으니 OK. 다중세션 트렌드 차트가 필요하면 `key_metrics.py:extract_key_metrics`에 cri/grade 추출 로직 추가.

### 10.6 Camera fps mismatch
일부 카메라가 60fps로 캡처되는데 mp4 metadata는 30fps로 저장될 수 있음. `cv2.VideoWriter` 호출 시 `fps`를 정확히 지정해야 함. 현재 `config.CAMERA_FPS` 의존.

---

## 11. 환경 / 의존성

### 11.1 Python
```
Python 3.11.x (Windows)
```

### 11.2 핵심 의존성
```
PyQt6 >= 6.5
pyqtgraph >= 0.13
nidaqmx >= 0.6
opencv-python >= 4.8
mediapipe >= 0.10
numpy
scipy
pandas
matplotlib
reportlab
openpyxl >= 3.1
```

### 11.3 하드웨어
- NI-DAQ USB-6210 또는 호환 (8 force + 2 encoder 채널)
- Logitech StreamCam (USB 3.0, 1280x720@60fps)
- Force plate: BOARD_WIDTH_MM × BOARD_HEIGHT_MM (config.py에 정의)

### 11.4 모델 캐시
첫 실행 시 MediaPipe `.task` 파일 (lite/full/heavy) 자동 다운로드 → `data/pose_models/`

---

## 12. Git / 배포

### 12.1 원격
- **Origin**: `https://github.com/fitwindev-oss/analysis.git`
- **브랜치**: `main`

### 12.2 태그
- `v.01.2026.04.28_testend` — 본 인수인계 시점

### 12.3 커밋 컨벤션
```
<type>(<phase>): <짧은 요약>

<상세 변경 내용>
- ...

<테스트 영향>
All N tests passing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### 12.4 주요 커밋 ref
| Tag/커밋 | 내용 |
|---|---|
| `37de66d` | Initial commit |
| `909e3fe` | V1 1RM grading foundation |
| `1370cc7` | V2 ATP-PCr recovery |
| `0981d3e` | V3 composite strength |
| `4131c16` | V4 SSC analysis |
| `f7b24af` | V5 squat precision |
| `e119bd7` | V6 cognitive_reaction (backend) |
| `5f10fb7` | V6-UI |
| `5c62abb` | V6 live ✓ feedback |
| `4b403f8`, `b24fdc5`, `c77607e`, `ae00483` | V6 fix series |
| `ec254e4`, `95f798f`, `819a3d8` | V6-G CRI + HUD + integration |
| `fb5cd36` | CSV export |
| **`v.01.2026.04.28_testend`** | **본 인수인계 시점 스냅샷** |

---

## 13. 빠른 시작 (차기 개발자)

### 13.1 클론 + 설정
```bash
git clone https://github.com/fitwindev-oss/analysis.git biomech-mocap
cd biomech-mocap
git checkout v.01.2026.04.28_testend  # 본 인수인계 시점
"C:\Program Files\Python311\python.exe" -m pip install -r requirements.txt
```

### 13.2 첫 실행
```bash
"C:\Program Files\Python311\python.exe" -m src.ui.app_window
# 또는
"C:\Program Files\Python311\python.exe" main.py   # 있을 경우
```

### 13.3 테스트 sweep
```bash
"C:\Program Files\Python311\python.exe" tests/test_csv_export.py
# (각 파일 직접 실행 또는 pytest)
```

### 13.4 데이터 / 세션 위치
```
data/sessions/<test>_<options>_<timestamp>_<subject_id>/
├── forces.csv
├── stimulus_log.csv (있으면)
├── events.csv (있으면)
├── <cam>.mp4
├── <cam>.timestamps.csv
├── poses_<cam>.npz (auto_pose=True 시)
├── session.json
├── result.json
├── *_timeseries.csv (CSV export 시)
├── *_pose_native.csv (CSV export 시)
└── *.xlsx (Excel export 시)
```

### 13.5 빠르게 한 세션 분석
```python
from src.analysis.dispatcher import analyze_session
result = analyze_session("data/sessions/cognitive_reaction_...")
print(result["result"]["cri"])
```

---

## 14. 연락처 / 라이센스

- 프로젝트 소유: **FITWIN**
- 본 인수인계 자료: 차기 개발자 onboarding 용
- 라이센스: 내부용

---

**문서 버전**: 1.0
**작성일**: 2026-04-28
**기준 태그**: `v.01.2026.04.28_testend`
**총 코드량**: ~22,000 LOC (src/) + ~6,000 LOC (tests/) + 332 테스트
