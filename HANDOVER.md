# biomech-mocap 인수인계 문서

> 본 문서는 FITWIN MoCap Clinic에서 운영 중인 `biomech-mocap` 코드베이스를 차세대
> 시스템 개발팀에 이관할 때 참조할 **단일 진실의 원천(source of truth)** 입니다.
> 기술 스펙·하드웨어 상태·인수 절차·미해결 이슈를 한 문서에 집약합니다.

**작성일:** 2026-04-23
**기준 코드 버전:** `APP_VERSION = "0.1.0"` (config.py 기준)
**인수 대상:** Windows 10 Pro + NI USB-6210 DAQ + 2 force plates + 2 linear encoders + 1 Logitech StreamCam

---

## 0. TL;DR

- 리포지토리 **git 미초기화 상태** — 커밋 히스토리 없음. 인수는 **디렉토리 zip + 본 문서**로 진행.
- Python **3.11.9**, 주 의존성: PyQt6, nidaqmx, reportlab, matplotlib, mediapipe.
- 현재 하드웨어: Board1 TL/TR/BL/BR + Board2 TL/TR/BL/BR (8 load cells), Encoder1 + Encoder2 (우측 고장), 1 USB camera (portrait 회전).
- 최근 캘리브레이션: Force scale 2026-04-22, Encoder1 scale 2026-04-23 (`784.23 mm/V`).
- **반드시 먼저 읽을 문서:**
  1. [`docs/FORCE_PLATE_SYSTEM_SPEC.md`](docs/FORCE_PLATE_SYSTEM_SPEC.md) — 하드웨어·채널·캘리 정본
  2. [`docs/CALIBRATION_WORKFLOW.md`](docs/CALIBRATION_WORKFLOW.md) — 카메라/월드프레임 캘리
  3. 본 문서의 §6 (코드 워크스루) → §7 (함정) → §9 (향후 통합 계획)

---

## 1. 전달 형식

### 1.1 현재 저장소 상태 (확인 필요 사항)

**이 프로젝트는 git 저장소가 아닙니다** (`.git/` 없음). 따라서:

| 옵션 | 가능 여부 | 비고 |
|---|---|---|
| GitHub/GitLab read 권한 | ❌ | 리포지토리 미등록 |
| `git bundle` / `git clone --mirror` | ❌ | 커밋 히스토리 없음 |
| **워킹 디렉토리 zip + 본 문서** | ✅ | **유일한 선택지** |

### 1.2 권장 조치 (인수 후 최우선 작업)

인수 팀은 **받자마자 git 초기화**를 권장합니다:

```bash
cd biomech-mocap
git init
git branch -m main
git add -A
git commit -m "Initial import from FITWIN MoCap 2026-04-23 handover"
git tag -a v0.1.0-handover -m "Handover baseline"
# 이후 차세대 개발 시작
```

본 handover 시점의 상태는 **태그 `v0.1.0-handover`** 로 고정하고 이후 변경은 브랜치에서.

### 1.3 패키징 스크립트

아래 명령으로 용량 최적화된 zip 생성 (가상환경·캐시·대용량 세션 제외):

```bash
# Windows PowerShell (프로젝트 루트에서)
python scripts/package_handover.py     # 아직 없음 — §11 수동 절차 참고
```

수동 zip 생성 절차는 **§11 인수인계 패키지 빌드**에 구체적으로 기술.

---

## 2. 반드시 포함 — 체크리스트

| 항목 | 경로 | 포함 여부 |
|---|---|---|
| 전체 소스 | [`src/`](src/) | ✅ (capture, analysis, ui, reports, calibration, db, io, pose) |
| 현재 캘리 값 | [`config.py`](config.py) | ✅ 가동 중 값 그대로 |
| 스크립트 전체 | [`scripts/`](scripts/) | ✅ 17개 (캘리/분석/진단/기록) |
| 의존성 | [`requirements.txt`](requirements.txt) | ✅ |
| 진입점 | [`main.py`](main.py) | ✅ |
| 문서 | [`docs/`](docs/) | ✅ FORCE_PLATE_SYSTEM_SPEC, CALIBRATION_WORKFLOW, CHARUCO_PRINT_GUIDE |
| 최신 캘리 결과 | [`data/calibration/`](data/calibration/) | ✅ encoder_verify_*.json ×5 + 카메라 intrinsics/extrinsics |
| 앱 리소스 | [`resources/`](resources/) | ✅ charuco_board.png + mediapipe/ (모델 캐시) |
| 폰트 README | [`src/reports/resources/fonts/README.md`](src/reports/resources/fonts/README.md) | ✅ 다운로드 안내 (TTF 자체는 별도 배치 필요) |

**설정/개발 환경 파일:**

| 항목 | 현재 상태 |
|---|---|
| `.gitignore` | ❌ **없음** — 권장 설정은 §11.3 참조 |
| `.editorconfig` | ❌ 없음 |
| `pyproject.toml` / `setup.py` | ❌ 없음 — 단일 스크립트 기반 앱이라 불필요 |
| `.python-version` | ❌ 없음 — **Python 3.11.x 명시 필요** (→ §5.2) |
| pre-commit hooks | ❌ 없음 |

**미비 항목은 인수 팀이 필요 시 추가**. 현재까지 내부 운영만 했으므로 최소 구성.

---

## 3. 함께 받으면 좋은 것

### 3.1 샘플 세션 (하드웨어 없이 분석·리플레이 검증용)

현재 `data/sessions/`에는 **58개 세션** (38 balance, 11 cmj, 7 free_exercise, 1 squat, 1 wba). 전부 피험자 PII(이름·ID) 포함 — 인수 시 **익명화 후 5~8개만 선별 권장**.

**추천 선별 기준:**

| Test | 권장 샘플 | 목적 |
|---|---|---|
| `balance_eo_two_*` | 최신 1개 | 눈 뜨고 양발 밸런스 기본 |
| `balance_eo_left_*` | 최신 1개 | 좌측 밸런스 (CoP L/R 검증) |
| `balance_ec_*` | 최신 1개 | 눈 감고 밸런스 |
| `cmj_*` | 최신 2개 | CMJ 점프 + flight 구간 |
| `squat_*` | 최신 1개 | 스쿼트 rep 검출 |
| `free_exercise_*` | 최신 1개 | 자유 운동 + 엔코더 rep |

**익명화 절차:**
1. 세션 폴더의 `session.json`에서 `subject_id`/`subject_name` 마스킹
2. 파일명의 피험자 ID 제거: `cmj_20260423_151233_67909427` → `cmj_20260423_151233_SUBJ01`
3. `forces.csv`에 PII 없음 (수치 데이터만) — 그대로 OK
4. `video_*.mp4`가 얼굴 인식 가능하면 프라이버시 검토 (해당 시 제외)

### 3.2 CI 설정
❌ 없음. 인수 후 GitHub Actions 기본 셋업 권장:
- `pytest` on push (스모크 테스트)
- `flake8` or `ruff` 린트
- 한글 문서 링크 체크

### 3.3 테스트 코드
- [`scripts/tests/smoke_test_analysis.py`](scripts/tests/smoke_test_analysis.py) — 분석 파이프라인 스모크만 있음. 정식 pytest 구조 **없음**. 인수 후 `tests/` 디렉토리 신설 권장.

---

## 4. 제외 권장

| 항목 | 이유 |
|---|---|
| `data/sessions/` 전체 (선별본 제외) | 피험자 PII, 수 GB |
| `data/calibration/session_*` 원본 녹화 | 카메라 캘리 raw (4-5 GB) — 결과 NPZ만 있으면 재실행 가능 |
| `__pycache__/`, `*.pyc` | 빌드 아티팩트 |
| `venv/`, `.venv/` (만약 있다면) | 가상환경 — `requirements.txt`로 재생성 |
| NI 드라이버 설치 파일 | ni.com에서 최신 다운로드 |
| `config.py.bak` | 자동 백업 (최근 캘리 전 버전) — 필요 시만 포함 |
| `_phase_q4_sample_cmj_trainer.pdf` | PDF 검증용 샘플 (data/calibration/) — 인수와 무관 |

---

## 5. 추가 정보 — 질의 응답

### 5.1 NI USB-6210 장치 이름

**`Dev1`이 맞습니다.** [`config.py:101`](config.py:101) 기본값과 일치.

**확인 방법:**
1. NI MAX (Measurement & Automation Explorer) 실행
2. `Devices and Interfaces` → `NI USB-6210 "Dev1"` 항목 존재 확인
3. 이름이 다르면 NI MAX에서 우클릭 → `Rename...`으로 `Dev1` 통일하거나
4. `config.DAQ_DEVICE_NAME`을 실제 이름으로 변경

**Python으로 즉석 검증:**
```python
from src.capture.daq_reader import DaqReader
r = DaqReader()
assert r.connect(), "DAQ not found — check NI MAX for 'Dev1'"
```

### 5.2 Python 버전

**Python 3.11.9** (현재 `python --version` 기준).

**호환 범위:**
- 권장: **Python 3.11.x** (개발·운영 전반 사용)
- 최소: Python 3.10 (`typing.Optional` 등은 3.10에 도입된 `A | B` 문법 사용 다수)
- ⚠️ Python 3.12+ 미검증 (`mediapipe<0.11` 호환성 주의)

`.python-version` 파일 권장 내용:
```
3.11
```

pyenv 또는 conda에서 동일 버전 지정 가능.

### 5.3 OS 레벨 NI 드라이버

**필요 드라이버:** **NI-DAQmx** (Python 패키지 `nidaqmx`는 이 드라이버 래퍼)

현재 설치된 버전 확인 방법:
```powershell
# PowerShell
Get-ItemProperty HKLM:\SOFTWARE\National\ Instruments\NI-DAQ\*CurrentVersion* -ErrorAction SilentlyContinue
```

또는 NI MAX 좌측 트리에서 `My System > Software > NI-DAQmx` 버전 확인.

**권장 최소 버전:** NI-DAQmx **20.0 이상** (USB-6210 지원은 훨씬 이전부터 있었으나 Python 3.11 호환은 최신 드라이버가 안전).

**설치 URL:** https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html

### 5.4 카메라 캘리브레이션

**위치:** [`data/calibration/`](data/calibration/)

현재 존재하는 카메라 캘리 산출물:
```
intrinsics_C0.npz          # 1680 B
intrinsics_C1.npz          # 1698 B
intrinsics_C2.npz          # 1698 B
extrinsics_unscaled.npz    # 4010 B
poses3d_session_*.npz      # 삼각측량 결과 (3-cam 시절)
poses3d_world_session_*.npz
world_frame.npz            # 월드 프레임 정합 (6.4 KB)
session_20260421_213656/   # 캘리 원본 녹화 세션
session_20260421_215241/
```

⚠️ **주의:** 위 intrinsics_C0/C1/C2는 **3-camera 시스템 시절 산출물** (2026-04-21). 현재는 단일 카메라(C0)만 사용하는 구성으로 전환됨 (2026-04-22). C1/C2 파일은 **역사적 기록용**이며 재활용 불가.

단일 카메라 시스템에서는 `intrinsics_C0.npz`만 유효. 추가 멀티캠 확장 시 재캘리 필수.

자세한 워크플로우: [`docs/CALIBRATION_WORKFLOW.md`](docs/CALIBRATION_WORKFLOW.md)

### 5.5 라이선스

**현재 라이선스 명시 안 됨** — 최상위에 `LICENSE` 파일 없음.

**해석:** 명시 부재 시 기본은 **© FITWIN (또는 개발 주체) 사내 자산, all rights reserved**. 오픈소스 아님.

**인수 시 반드시 합의 필요:**
- 사내 자산 (계속 closed-source)인지
- 오픈소스화할지 (MIT/Apache/GPL 중 택)
- 제3자 라이브러리 라이선스 상충 여부 (주 의존성 중 `mediapipe`=Apache 2.0, `PyQt6`=GPL v3/상업, `reportlab`=BSD/상업)

**PyQt6 주의:** GPL v3 하위 자동 의존 — 상업 재배포 시 **상업 라이선스 구매 필요** (Riverbank Computing). 현재는 사내 운영이라 문제 없음.

---

## 6. 코드 구조 워크스루

### 6.1 디렉토리 구조 및 역할

```
biomech-mocap/
├── main.py                       # PyQt6 앱 진입점
├── app.py                        # (레거시? main.py와 관계 확인 필요)
├── config.py                     # ★ 모든 상수 (하드웨어/경로/테스트/브랜딩)
├── requirements.txt
│
├── src/
│   ├── capture/                  # DAQ/카메라 실시간 데이터 취득
│   │   ├── daq_reader.py         # ★ NI USB-6210 스레드 리더 + DaqFrame + CoP 계산
│   │   ├── session_recorder.py   # ★ 녹화 오케스트레이션 (DAQ+cam 동기)
│   │   └── wait_for_stance.py    # smart-wait 안정 감지
│   │
│   ├── analysis/                 # 오프라인 분석
│   │   ├── common.py             # ★ forces.csv → ForceSession 로더
│   │   ├── dispatcher.py         # ★ test_type → analyzer 라우팅
│   │   ├── balance.py            # 밸런스 (CoP 스웨이)
│   │   ├── cmj.py                # CMJ 점프
│   │   ├── squat.py              # 스쿼트
│   │   ├── encoder.py            # 엔코더 rep + RealtimeRepCounter
│   │   ├── free_exercise.py      # 자유 운동 (엔코더 wrapper)
│   │   ├── reaction.py           # 반응시간
│   │   ├── proprio.py            # 고유감각
│   │   ├── pose2d.py             # 2D 포즈 시리즈
│   │   └── excel_export.py       # xlsx 다시트 내보내기
│   │
│   ├── calibration/              # 카메라/월드 프레임 캘리
│   │
│   ├── ui/
│   │   ├── app_window.py         # ★ 메인 윈도우 (탭 컨테이너)
│   │   ├── tabs/
│   │   │   ├── subjects_tab.py   # 피험자 CRUD
│   │   │   ├── measure_tab.py    # ★ 측정 (카메라+대시보드+컨트롤)
│   │   │   └── reports_tab.py    # ★ 리포트 (browse + replay 스택)
│   │   ├── widgets/
│   │   │   ├── force_dashboard.py
│   │   │   ├── force_widgets.py      # VGRFPlot, COPTrajectory
│   │   │   ├── encoder_bar.py
│   │   │   ├── encoder_timeline.py
│   │   │   ├── replay_panel.py       # ★ 전용 리플레이 페이지
│   │   │   ├── report_viewer.py
│   │   │   ├── test_options_panel.py
│   │   │   ├── stability_overlay.py
│   │   │   ├── video_player.py
│   │   │   ├── force_timeline.py
│   │   │   ├── cop_trail.py
│   │   │   ├── joint_coord_trail.py
│   │   │   ├── angle_timeline.py
│   │   │   ├── protocol_header.py
│   │   │   ├── sidebar_toggle.py
│   │   │   ├── camera_view.py
│   │   │   ├── playback_controller.py
│   │   │   └── replay_colors.py
│   │   └── workers/
│   │       ├── record_worker.py      # QThread wrapping SessionRecorder
│   │       ├── analysis_worker.py    # 오프라인 분석 비동기
│   │       ├── pose_worker.py        # MediaPipe 포즈 처리
│   │       ├── pose_live_worker.py   # 실시간 포즈 오버레이
│   │       ├── excel_export_worker.py
│   │       └── pdf_export_worker.py
│   │
│   ├── reports/                  # PDF/HTML 리포트 시스템
│   │   ├── base.py               # ReportContext, ReportSection 추상
│   │   ├── report_builder.py     # ★ 트레이너용/피험자용 섹션 조립
│   │   ├── pdf_renderer.py       # reportlab NumberedCanvas + 브랜드 프레임
│   │   ├── html_renderer.py      # HTML 변환
│   │   ├── charts.py             # matplotlib 차트 생성
│   │   ├── key_metrics.py        # KEY_METRICS 딕셔너리 (지표 레이블)
│   │   ├── norms.py              # 권고 범위 + 상태 분류
│   │   ├── fonts.py              # 한글 폰트 resolver
│   │   ├── palette.py            # 색상 상수
│   │   ├── sections/             # 섹션별 구현
│   │   │   ├── common.py         # Header/SummaryCards/Footer/Notes
│   │   │   ├── cover.py          # (Q4b) 커버 페이지
│   │   │   ├── verdict.py        # (Q4e) 요약 배너
│   │   │   ├── glossary.py       # (Q4h) 피험자용 용어
│   │   │   ├── detail.py
│   │   │   ├── history.py
│   │   │   ├── pose_angles.py
│   │   │   ├── balance.py / cmj.py / squat.py / encoder.py / reaction.py / proprio.py
│   │   └── resources/
│   │       └── fonts/README.md   # 폰트 번들 가이드
│   │
│   ├── db/                       # SQLite (피험자/세션 메타)
│   ├── io/                       # TDMS 등 파일 포맷
│   └── pose/                     # 포즈 관련 유틸
│
├── scripts/                      # CLI 도구 (각각 독립 실행)
│   ├── calibrate_daq_scale.py    # ★ 포스 스케일 다지점 회귀 캘리
│   ├── verify_encoders.py        # ★ 엔코더 스케일 자동 검증
│   ├── calibrate_intrinsics.py / calibrate_from_poses.py / align_world_from_force.py
│   ├── record_session.py / record_calibration_session.py
│   ├── analyze.py                # 세션 일괄 재분석
│   ├── diagnose_session.py / diagnose_calibration.py
│   ├── recompute_cop.py / fix_forces_csv_units.py  # 데이터 마이그레이션
│   ├── backfill_session_metrics.py
│   ├── generate_charuco_a4.py / measure_camera_fps.py
│   ├── process_pose_for_session.py
│   └── tests/smoke_test_analysis.py
│
├── docs/
│   ├── FORCE_PLATE_SYSTEM_SPEC.md  # ★ 하드웨어 정본
│   ├── CALIBRATION_WORKFLOW.md     # 카메라/월드 캘리
│   └── CHARUCO_PRINT_GUIDE.md      # ChArUco 인쇄 가이드
│
├── data/
│   ├── sessions/     # 세션별 녹화 (forces.csv, *.mp4, session.json, result.json)
│   └── calibration/  # 카메라/엔코더/포스 캘리 산출물
│
└── resources/
    ├── charuco_board.png
    └── mediapipe/    # BlazePose .task 파일 (auto-downloaded)
```

### 6.2 데이터 흐름 (capture → analysis → UI)

```
┌─────────────────────────────────────────────────────────────┐
│  실시간 녹화 (MeasureTab)                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [사용자] ── "측정 시작" 클릭 (F5)                         │
│     │                                                       │
│     ▼                                                       │
│  MeasureTab._start_next_in_queue                           │
│     ├─ RecorderConfig 생성 (test/options/subject_kg)        │
│     │     └─ __post_init__: use_bodyweight 적용            │
│     └─ RecordWorker(cfg) 시작 (QThread)                     │
│           │                                                 │
│           ▼                                                 │
│    SessionRecorder                                          │
│       ├─ DaqReader.start() → 100 Hz 스트림                 │
│       │     └─ per-frame: subtract zero-cal,               │
│       │                   moving-average, V→N/mm           │
│       │                   cop_world_mm() 계산              │
│       ├─ MultiCameraCapture (mp4 저장)                      │
│       ├─ StabilityDetector (smart-wait 관찰)               │
│       └─ Signal emit: daq_frame / camera_frame / state_changed  │
│                                                             │
│  [GUI 스레드]                                                │
│     ├─ ForceDashboard.on_daq_frame(fr)                     │
│     │     └─ VGRFPlot.push + COPTrajectory.push            │
│     │     └─ phase-aware suppression (녹화 중이 아니면 <98N 차단)│
│     ├─ EncoderBar.set_value(fr.enc1_mm)                    │
│     ├─ RealtimeRepCounter.push (free_exercise)             │
│     ├─ StabilityOverlay.update_from_state(st)              │
│     └─ CameraView.on_camera_frame(frame)                   │
│                                                             │
│  ★ 녹화 완료 시 ──────────────────────────────────────▶    │
│     session_dir/                                            │
│       ├─ forces.csv            (100 Hz, 15 컬럼)            │
│       ├─ C0.mp4                (영상)                       │
│       ├─ session.json          (메타)                       │
│       └─ stimulus_log.csv      (reaction 시 only)           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (auto-trigger)
┌─────────────────────────────────────────────────────────────┐
│  분석 (AnalysisWorker)                                      │
├─────────────────────────────────────────────────────────────┤
│  dispatcher.analyze_session(session_dir)                    │
│     ├─ session.json에서 test_type 읽음                       │
│     ├─ _ANALYZERS[test_type] 호출                            │
│     │   │                                                   │
│     │   ├─ common.load_force_session(sd)                    │
│     │   │    └─ forces.csv → ForceSession (numpy arrays)    │
│     │   ├─ 테스트별 분석:                                    │
│     │   │    balance: CoP sway, 95% ellipse, path length    │
│     │   │    cmj:     flight detection, jump height, power  │
│     │   │    squat:   rep detection, WBA, depth             │
│     │   │    encoder: detect_reps + per-rep velocity/power  │
│     │   │    free_ex: wrapper of encoder + exercise name    │
│     │   │    reaction: RT distribution, recovery time       │
│     │   │    proprio: target vs actual                      │
│     │   └─ Optional: PoseWorker → pose2d.py → 관절 각도      │
│     │                                                        │
│     └─ result.json 저장 (session_dir/)                      │
│                                                              │
│  DB: sessions 테이블 status="analyzed" 갱신                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  리포트 (ReportsTab browse → replay)                        │
├─────────────────────────────────────────────────────────────┤
│  [Browse 모드]                                              │
│   ├─ DB list_sessions → 테이블 렌더                         │
│   ├─ 세션 선택:                                             │
│   │   └─ ReportViewer.show_session                          │
│   │       └─ report_builder.build_trainer/subject_report    │
│   │           └─ list[ReportSection] (cover+verdict+...)    │
│   │       └─ html_renderer.render_html (embedded view)      │
│   │                                                          │
│   ├─ 액션 버튼 (우측 상단):                                  │
│   │   ├─ 🔍 재분석       → AnalysisWorker 재실행            │
│   │   ├─ 🎯 2D 포즈     → PoseWorker + re-analysis          │
│   │   ├─ 📊 Excel       → ExcelExportWorker (xlsx)          │
│   │   ├─ 📄 PDF         → PdfExportWorker                   │
│   │   │                   └─ pdf_renderer.render_pdf        │
│   │   │                       └─ NumberedCanvas + sections  │
│   │   ├─ ▶ 리플레이      → QStackedWidget page 1            │
│   │   └─ 📁 세션 폴더   → 파일 탐색기 open                  │
│   │                                                          │
│  [Replay 모드 — 독립 페이지]                                 │
│   └─ ReplayPanel.load_session                                │
│       ├─ VideoPlayerWidget.load(mp4)                         │
│       ├─ ForceTimelineWidget.load(forces.csv)                │
│       ├─ EncoderTimelineWidget.load                          │
│       ├─ CopTrailWidget.load                                 │
│       ├─ AngleTimelineStack.load (pose가 있으면)             │
│       └─ PlaybackController → 모든 위젯 시간 동기화          │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 주요 클래스·진입점 map

| 기능 | 파일 | 클래스/함수 |
|---|---|---|
| 앱 진입 | `main.py` | (module-level: QApplication + AppWindow) |
| 메인 윈도우 | `src/ui/app_window.py` | `AppWindow` |
| 실시간 DAQ | `src/capture/daq_reader.py` | `DaqReader`, `DaqFrame` |
| 녹화 오케스트레이션 | `src/capture/session_recorder.py` | `SessionRecorder`, `RecorderConfig`, `RecorderState` |
| 분석 라우팅 | `src/analysis/dispatcher.py` | `analyze_session`, `_ANALYZERS` |
| 실시간 rep 카운트 | `src/analysis/encoder.py` | `RealtimeRepCounter` |
| PDF 렌더 | `src/reports/pdf_renderer.py` | `render_pdf`, `NumberedCanvas` |
| 리포트 조립 | `src/reports/report_builder.py` | `build_trainer_report`, `build_subject_report` |

---

## 7. 알려진 함정 / 주의사항

### 7.1 Zero-cal 타이밍
- DAQ 스트림 시작 후 **첫 5초**는 zero-cal 구간. 피험자가 **플레이트 밖**에 있어야 함.
- 플레이트 위에서 시작하면 해당 하중이 offset으로 빼지므로 **실제 측정치 전부 틀어짐**.
- 앱은 "플레이트에서 내려와 계세요" 프롬프트를 띄우지만 **강제 차단은 안 함** — 트레이너 교육 필요.
- 엔코더는 zero-cal 미적용 (절대 위치값이라).

### 7.2 Encoder 2 비활성 처리
- 우측 엔코더는 **되감기 기구 고장** (2026-04 기준).
- `config.ENCODER2_AVAILABLE = False` → UI와 분석기 모두 "비활성" 처리.
- **수리 후 재활성 절차:**
  1. `config.ENCODER2_AVAILABLE = True`로 변경
  2. `python scripts/verify_encoders.py --auto --channel 2` 실행
  3. VERDICT "calibration looks correct" 확인
  4. 분석기에 `use_encoder=2` 전달 가능해짐

### 7.3 Side-by-side 좌표계 혼동
- **World frame: 원점 = 통합 플레이트 좌하단, +X 오른쪽, +Y 앞쪽.**
- Board1 = LEFT, Board2 = RIGHT (피험자가 위에서 본 시점).
- **AI 채널 번호는 물리 배선 순서 그대로**: Board1은 ai6-4 (역순!), Board2는 ai0-3 (BL/BR 역순!) — 재배선 금지.
- CoP 시각화에서 **y축이 위가 전방**. pyqtgraph 기본 좌표계와 일관.

### 7.4 CoP 축 자동 확장 문제 (해결됨 — Phase I1)
- pyqtgraph가 기본적으로 autoRange 활성 → 노이즈 샘플 하나로 뷰포트가 `[-4000, 4000]`까지 확장.
- Board outline이 구석에 뭉치고 **L/R 반전처럼** 보임 (실제 반전 아님).
- 해결: `setLimits` + `enableAutoRange(False)`로 하드 클램프.
- 향후 CoP 위젯 수정 시 이 제약 유지 필수.

### 7.5 Phase-aware 노이즈 억제 (Phase I3)
- `RecorderState.phase != "recording"` 일 때 `< 98 N` 신호는 0으로 클램프.
- 점프 flight(20-30 N)는 녹화 중이라 억제 안 됨 — 의도된 동작.
- 만약 새 테스트 타입이 이 threshold 로직과 충돌하면 `ForceDashboard.set_recording(False)` 조건 재검토.

### 7.6 Force scale per-corner
- 8-element 리스트 (Board1 4개, Board2 4개). 스칼라 하나로 퇴화 가능하나 **보드별 제조공차** 반영을 위해 분리 유지.
- 캘리 스크립트 `calibrate_daq_scale.py`가 자동으로 per-corner 회귀 — 결과를 `.bak` 저장 후 in-place 업데이트.

### 7.7 Encoder scale 실측 필수
- 스펙시트 값 맹신 금지 — 초기 `200 mm/V` 가정이 실측 결과 `784.23 mm/V`로 4× 차이.
- 센서 교체/재배선 시 반드시 `verify_encoders.py` 재실행.

### 7.8 카메라 회전 + 미러
- `CAMERA_ROTATION=90` + `CAMERA_MIRROR=True`. 캡처 서브프로세스 내부에서 적용 → mp4/실시간/리플레이 **전부 동일 프레임** 보장.
- 재장착 시 config만 바꾸면 되도록 추상화됨.
- 주의: CAMERA_ROTATION이 90/270이면 저장 mp4의 width/height가 swap됨 (portrait).

### 7.9 Reports "encoder" 테스트 레거시
- `encoder` 테스트는 **UI 드롭다운에서 제거됨** (G2). 그러나:
  - `TEST_KO["encoder"]` 레이블 유지 (과거 세션 표시)
  - `dispatcher._ANALYZERS["encoder"]` 등록 유지 (과거 분석 호환)
  - 신규 측정은 `free_exercise`로 대체
- 절대 `_ANALYZERS`에서 제거하지 말 것 — 기존 세션 재분석 막힘.

### 7.10 PyQt6 shortcut scope
- `QShortcut(key, self, context=WidgetWithChildrenShortcut)` — 탭이 활성일 때만 작동.
- 전역 scope 안 사용. 다른 탭 입력과 충돌 방지.

### 7.11 한글 폰트 폴백 체인
- 1순위: `src/reports/resources/fonts/NanumGothic-*.ttf` (없음)
- 2순위: Windows 시스템 NanumGothic
- 3순위: Windows 기본 Malgun Gothic
- 4순위: Helvetica (한글 `□` 표시)
- 대부분 PC는 3번으로 정상 동작하지만 상용 배포 시 1번 번들 권장.

---

## 8. 미해결 버그 / TODO

현재 코드베이스에 `TODO/FIXME/XXX` 검색 결과:

| 위치 | 내용 |
|---|---|
| `src/ui/main_window.py:225` | "TODO: wire to src.io.recorder once implemented" (카메라 미리보기, 레거시 main_window 관련 — main.py가 app_window 사용하므로 사실상 사문화) |

**비문서화된 알려진 이슈:**

1. **`data/sessions/wba_*` 레거시 폴더** — 과거 WBA 전용 분석 시절. test_type="wba"는 현재 dispatcher에 등록 없음. 재분석 시 에러 발생 가능 → 무시 or 삭제.

2. **`balance_20260422_001752` 등 suffix 없는 balance 세션** — G3 이전 `balance_eo/ec` 분화 전 산물. 상태 `test_type="balance"` (정확한 vision/stance 없음) → 분석기가 fallback.

3. **3-camera 시절 poses3d_*.npz** — 현재 1-camera 시스템과 호환 안 됨. `data/calibration/` 내 동명 파일은 역사 기록용.

4. **PDF 생성 시 한글 깨짐 가능성** — Malgun Gothic 없는 환경(macOS/Linux) 배포 시 Helvetica 폴백으로 한글 `□`. 번들 TTF 해결 필요.

---

## 9. 향후 통합 계획 — 영향받을 모듈

### 9.1 OpenSim 통합 (3D 생체역학 시뮬레이션)

OpenSim 파이프라인 연동 시 영향받는 모듈:

| 모듈 | 예상 변경 |
|---|---|
| `src/io/` (신규 파일) | `opensim_export.py` — forces.csv + pose2d → `.mot` + `.trc` 변환 |
| `src/analysis/pose2d.py` | 3D 재구성이 필요하면 pose3d.py 추가 (멀티캠 복귀 or depth camera 필요) |
| `config.py` | 골격 모델 경로, OpenSim GUI/CLI 경로 상수 추가 |
| `scripts/export_opensim.py` | 세션 하나를 OpenSim 프로젝트로 export |
| `src/reports/sections/` | 관절 모멘트/파워 섹션 추가 (OpenSim 결과 활용) |

**주의점:**
- 단일 카메라만으로는 3D 관절 위치 추정 불가 — MediaPipe 3D Landmarks는 개략값. OpenSim 정밀 IK에는 부족.
- Orbbec (아래) 또는 멀티캠 복귀 선행 필요.

### 9.2 Orbbec (ToF/Stereo depth camera) 통합

Orbbec Femto/Gemini 같은 depth 카메라로 단일-시점 3D 관절 취득 확장:

| 모듈 | 예상 변경 |
|---|---|
| `src/capture/` (신규) | `orbbec_reader.py` — Orbbec SDK (pyorbbecsdk) 래핑 |
| `src/capture/session_recorder.py` | 카메라 소스 추상화 (Logitech webcam / Orbbec 전환) |
| `config.py` | `CAMERA_TYPE = "orbbec" \| "webcam"` + 해상도 설정 |
| `src/analysis/` | 3D pose 직접 취득 → triangulation/BA 스킵 가능 |
| `src/pose/` | Orbbec body tracking SDK 결과 ingestion |
| `resources/` | Orbbec 모델·런타임 파일 캐시 |
| `src/ui/widgets/camera_view.py` | depth visualization 모드 (선택적) |
| `requirements.txt` | `pyorbbecsdk` (Windows x64 wheels 확인 필요) |

**선 검증 필요:**
- Orbbec pyorbbecsdk가 Python 3.11 wheels 제공하는지 (현재 Python 3.8~3.10 권장이 많음)
- SDK 라이선스 (상업 재배포 가능 여부)
- 포스 플레이트와의 타임 싱크 — USB jitter 보정 로직 설계

### 9.3 원격 측정 서비스 (SaaS 방향)

클리닉 다지점 배포 + 중앙 대시보드 고려 시:

| 요소 | 변경 필요 |
|---|---|
| DB | SQLite → PostgreSQL (centralized) |
| 인증 | `src/db/auth.py` 추가 + 로그인 UI |
| 동기화 | 세션 녹화 → S3/MinIO 업로드 |
| 리포트 | Web 뷰 추가 (Next.js 등) |
| 코드 격리 | `src/core/` (하드웨어/분석) vs `src/cloud/` (업로드) 분리 |

향후 차세대 시스템 이 방향이면 **현재 app.py/main.py 기반 단일 binary** 구조를 모듈화해야 함.

---

## 10. 인수인계 체크리스트

인수 팀이 **단계적으로** 확인·실행할 절차:

### 10.1 코드 수령 직후 (Day 0-1)

- [ ] 본 문서(HANDOVER.md) + FORCE_PLATE_SYSTEM_SPEC.md + CALIBRATION_WORKFLOW.md 통독
- [ ] Python 3.11.x 환경 구성 (pyenv/conda)
- [ ] `pip install -r requirements.txt` 성공
- [ ] `git init` + 초기 커밋 (§1.2)
- [ ] NI-DAQmx 드라이버 설치 확인

### 10.2 하드웨어 없는 검증 (Day 1-2)

- [ ] `python main.py` 실행 — 앱 GUI 뜨는지
- [ ] Reports 탭 → 샘플 세션 한 개 선택 → HTML 리포트 표시 확인
- [ ] PDF 내보내기 → `data/sessions/<sample>/*.pdf` 생성 확인
- [ ] Replay 버튼 → 영상+그래프 재생 확인
- [ ] `python scripts/tests/smoke_test_analysis.py` 통과

### 10.3 하드웨어 연결 검증 (Day 2-3)

- [ ] DAQ 연결: `from src.capture.daq_reader import DaqReader; DaqReader().connect()` → True
- [ ] Zero-cal 작동: 측정 시작 → "플레이트에서 내려와..." 프롬프트 표시
- [ ] balance 1회 녹화 → CoP 궤적이 왼쪽/오른쪽 예상대로 표시 (§7.3 참고)
- [ ] CMJ 1회 녹화 → flight 구간 VGRF가 근처 0N까지 떨어짐 (억제 안 됨)
- [ ] free_exercise + 엔코더 → 실시간 rep 카운트 증가

### 10.4 캘리 재검증 (Day 3-5)

- [ ] 현재 캘리 값(2026-04 기준)이 여전히 유효한지 검증:
  - 기준 무게 20kg로 양 보드 → 리드아웃이 ±0.5 kg 내
  - 엔코더 1m 스탠드 지점에서 측정 → ±5mm 내
- [ ] 오차 크면 §5 (force) / §5.3 (encoder) 재캘리 절차 실행

### 10.5 개발 환경 셋업 (Day 5-7)

- [ ] IDE/에디터 설정 (VS Code + Python/Ruff 확장 권장)
- [ ] pre-commit 훅 도입 (ruff, black)
- [ ] `tests/` 디렉토리 신설 → 기본 pytest 스모크 이관
- [ ] CI (GitHub Actions) 설정 — push 시 lint + tests

### 10.6 인수 완료 확인 (Day 7)

- [ ] 핵심 기능(측정 → 분석 → 리포트 → 리플레이) 전 과정 E2E 수행
- [ ] 모든 테스트 타입(balance_eo/ec, cmj, squat, free_exercise, reaction, proprio) 각 1회 녹화·분석 성공
- [ ] 미해결 이슈 리스트(본 문서 §8) 리뷰 + 우선순위 조정
- [ ] 향후 로드맵(§9) 합의

---

## 11. 인수인계 패키지 빌드 절차

### 11.1 수동 zip (권장)

PowerShell에서 실행:

```powershell
# 프로젝트 루트에서
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$out = "..\biomech-mocap-handover-$ts.zip"
$exclude = @(
    "venv", ".venv", "__pycache__", "*.pyc",
    "data\sessions",          # PII 제외 — 샘플은 별도 수동 복사
    "config.py.bak"           # 선택적
)
# PowerShell Compress-Archive는 exclude 제한적 — robocopy로 임시 복사 후 압축 권장
$staging = "..\biomech-mocap-staging"
robocopy . $staging /E /XD venv .venv __pycache__ "data\sessions" "data\calibration\session_*" /XF "*.pyc" "config.py.bak"
# 샘플 세션 5개 선별 복사
mkdir "$staging\data\sessions" -Force | Out-Null
$samples = @(
    "balance_eo_two_20260422_224542_67909427",
    "balance_ec_20260422_175433_67909427",
    "cmj_20260423_151233_67909427",
    "squat_20260422_012329",
    "free_exercise_20260423_182527_67909427"
)
foreach ($s in $samples) {
    $src = "data\sessions\$s"
    if (Test-Path $src) {
        robocopy $src "$staging\$src" /E
    }
}
Compress-Archive -Path "$staging\*" -DestinationPath $out -Force
Remove-Item $staging -Recurse -Force
Write-Host "Package: $(Resolve-Path $out)"
```

**용량 예상:** ~100-300 MB (mediapipe 모델 캐시 포함).

### 11.2 전달 시 함께 첨부

1. **zip 파일** (위 procedure 결과)
2. **SHA-256 체크섬** — 전송 무결성 검증용
   ```powershell
   Get-FileHash $out -Algorithm SHA256 | Format-List
   ```
3. **본 HANDOVER.md** (zip 외부에도 이메일 첨부 권장 — 받자마자 먼저 읽을 수 있게)

### 11.3 인수 팀용 `.gitignore` 제안

수령 후 git init 하기 전 [`.gitignore`](#) 생성 권장:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
build/
dist/

# Virtualenv
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# App-specific: runtime data
data/sessions/*/*.mp4           # 영상은 커밋 금지
data/sessions/*/C*.timestamps.csv
data/calibration/session_*/     # 카메라 캘리 원본 녹화
config.py.bak

# PDF samples, test artifacts
*.pdf
_phase_*.py                     # 임시 스모크 테스트
```

샘플 세션은 `data/sessions/*/forces.csv`, `session.json`, `result.json`만 커밋 — 영상은 별도 스토리지 권장.

---

## 12. 연락처 / 이력

- **현 운영 주체:** FITWIN MoCap Clinic
- **최초 개발 기간:** 2026-04-21 ~ 2026-04-23 (집중 구축)
- **본 인수인계 시점:** 2026-04-23
- **문서 버전:** 1.0

**질문·불명료한 점은 본 문서를 우선 참고하고, 해결 안 되면:**
- `docs/FORCE_PLATE_SYSTEM_SPEC.md` 상세 스펙
- `git log` (인수 후 git init 시점부터)
- 원 개발 주체 직접 문의

---

**끝.**
