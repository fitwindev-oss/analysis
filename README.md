# Biomech MoCap — 3-Camera + Force Plate System

**3D markerless motion capture + 지면반력 통합 분석 시스템**

- **카메라**: 3대 (LifeCam Studio / Samsung PC-CAM / Logitech StreamCam) @ 640×480 30 fps
- **지면반력**: NI USB-6210 DAQ로 8채널 로드셀 + 2채널 선형 엔코더 (@100 Hz)
- **GPU**: NVIDIA GTX 960 4GB (RTMPose-s/m 실시간 추론 가능)
- **좌표계**: 포스 플레이트 좌하단 = (0, 0), 우상단 = (558, 432) mm

---

## Project Layout

```
biomech-mocap/
├── main.py                   ← PyQt6 앱 진입점
├── config.py                 ← 모든 튜닝 파라미터 (플레이트 치수, DAQ 채널, 모델)
├── requirements.txt
├── scripts/
│   └── generate_charuco_a4.py      ← A4 ChArUco 보드 생성
├── docs/
│   └── CHARUCO_PRINT_GUIDE.md      ← 출력/부착/보정 가이드
├── src/
│   ├── ui/
│   │   ├── main_window.py          ← 4-column top + tabbed bottom
│   │   └── widgets/
│   │       ├── camera_tile.py      ← 카메라 프리뷰 (2D keypoint overlay 지원)
│   │       ├── skeleton3d.py       ← 3D 스켈레톤 뷰어 (OpenGL)
│   │       └── force_widgets.py    ← vGRF 플롯, CoP 궤적, 엔코더 바
│   ├── capture/
│   │   ├── camera_worker.py        ← 3-cam 동기 멀티프로세스 캡처
│   │   └── daq_reader.py           ← NI USB-6210 실시간 리더
│   ├── io/
│   │   └── tdms_reader.py          ← 기존 TDMS 파일 오프라인 로더
│   ├── calibration/                ← (Phase 2) intrinsic/extrinsic/world-frame
│   ├── pose/                       ← (Phase 3) 2D 검출 + 삼각측량 + 필터
│   └── analysis/                   ← (Phase 4) 분석 모듈들
└── data/
    ├── calibration/                ← .npz intrinsics, extrinsics 저장
    └── sessions/                   ← 녹화된 .mp4×3 + .tdms + 메타
```

---

## Quick Start

### 1) 의존성 설치

```bash
cd C:\Users\FITWIN\Desktop\biomech-mocap
pip install -r requirements.txt
```

> **참고**: `onnxruntime-gpu`는 CUDA 11.8을 기본 사용합니다. 문제 발생 시 `onnxruntime` (CPU판)으로 대체 가능 — `config.POSE_DEVICE = "cpu"`로 전환.

### 2) ChArUco 보드 생성 → 출력 → 부착

```bash
python scripts/generate_charuco_a4.py
```

`resources/charuco_board.png` 가 생성됩니다. **docs/CHARUCO_PRINT_GUIDE.md** 의 지침대로:
- A4 100% 배율로 출력 (반드시 "fit to page" **끄기**)
- 자로 한 칸(30 mm) 실측 확인
- 평평한 판(폼보드/디본드)에 부착 후 무광 라미네이션
- 캘리브레이션 시 **보드 좌하단 = 포스 플레이트 좌하단 (0,0)** 에 정확히 맞춤

### 3) 앱 실행 (Phase 1 현재 상태)

```bash
python main.py
```

**현재 가동 기능**:
- 4-column 상단 레이아웃 (3 카메라 + 3D 스켈레톤 placeholder)
- `Start Preview` → 3대 카메라 라이브 표시
- 3D 뷰어에 포스 플레이트 Footprint + 세계좌표축 렌더
- Live/Calibration/Analysis/Playback 탭 골격
- TDMS 파일 로드 다이얼로그 (파싱만, 아직 playback 시각화 미연결)

**아직 미연결** (Phase 2 이후 작업):
- ChArUco 캘리브레이션 스크립트
- 2D → 3D 포즈 추정 파이프라인
- DAQ + 카메라 동기 녹화 통합
- 분석 모듈 (CMJ, WBA, 균형, 스쿼트)

---

## 두 가지 실행 모드

### Online (실시간)
- 카메라 3대 + DAQ 동시 구동
- 화면에 실시간 2D 오버레이 + 3D 스켈레톤 + vGRF/CoP
- GTX 960에서 RTMPose-s @ 3 cams ≈ 15~25 Hz 예상 (해상도/모델에 따라)

### Offline (배치 분석)
- 녹화된 `session_*/C0.mp4, C1.mp4, C2.mp4` + `.tdms` 를 입력
- 더 무거운 RTMPose-l, bundle adjustment, Pose2Sim 파이프라인 사용 가능
- 정확도 우선, 연구용 결과 생성

각 모드는 **같은 파일 포맷**을 공유합니다 (`.mp4`×3 + `.tdms` + `timestamps.csv`×3).

---

## 개발 로드맵

| Phase | 내용 | 상태 |
|:-----:|------|:----:|
| **0** | 프로젝트 골격, config, ChArUco A4 가이드 | ✅ 완료 |
| **1** | UI 골격, 3-cam 프리뷰, DAQ 리더, TDMS 로더 | ✅ 완료 |
| **2** | ChArUco 내부·외부 캘리브레이션, 세계좌표 정렬 | ⏳ 다음 |
| **3** | rtmlib 기반 2D keypoint, DLT 삼각측량, 필터 | ⏳ |
| **4** | 동기 녹화(카메라+DAQ+엔코더), 세션 관리 | ⏳ |
| **5** | 분석 모듈: Balance, WBA, CMJ, Squat, Encoder-kinematics | ⏳ |
| **6** | 반응시간/고유수용성 검사 프로토콜 | ⏳ |
| **7** | 실시간 최적화 (FP16 실패 시 FP32 fallback, 프레임 스킵) | ⏳ |

---

## 기존 Force Plate Viewer와의 관계

본 프로젝트는 **완전히 분리된 새 프로젝트**입니다. 기존
`C:\Users\FITWIN\Desktop\force_plate\force_plate\new-python-version\` 의 TDMS
파일은 `src/io/tdms_reader.py` 를 통해 오프라인 분석 입력으로 그대로 사용할
수 있습니다. DAQ 드라이버는 같은 nidaqmx 기반이지만 다음이 개선되었습니다:

- **유니코드 안전**: `main.py`에서 `sys.stdout.reconfigure(encoding="utf-8")`
- **음수 force 클리핑 제거** (`config.CLIP_NEGATIVE_FORCE = False`)
- kg 대신 N을 기본 단위로 emit
- 월드 좌표계 CoP(mm) 직접 계산 (board별 오프셋 반영)

---

## 라이선스 / 작성

개인 연구 용도. 외부 라이브러리 라이선스는 각 패키지 참조.

작성: Claude Code 세션, 2026-04-21.
