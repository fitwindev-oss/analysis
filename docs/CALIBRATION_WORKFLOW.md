# 스켈레톤 기반 캘리브레이션 워크플로우

ChArUco 보드 없이, 피험자가 측정 공간에서 움직이는 영상만으로 3대 카메라의 내부·외부 파라미터와 세계좌표 정렬을 모두 해결합니다.

포스 플레이트(558×432 mm)의 CoP가 월드 스케일과 원점을 자동으로 고정합니다.

---

## 전체 흐름 (3단계)

```
  [Step 1] record_calibration_session.py
     └ 60초 동기 녹화 (3대 카메라 + NI DAQ 포스/엔코더)
     └ 화면에 카운트다운, 동작 프롬프트 표시
     └ ESC/SPACE로 원격 취소 가능
     └ 저장: data/calibration/session_YYYYMMDD_HHMMSS/
              C0.mp4, C1.mp4, C2.mp4
              C0.timestamps.csv, ...
              forces.csv
              session.json

  [Step 2] calibrate_from_poses.py --session ...
     └ rtmlib RTMPose로 각 영상 2D keypoint 검출 (캐시됨)
     └ Essential matrix → PnP → Bundle adjustment
     └ 저장: intrinsics_C0.npz, intrinsics_C1.npz, intrinsics_C2.npz
             extrinsics_unscaled.npz   (아직 스케일 모호)
             poses3d_session_*.npz     (삼각측량된 3D 관절 시퀀스)

  [Step 3] align_world_from_force.py --session ...
     └ STAND STILL 구간에서 CoP vs ankle 3D midpoint로 Umeyama similarity
     └ 스케일·회전·병진 확정 후 월드 좌표계로 변환
     └ 저장: world_frame.npz  (최종 결과물)
             poses3d_world_session_*.npz  (mm 스케일 3D 궤적)
```

예상 소요 시간: 총 10~15분 (녹화 1분 + 포즈 검출 3~5분 + 나머지 자동)

---

## 사전 준비

- [ ] 3대 카메라가 측정 공간을 모두 커버하는 위치에 **고정**됨
- [ ] NI USB-6210 연결되어 `forces.csv` 기록 가능 상태
- [ ] 포스 플레이트 좌우 배치 (좌=Board1, 우=Board2)
- [ ] 공간 조명이 **균일**, 직사광·강한 그림자 없음
- [ ] 피험자는 카메라 FOV 내에서 **전신이 보이는** 위치 확인
  - 팁: `python main.py` 로 앱 실행 → Start Preview → 세 카메라에서 피험자 발끝~머리 모두 잘림 없이 보이는 위치에 서기
- [ ] 피험자가 편한 복장 (관절이 잘 인식되도록 옷이 팔다리 형태를 드러내는 게 유리)
- [ ] **최소 한 명 촬영자 필요** — 피험자가 영상 속에 있고, 촬영자는 `ESC/SPACE` 키로 취소 제어

---

## Step 1. 녹화

```
python scripts/record_calibration_session.py
```

옵션:
```
python scripts/record_calibration_session.py --duration 60 --countdown 5
python scripts/record_calibration_session.py --no-daq    # DAQ 없이
python scripts/record_calibration_session.py --name wk1  # 세션 이름 접미사
```

### 화면 표시

- 시작 전 **5초 카운트다운** (큰 숫자, 피험자 위치에서 2m 떨어져도 보임)
- 녹화 중: 상단에 경과 시간 + 잔여 시간, 하단에 **현재 동작 프롬프트**
- 스피커에서 **각 단계 전환마다 비프음** (800~1200Hz)

### 60초 동작 프로토콜 (플레이트 밖으로 나가지 않는 환경용)

| 시간 | 동작 | 목적 |
|------|------|------|
| 0–7s  | **N-pose** (팔 몸 옆, 정지, 정면) | 팔 분리 기준 자세 |
| 7–14s | **T-pose** (팔 수평으로 벌리기, 정지) | 어깨·팔꿈치·손목 관절 분리 |
| 14–24s | **제자리 걷기** (양발 플레이트, 교대로 들기) | 다리 분리 + CoP 좌우 이동 |
| 24–34s | **제자리 걷기 + 좌회전 360°** (천천히) | 전방위 3D 각도 커버 |
| 34–42s | **팔 위/아래 반복** (3회, 천천히) | 수직 관절 궤적 |
| 42–50s | **스쿼트** (3회, 천천히) | Z축 변화 + 하체 관절 |
| 50–60s | **T-pose STAND STILL** ⭐ | 월드 스케일·원점 앵커 |

⭐ 마지막 10초의 **STAND STILL** 구간은 월드 좌표계 정렬의 기준이므로 **완전히 정지**해 주세요. 몸이 흔들리면 ankle midpoint와 CoP의 매칭이 흐트러집니다.

### 조작

- **ESC** 또는 **SPACE**: 녹화 취소
- 그 외 키: 반응 없음

### 출력

`data/calibration/session_YYYYMMDD_HHMMSS/` 폴더에:
- `C0.mp4` `C1.mp4` `C2.mp4` — 640×480 @ 30fps 영상
- `C*.timestamps.csv` — 프레임별 (frame_idx, t_monotonic_ns, t_wall)
- `forces.csv` — 100 Hz 포스/엔코더/CoP 데이터
- `session.json` — 세션 메타데이터

---

## Step 2. 포즈 기반 캘리브레이션

```
python scripts/calibrate_from_poses.py --session session_20260421_150000
```

옵션:
```
--conf 0.45        joint confidence threshold (default 0.45)
--min-joints 6     3-view에서 동시에 보여야 할 최소 joint 수 (default 6)
--no-ba            bundle adjustment 생략 (빠르지만 정확도↓)
--redetect         2D 검출 캐시 무시하고 재실행
```

### 동작 단계

1. **2D keypoint 검출** (rtmlib RTMPose)
   - GTX 960이면 GPU 사용, 실패 시 CPU fallback
   - 3대 × 1800프레임 검출: 약 2~5분 (GPU), 10~20분 (CPU)
   - 결과 캐시: `poses_C0.npz` 등 (재실행 시 빠름)

2. **3-view 대응점 수집**
   - 세 카메라 모두에서 관절 confidence ≥ conf_thresh 인 프레임만 선택

3. **Essential matrix + PnP**
   - cam0 ↔ cam1 E matrix → 초기 R, t (baseline = 1)
   - 3D 점 삼각측량 → cam2 PnP로 일관된 스케일

4. **Bundle adjustment** (scipy.optimize.least_squares, sparse Jacobian)
   - 전체 카메라 extrinsics + 3D 관절 위치 공동 최적화
   - RMS 재투영 오차 출력

### 정상 판정 기준

- `3-view correspondences: 500+` (충분한 데이터)
- `cam0-cam1 essential inliers: 80%+`
- `BA RMS reprojection error: < 3.0 px` (양호), `< 5.0 px` (수용 가능)
- `frames with >=10 joints triangulated: 1000+` (1800 중)

### 출력

- `intrinsics_C0.npz` `intrinsics_C1.npz` `intrinsics_C2.npz`
- `extrinsics_unscaled.npz` — cam0=identity, 전역 스케일 임의
- `poses3d_session_*.npz` — 프레임별 3D 관절 (스케일 임의)

---

## Step 3. 월드 프레임 정렬

```
python scripts/align_world_from_force.py --session session_20260421_150000
```

옵션:
```
--still-start 50     STAND STILL 시작 시각(s) (default 50)
--still-end   60     STAND STILL 종료 시각(s) (default 60)
--auto               프로토콜 시간 무시, 저-sway 자동 감지
```

### 알고리즘

1. STAND STILL 구간의 카메라 프레임 선택
2. 각 프레임마다:
   - ankle midpoint = mean(L_ankle_3D, R_ankle_3D)  — 삼각측량된 3D
   - CoP = (cop_world_x_mm, cop_world_y_mm, 0)  — 포스에서 직접
3. **Umeyama similarity transform** (scale s, rotation R, translation t)
   - `y_i ≈ s * R @ x_i + t` 를 최소제곱으로 풀기
4. 모든 카메라 extrinsics + 3D 궤적에 변환 적용
5. Sanity check: ankle Z ≈ 0, nose Z > 1300 mm

### 정상 판정

- `ankle Z mean`: **–100 ~ +100 mm** (플레이트 표면 근처)
- `nose Z mean`: **1300 ~ 2100 mm** (일반 성인 머리 높이)
- 둘 다 벗어나면 `[warn]` 표시 → 재녹화 또는 `--auto` 시도

### 출력

- **`world_frame.npz`** ⭐ 최종 캘리브레이션 결과
- `poses3d_world_session_*.npz` — mm 단위 3D 궤적 (Phase 3 분석용 시드 데이터)

---

## 전체 실행 예시

```
# 녹화
python scripts/record_calibration_session.py
# → session_20260421_150000 생성

# 캘리브레이션 (2분 ~ 5분)
python scripts/calibrate_from_poses.py --session session_20260421_150000

# 월드 정렬 (몇 초)
python scripts/align_world_from_force.py --session session_20260421_150000
```

완료 후 `data/calibration/` 에 `world_frame.npz` 가 생성되면 Phase 3 (실시간 포즈 추정 + 분석)으로 진행 가능합니다.

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `rtmlib init failed` | onnxruntime-gpu 실패 | `config.POSE_DEVICE = "cpu"` 로 변경 |
| `only N 3-view observations` (N < 60) | 피험자가 일부 카메라 FOV 밖 | 재녹화, 카메라 배치 조정 |
| BA RMS > 5 px | 2D 검출 품질 낮음 or 자세 다양성 부족 | `--conf 0.35` 낮춰 시도, 또는 재녹화 |
| `ankle Z mean` 이 200 mm 이상 | STAND STILL 실패 or CoP 오류 | `--auto` 옵션 사용 |
| `nose Z < 0` | 좌표축 반전 (자동 보정됨) | 스크립트가 자동 Z-flip. 여전히 이상하면 재녹화 |
| DAQ `forces.csv` 없음 | 녹화 시 `--no-daq` 사용됐거나 DAQ 연결 실패 | NI DAQ 연결 확인 후 재녹화 |
| 2D 검출 매우 느림 (>20분) | CPU 모드 | `POSE_MODEL_MODE = "lightweight"` |

## 재캘리브레이션이 필요한 경우

- 카메라 위치·각도 변경 → 전체 Step 1~3 재실행
- 포스 플레이트 위치 변경 → 전체 재실행 (월드 원점 이동)
- 카메라 초점/줌 변경 → 전체 재실행
- 조명 대폭 변화 → Step 2 `--redetect` 옵션 시도 후 필요 시 재녹화

## 정확도 기대치

이 접근법의 현실적 정확도 (biomech 문헌 참조):

| 지표 | 기대 성능 |
|------|-----------|
| 3D 관절 위치 RMSE | 20–50 mm |
| 시상면 관절 각도 | ±3–5° |
| 전두면 관절 각도 | ±5–10° (제한적) |
| 스케일 정확도 (CoP 기반) | ±2% |

Gold-standard(Vicon) 대비 2–4배 오차이지만, **정적 균형 / WBA / CMJ / 스쿼트 분석**에 충분한 수준입니다.
