# 포스 플레이트 시스템 개발/캘리브레이션 스펙

**문서 목적:** 현재 FITWIN MoCap 연구실에 구축된 지면반력(force plate) + 선형 엔코더 시스템의 **하드웨어 구성, 채널 매핑, 신호 처리, 캘리브레이션 값, 데이터 포맷**을 빠짐없이 기록해, 차세대 개발/유지보수 시 동일 동작을 그대로 재현할 수 있게 한다.

**대상 독자:**
- 이 시스템을 인수하는 후속 개발자
- 새 하드웨어로 시스템을 복제할 엔지니어
- 캘리브레이션 주기 점검·재실행을 담당할 운용자

**기준일:** 2026-04-23 (최신 엔코더 재캘리 완료일 기준)
**관련 코드 버전:** `src/capture/daq_reader.py`, `config.py`, `scripts/verify_encoders.py`, `scripts/calibrate_daq_scale.py`

---

## 1. 시스템 개요

```
┌────────────────────────────────────────────────────────────────┐
│  피험자 (플레이트 위)                                           │
├────────────────────────────────────────────────────────────────┤
│  Board1 (LEFT, 279×432 mm)  │  Board2 (RIGHT, 279×432 mm)      │
│   TL TR                      │   TL TR                          │
│   BL BR  × 4 load cells      │   BL BR  × 4 load cells          │
├────────────────────────────────────────────────────────────────┤
│  Encoder 1 (좌)   │   Encoder 2 (우, 현재 비활성)               │
└────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
     (8×load cell V)      (2×encoder V)        (1×USB camera)
           │                    │
           └────────────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │  NI USB-6210 DAQ      │
           │  100 Hz, ±10 V, RSE   │
           │  10 AI channels       │
           └───────────────────────┘
                      │ (USB)
                      ▼
                  PC (Windows)
              biomech-mocap 앱
```

**핵심 수치:**
- 물리 치수: 통합 플레이트 **558 × 432 mm** (= 2 × 279 mm + 0 gap)
- 샘플레이트: **100 Hz** (8 force + 2 encoder = 10 채널)
- 이동평균: **10 샘플 (100 ms)** — 노이즈 억제용
- DAQ 전압 범위: **±10 V RSE** (Referenced Single-Ended)
- 좌표계: World frame, 원점 = 통합 플레이트 **좌측-하단 모서리** (위에서 내려다본 시점)

---

## 2. 하드웨어 인벤토리

### 2.1 DAQ
| 항목 | 사양 |
|---|---|
| 모델 | **NI USB-6210** |
| Device name | `Dev1` (`config.DAQ_DEVICE_NAME`) |
| 분해능 | 16 bit |
| AI 채널 사용 | 10개 (ai0-ai9) |
| Terminal config | **RSE** (Referenced Single-Ended) |
| 전압 범위 | -10.0 ~ +10.0 V |
| 샘플링 모드 | Continuous, clocked at 100 Hz |

### 2.2 포스 플레이트
| 항목 | 사양 |
|---|---|
| 구성 | 독립 2개 보드 (side-by-side) |
| 각 보드 | 279 × 432 mm, 4-point 로드셀 (TL/TR/BL/BR) |
| 통합 풋프린트 | 558 × 432 mm |
| 물리 배치 | `PLATE_LAYOUT = "side_by_side"` — X축(좌우)으로 나란히 |
| Board1 (LEFT) 원점 | `(0.0, 0.0)` mm (world frame) |
| Board2 (RIGHT) 원점 | `(279.0, 0.0)` mm (world frame) |
| 로드셀 타입 | 압축형 (양의 힘 = 아래로 누름) |

**로드셀 물리 배치 (각 보드 내부, 피험자가 위에서 내려다본 시점):**
```
    TL ──── TR
    │       │
    │       │
    BL ──── BR
```
- Y축 아래 = 피험자 전방 아님. 실제 매핑: **TL/TR는 Y=432 쪽 (피험자 뒤), BL/BR는 Y=0 쪽 (피험자 앞)**.
  → `DaqFrame._board_local_cop()` 함수의 cx/cy 공식과 일관 유지 필수.

### 2.3 선형 엔코더
| 항목 | 사양 |
|---|---|
| 물리적 가용 범위 | **~2000 mm** (하드웨어 사양) |
| 현재 검증 범위 | **0 ~ 1500 mm** (2026-04-23 테스트 기준) |
| 디스플레이 범위 | 0 ~ 2000 mm (`config.ENCODER_MAX_DISPLAY_MM`) |
| 출력 | Analog voltage, 0 mm → ~0 V, 직선성 우수 |
| **Encoder 1 (enc1, 좌)** | `config.ENCODER1_AVAILABLE = True` — 사용 가능 |
| **Encoder 2 (enc2, 우)** | `config.ENCODER2_AVAILABLE = False` — **현재 비활성 (되감기 기구 고장)** |
| 수리 후 재활성 | `ENCODER2_AVAILABLE = True`로 설정만 바꾸고 `scripts/verify_encoders.py --auto --channel 2` 실행 |

### 2.4 카메라
| 항목 | 사양 |
|---|---|
| 모델 | Logitech StreamCam (USB 3.0) |
| 해상도 | 1280×720 @ 60 fps |
| 코덱 | MJPG |
| 물리 장착 | 세로 90° 회전 (피험자의 우측 상단), 거울 효과 |
| Config | `config.CAMERAS`, `CAMERA_ROTATION=90`, `CAMERA_MIRROR=True` |

---

## 3. DAQ 채널 매핑

### 3.1 Analog Input 매핑 (config.py:101-113)

**절대 변경 금지:** 이 매핑은 물리 케이블 연결과 1:1 대응. 바꾸면 CoP 계산이 망가짐.

| AI 핀 | 역할 | `DAQ_CHANNEL_MAP` 인덱스 |
|---|---|---|
| `Dev1/ai6` | Board1 **TL** | 0 |
| `Dev1/ai7` | Board1 **TR** | 1 |
| `Dev1/ai5` | Board1 **BL** | 2 |
| `Dev1/ai4` | Board1 **BR** | 3 |
| `Dev1/ai0` | Board2 **TL** | 4 |
| `Dev1/ai1` | Board2 **TR** | 5 |
| `Dev1/ai3` | Board2 **BL** | 6 |
| `Dev1/ai2` | Board2 **BR** | 7 |
| `Dev1/ai8` | **Encoder 1** (left) | `DAQ_ENCODER1_CHANNEL` |
| `Dev1/ai9` | **Encoder 2** (right, 비활성) | `DAQ_ENCODER2_CHANNEL` |

**주의:** Board1 측은 ai4-7 (역순), Board2 측은 ai0-3 (BL/BR 부분 역순). 이는 실제 케이블이 꽂힌 순서를 그대로 반영한 것이므로 **하드웨어 재배선하지 않는 한 수정하지 말 것**.

### 3.2 채널 재배선이 불가피할 경우
1. 새 물리 배선도 종이에 그림 → TL/TR/BL/BR별 AI 핀 번호 확정
2. `config.DAQ_CHANNEL_MAP` 리스트 위 순서로 업데이트
3. `scripts/calibrate_daq_scale.py --apply` 재실행 (코너별 스케일 재산출)
4. 알려진 무게 (10 kg 이상)를 TL/TR/BL/BR 각 코너에 개별 얹어 CoP 출력 확인 — TL 코너에 두면 CoP가 해당 보드의 (w/2 · 0.9, h · 0.9) 근처에 잡혀야 함

---

## 4. 신호 처리 파이프라인

### 4.1 DAQ → DaqFrame 변환 (src/capture/daq_reader.py:239-269)

모든 샘플은 GUI 스레드가 아닌 **DAQ 스레드**에서 처리되어 `DaqFrame` 객체로 wrap 후 콜백 호출.

```
raw_voltage (V)                         <- task.read()
   │
   ├─ force channels [0..7]:
   │    ├─ subtract zero_offset (zero-cal에서 측정)
   │    ├─ moving average (10-sample deque)
   │    ├─ multiply by DAQ_VOLTAGE_SCALE[i]  (V → kg per corner)
   │    ├─ multiply by GRAVITY (9.80665)     (kg → N)
   │    └─ (optional) clip to >= 0 if CLIP_NEGATIVE_FORCE
   │                                                    → forces_n[0..7]
   │
   └─ encoder channels [8, 9]:
        ├─ NO zero offset (엔코더는 absolute position)
        ├─ moving average (same 10-sample deque)
        └─ multiply by ENCODER_VOLTAGE_SCALE  (V → mm)
                                                → enc1_mm, enc2_mm
```

### 4.2 CoP 계산 (src/capture/daq_reader.py:56-110)

각 보드의 local CoP를 먼저 구하고, 두 보드의 force-weighted average로 통합.

**각 보드 local CoP (원점 = 해당 보드의 좌-하단):**
```
corners = [TL, TR, BL, BR]  # N 단위
clipped = max(corners, 0)   # 음수 노이즈 제거
total   = sum(clipped)

cx_local = (TR * w + BR * w) / total     # TL, BL는 x=0에 있음
cy_local = (TL * h + TR * h) / total     # BL, BR는 y=0에 있음
```

**World CoP (통합 플레이트 기준):**
```
b1_world_cop = (cx_local_b1 + BOARD1_ORIGIN_MM[0],  # +0
                cy_local_b1 + BOARD1_ORIGIN_MM[1])  # +0
b2_world_cop = (cx_local_b2 + BOARD2_ORIGIN_MM[0],  # +279
                cy_local_b2 + BOARD2_ORIGIN_MM[1])  # +0

world_cx = (b1_world_cop[0] * f1 + b2_world_cop[0] * f2) / (f1 + f2)
world_cy = (b1_world_cop[1] * f1 + b2_world_cop[1] * f2) / (f1 + f2)
```
- `f1`, `f2` = 각 보드의 total vertical force (N)
- **5 N 미만**이면 CoP = NaN (발을 띄운 것으로 판단)

### 4.3 실시간 표시 노이즈 억제

**Phase-aware suppression (Phase I 결과):**
- `config.LIVE_DISPLAY_MIN_N = 10 × 9.80665 ≈ 98.07 N` (= 10 kg)
- 녹화 중이 아니고 `total_n < LIVE_DISPLAY_MIN_N`이면:
  - VGRF 플롯에 `0.0`을 push (깨끗한 baseline 유지)
  - CoP push 스킵 (궤적 얼어붙음)
  - 리드아웃 "— — —" 표시
- 녹화 중(`RecorderState.phase == "recording"`)에는 always raw signal → 점프 flight 구간 손상 없음

---

## 5. 캘리브레이션

### 5.1 Zero Offset (매 세션 자동)

**대상:** 8 force channels only (엔코더는 절대값이라 미실행)

**실행:** DAQ 스트림 시작 후 **5초간** (`config.ZERO_CAL_SECONDS`)
- 피험자가 플레이트 밖에 있어야 함
- 각 채널의 평균값을 `_zero_offsets[i]`로 저장
- 이후 모든 샘플에서 해당 offset을 뺌

**주의:** 부하가 있는 상태에서 zero-cal이 시작되면 offset이 잘못 계산되어 측정치가 전부 틀어진다. 앱은 zero-cal 동안 "플레이트 밖으로 나가세요" 프롬프트를 띄운다.

### 5.2 Force Scale (per-channel, 수동 실행)

**캘리브레이션 값 (2026-04-22 기준):**
```python
DAQ_VOLTAGE_SCALE = [
    200.771, 200.771, 200.771, 200.771,    # Board1 TL, TR, BL, BR
    200.430, 200.430, 200.430, 200.430,    # Board2 TL, TR, BL, BR
]
# 단위: kg/V per corner. 실제 계산에서 × GRAVITY(9.80665) 하여 N으로 변환.
```

**재캘리 절차:**
1. 기준 무게 준비: 5 kg / 10 kg / 15 kg / 20 kg (± 1% 오차)
2. 실행: `python scripts/calibrate_daq_scale.py --apply`
3. 프로토콜:
   - Step 0: 빈 플레이트 (zero re-check)
   - Step 1: 각 보드에 5 kg (총 10 kg)
   - Step 2: 각 보드에 10 kg (총 20 kg)
   - Step 3: 각 보드에 15 kg (총 30 kg)
   - Step 4: 각 보드에 20 kg (총 40 kg)
   - Step 5: 모두 제거 (drift check)
   - 각 스텝마다 SPACE 키 → 3초 평균 수집 → 자동 진행
4. 스크립트가 per-corner 회귀를 계산하고 `config.py`를 in-place 업데이트 (`.bak` 저장).
5. 검증: R² < 0.99 또는 코너간 편차 > 15%이면 경고 출력 — 부정확 시 물리 설치 재점검.

**재캘리 권장 주기:** 3~6개월마다, 또는 센서/케이블 교체 시.

### 5.3 Encoder Scale

**캘리브레이션 값 (2026-04-23 기준):**
```python
ENCODER_VOLTAGE_SCALE  = 784.23    # 1 V = 784.23 mm
ENCODER_VOLTAGE_OFFSET = 0.0
```

**측정 근거 (4-point linear fit):**
| true (mm) | read (mm, 기준값) | delta (mm) |
|---|---|---|
| 0 | 0.19 | +0.19 |
| 500 | 491 (new scale 적용 후 추정) | ~ 0 |
| 1000 | 996 | ~ 0 |
| 1500 | 1499 | ~ 0 |

- 선형 fit: slope=1.00253 (ideal 1.0), intercept=-1.00 mm, residual std=2.81 mm, MAE=2.87 mm

**초기값의 잘못된 가정 이력:**
- 2026-04-22 이전: `ENCODER_VOLTAGE_SCALE = 200.0` (스펙 추정치)
- 실측 결과 slope 0.255 → 4× 부족 판정
- 재산출: `200 / 0.255 = 784.23`
- **교훈:** 엔코더 센서는 스펙시트의 V/mm 값을 신뢰하지 말고 반드시 실측으로 결정할 것.

**재캘리 절차:**
1. 줄자로 0 / 500 / 1000 / 1500 mm 기준점 표시
2. 실행: `set PYTHONIOENCODING=utf-8 && python scripts/verify_encoders.py --auto`
3. 각 타겟에서 SPACE 누른 뒤 80 mm 이동 → 3초 홀드 → 자동 캡처
4. 종료 후 `data/calibration/encoder_verify_<ts>.json`에 결과 저장
5. 스크립트가 suggested_scale 제안 → `config.ENCODER_VOLTAGE_SCALE` 업데이트
6. 재실행 → VERDICT "calibration looks correct" 확인

**재캘리 권장 주기:** 6개월 또는 센서 교체/수리 시. 엔코더 2 수리 후에는 `--channel 2`로 실행.

### 5.4 CoP 물리적 정확도 검증 (L/R swap 점검)

CoP 시각화 문제를 방지하기 위한 주기적 sanity check:

1. 피험자가 **왼쪽 보드만 (Board1)** 밟고 5초 유지
2. Measure 탭 CoP 궤적 확인 → X < 279 범위에 있어야 함 (왼쪽 절반)
3. Reports 탭에서 같은 세션 리플레이 → 동일 위치에 CoP 표시되어야 함
4. 일치하지 않으면: 물리 배선 vs `DAQ_CHANNEL_MAP` vs `BOARD*_ORIGIN_MM` 매핑 재점검

---

## 6. 데이터 포맷

### 6.1 forces.csv (src/capture/session_recorder.py:556-562)

매 세션에 저장. 15 컬럼, 100 Hz:
```csv
t_ns, t_wall,
b1_tl_N, b1_tr_N, b1_bl_N, b1_br_N,
b2_tl_N, b2_tr_N, b2_bl_N, b2_br_N,
enc1_mm, enc2_mm,
total_n, cop_world_x_mm, cop_world_y_mm
```
- `t_ns`: monotonic 타임스탬프 (int ns)
- `t_wall`: wall clock (float seconds since epoch)
- 8개 corner force: Newtons (양수=누름)
- 2개 encoder: mm (절대 위치)
- `total_n`: 8 corner force의 합
- `cop_world_x/y`: mm, 통합 플레이트 world frame, NaN 가능 (force < 5 N일 때)

### 6.2 session.json (src/capture/session_recorder.py:589-622)

세션 메타. 전체 키 목록:
```json
{
  "name": "free_exercise_20260423_182527_67909427",
  "test": "free_exercise",
  "duration_s": 60.0,
  "cancelled": false,
  "fell_off_detected": false,
  "cameras": [{"id": "C0", "index": 0, "label": "Logitech StreamCam"}],
  "daq_connected": true,
  "n_daq_samples": 6209,
  "n_stimuli": 0,
  "record_start_monotonic_ns": 1234567890,
  "record_start_wall_s": 1714123456.789,
  "wait_duration_s": 3.2,
  "subject_id": "67909427",
  "subject_name": "유기웅",
  "subject_kg": 93.0,
  "smart_wait": true,
  "vision": null,           // "open"/"closed"/null (balance tests only)
  "stance": "two",          // "two"/"left"/"right"
  "reaction_trigger": null, // "auto"/"manual" (reaction only)
  "reaction_responses": null,
  "encoder_prompt": null,   // (deprecated — encoder test removed from UI)
  "exercise_name": "바벨 스쿼트",   // free_exercise only
  "load_kg": 93.0,                  // effective load (bodyweight added if use_bw)
  "use_bodyweight_load": true,      // free_exercise only
  "uses_encoder": true              // non-balance tests only; null for balance
}
```

### 6.3 result.json (src/analysis/dispatcher.py:115-137)

분석 결과. Dispatcher가 test_type에 따라 적절한 analyzer에 위임.
```json
{
  "test": "free_exercise",
  "analyzed_at": "2026-04-23T19:00:00+09:00",
  "duration_analysis_s": 0.83,
  "result": { <analyzer-specific payload> },
  "error": null
}
```

테스트별 result payload는 각 analyzer 모듈 참고:
- `src/analysis/balance.py` — balance_eo/ec
- `src/analysis/cmj.py` — cmj
- `src/analysis/squat.py` — squat, overhead_squat
- `src/analysis/encoder.py` — encoder (레거시; UI에선 제거됨)
- `src/analysis/free_exercise.py` — free_exercise (wraps encoder analyzer)
- `src/analysis/reaction.py` — reaction
- `src/analysis/proprio.py` — proprio

---

## 7. 핵심 config 상수 요약 (config.py)

**변경 시 하드웨어 재검증 필수인 항목:**
```python
DAQ_DEVICE_NAME   = "Dev1"
DAQ_CHANNEL_MAP   = [...]                  # 물리 배선과 1:1
DAQ_VOLTAGE_SCALE = [8개 per-corner kg/V]  # 최근 캘리: 2026-04-22
ENCODER_VOLTAGE_SCALE = 784.23             # 최근 캘리: 2026-04-23
PLATE_LAYOUT      = "side_by_side"
BOARD1_ORIGIN_MM  = (0.0, 0.0)
BOARD2_ORIGIN_MM  = (279.0, 0.0)
BOARD_WIDTH_MM    = 279.0
BOARD_HEIGHT_MM   = 432.0
```

**사용자가 조정해도 안전한 항목:**
```python
SAMPLE_RATE_HZ         = 100
MOVING_AVERAGE_SAMPLES = 10     # 낮추면 응답↑ 노이즈↑
ZERO_CAL_SECONDS       = 5      # 늘리면 offset 정확도↑ 대기시간↑
LIVE_DISPLAY_MIN_N     = 98.07  # 10 kg × g
ENCODER_MAX_DISPLAY_MM = 2000.0
CLIP_NEGATIVE_FORCE    = False  # True로 하면 음수 force가 0으로 클리핑됨 (CoP만 이미 적용)
```

**하드웨어 상태 플래그:**
```python
ENCODER1_AVAILABLE = True      # left
ENCODER2_AVAILABLE = False     # right — 되감기 기구 고장, 수리 후 True로 변경
```

---

## 8. 알려진 이슈 / 현재 하드웨어 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| Encoder 2 (right) | **비활성** | 되감기 기구 고장. `ENCODER2_AVAILABLE=False`로 config에서 플래그만 끄면 UI/분석이 전부 "비활성" 표시. 수리 후 플래그 True + `verify_encoders.py --channel 2` 실행. |
| Board1 / Board2 voltage scale | 다름 (200.771 vs 200.430) | 각 보드 제조공차. 혼동 방지 위해 8-element 리스트로 분리 관리. |
| Encoder 데이터 정확도 | ±3 mm (MAE) | 인체가 정지 상태에서 홀드한 측정이라 노이즈 포함. 기계적 엔코더로는 정상. |
| 카메라 1대 제한 | 의도된 설계 | 이전 3-카메라 → USB 대역폭으로 fps 저하 → 1-카메라 sagittal view로 전환 (2026-04-22). |

---

## 9. 재현 체크리스트 (신규 설치 시)

새 하드웨어로 동일 시스템을 구축할 때 순서:

- [ ] **하드웨어 연결**
  - [ ] NI USB-6210 → PC USB 2.0+
  - [ ] `nidaqmx` Python 패키지 + NI 드라이버 설치 (`pip install nidaqmx`)
  - [ ] NI MAX (Measurement & Automation Explorer)에서 device 이름을 `Dev1`로 설정
  - [ ] 8개 force cable + 2개 encoder cable 연결 (섹션 3.1 매핑 그대로)
- [ ] **코드 설치**
  - [ ] 저장소 클론 + `pip install -r requirements.txt`
  - [ ] `data/`, `data/calibration/`, `data/sessions/`, `resources/` 생성 확인
- [ ] **DAQ 연결 테스트**
  - [ ] `python -c "from src.capture.daq_reader import DaqReader; r = DaqReader(); print(r.connect())"` → True
- [ ] **Force scale 캘리**
  - [ ] 기준 무게 5/10/15/20 kg 준비
  - [ ] `python scripts/calibrate_daq_scale.py --apply`
  - [ ] R² > 0.99 확인
  - [ ] config.py 갱신 확인 + 주석에 캘리 날짜 기록
- [ ] **Encoder scale 캘리**
  - [ ] 줄자 0/500/1000/1500 mm 표시
  - [ ] `set PYTHONIOENCODING=utf-8 && python scripts/verify_encoders.py --auto`
  - [ ] VERDICT "calibration looks correct" 확인
  - [ ] config.py 갱신 + 캘리 날짜 기록
- [ ] **CoP 정합 검증**
  - [ ] 피험자 Board1에만 서서 5초 녹화
  - [ ] Measure/Replay 모두에서 CoP가 X < 279에 있는지 확인
- [ ] **앱 실행 & 녹화 테스트**
  - [ ] `python main.py` → MeasureTab에서 짧은 테스트
  - [ ] forces.csv, session.json이 생성되는지 확인
  - [ ] Reports 탭에서 분석/리플레이 동작
- [ ] **엔코더 2 활성 여부 결정**
  - [ ] 고장 상태면 `ENCODER2_AVAILABLE = False` 유지
  - [ ] 정상이면 `True`로 설정 + `--channel 2` 캘리 실행

---

## 10. 관련 파일 인덱스

| 파일 | 역할 |
|---|---|
| [config.py](../config.py) | 모든 상수 |
| [src/capture/daq_reader.py](../src/capture/daq_reader.py) | DAQ 스트림 + DaqFrame + CoP 계산 |
| [src/capture/session_recorder.py](../src/capture/session_recorder.py) | 녹화 오케스트레이션 + forces.csv/session.json 쓰기 |
| [src/analysis/common.py](../src/analysis/common.py) | forces.csv → ForceSession 로더 |
| [src/analysis/dispatcher.py](../src/analysis/dispatcher.py) | test_type → analyzer 라우팅 |
| [scripts/calibrate_daq_scale.py](../scripts/calibrate_daq_scale.py) | Force scale 다지점 회귀 캘리 |
| [scripts/verify_encoders.py](../scripts/verify_encoders.py) | Encoder scale 자동 캡처 검증 |
| [scripts/diagnose_session.py](../scripts/diagnose_session.py) | 저장된 세션 품질 점검 |
| [docs/CALIBRATION_WORKFLOW.md](./CALIBRATION_WORKFLOW.md) | 카메라/월드프레임 캘리 (별도) |

---

## 11. 향후 변경 시 주의사항

1. **채널 번호 변경은 배선 변경과 동시에.** config만 바꾸면 조용히 잘못된 CoP가 나온다.
2. **Voltage scale은 실측이 유일한 진실.** 스펙시트 값 믿지 말 것 (엔코더 4× 오차 사례 참고).
3. **Zero offset은 절대 수동으로 하드코딩하지 말 것.** 매 세션 zero-cal이 정답.
4. **새 테스트 유형 추가 시** 반드시 `config.LIVE_DISPLAY_MIN_N` 억제가 적절한지 재검토 (점프 flight처럼 저중력 구간이 있는지).
5. **Plate layout이 바뀌면** (예: front-back로 재배치) `PLATE_LAYOUT`, `BOARD*_ORIGIN_MM`, `PLATE_TOTAL_WIDTH_MM`, `PLATE_TOTAL_HEIGHT_MM` 모두 업데이트 + CoP 위젯 `setLimits` 재계산.
6. **100 Hz를 초과하는 샘플레이트**로 변경 시 USB 대역폭 + `MOVING_AVERAGE_SAMPLES` 비례 스케일 조정 + 분석 필터 cutoff 재점검.

---

**문서 버전:** 1.0
**최종 갱신:** 2026-04-23 (Phase E 엔코더 재캘리 완료 시점)
**다음 리뷰 권장:** 하드웨어 변경 시 또는 6개월 후
