"""
Project-wide configuration.

Coordinate system (world):
    Origin (0, 0, 0) = left-bottom corner of combined force plate surface,
                       looking DOWN from above (bird's-eye).
    +X  = right  (toward Board2)
    +Y  = forward / away from user
    +Z  = up (vertical)

Units: mm for position, N for force, s for time.
"""

from __future__ import annotations
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data"
SESSIONS_DIR   = DATA_DIR / "sessions"
CALIB_DIR      = DATA_DIR / "calibration"
RESOURCES_DIR  = ROOT_DIR / "resources"
for d in (SESSIONS_DIR, CALIB_DIR, RESOURCES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Force plate geometry (user-specified) ─────────────────────────────────────
# Combined plate footprint. Origin at left-bottom, upper-right at (558, 432).
PLATE_TOTAL_WIDTH_MM  = 558.0    # X span
PLATE_TOTAL_HEIGHT_MM = 432.0    # Y span

# Assumption: two plates are laid SIDE-BY-SIDE along X, each 279 mm wide.
# Board1 = LEFT half  (X: 0 .. 279)
# Board2 = RIGHT half (X: 279 .. 558)
# Update if boards are front-back instead.
PLATE_LAYOUT = "side_by_side"    # or "front_back"
BOARD1_ORIGIN_MM = (0.0,   0.0)  # (x, y) of Board1 left-bottom
BOARD2_ORIGIN_MM = (279.0, 0.0)  # (x, y) of Board2 left-bottom
BOARD_WIDTH_MM   = 279.0         # single plate width
BOARD_HEIGHT_MM  = 432.0         # single plate height

# ── Camera (single Logitech StreamCam on USB 3.0) ─────────────────────────────
# Decision 2026-04-22: switched from 3 cams to 1 cam for these reasons:
#   - biomech only needs sagittal (side) view for all tests
#   - 3-cam USB sharing dropped effective fps to ~17 fps per cam
#   - simpler pipeline: 1 subprocess, faster post-record pose, cleaner reports
#
# StreamCam native spec: up to 1080p60, MJPG, USB 3.1 Type-C.
# camera_detector.detect_available_cameras() still filters; if idx 0 is
# unreachable at launch, CAMERAS ends up empty and pose features disable.
CAMERAS = [
    {"id": "C0", "index": 0, "label": "Logitech StreamCam"},
]
# Active capture settings. Change here to try different combinations —
# use scripts/measure_camera_fps.py to benchmark what the host can
# actually sustain before committing to a value.
CAMERA_WIDTH      = 1280        # 640 / 960 / 1280 / 1920
CAMERA_HEIGHT     = 720         # 480 / 540 / 720  / 1080
CAMERA_FPS        = 60          # StreamCam caps at 60 fps (all resolutions)
CAMERA_FOURCC     = "MJPG"      # MJPG keeps USB bandwidth low; YUY2 for uncompressed
CAMERA_BUFFERSIZE = 1
# Auto-exposure: 0.25 = manual on DirectShow (Windows). Forcing manual keeps
# a fixed shutter so fps is not throttled by low-light auto-slowdown.
CAMERA_AUTO_EXPOSURE = 0.25
CAMERA_EXPOSURE      = -6       # -13..-1 typical range on Windows DirectShow

# Physical-to-image orientation corrections — applied inside the capture
# subprocess so saved mp4, realtime overlay, post-record pose, and replay
# all see the same frame. Edit here when the camera is physically
# remounted; no other change is needed.
#
#   CAMERA_ROTATION : 0 / 90 / 180 / 270   (degrees clockwise)
#     Rotates the captured frame. Use 90 when the camera is mounted with
#     its top pointing toward the subject's right (sagittal-view setup).
#     For 90 / 270, the output mp4's width ↔ height swap automatically.
#
#   CAMERA_MIRROR : bool
#     Horizontal flip applied AFTER rotation. Makes the view intuitive —
#     subject's right hand appears on the right of the screen.
CAMERA_ROTATION = 90
CAMERA_MIRROR   = True

# ── 2D pose estimation (MediaPipe BlazePose, CPU) ────────────────────────────
# Model complexity:
#   0 = Lite   (fastest, ~15 ms/frame on desktop CPU; preferred for realtime)
#   1 = Full   (balanced, default for post-record batch processing)
#   2 = Heavy  (most accurate, ~3-4× slower than Full)
#
# Detection/tracking thresholds follow MediaPipe defaults. Realtime settings
# are overridable from the UI (Measure-tab options); these values are the
# fallbacks used when no override is specified.
POSE_POSTRECORD_COMPLEXITY = 1      # Full — used by offline PoseWorker
POSE_REALTIME_COMPLEXITY   = 0      # Lite — used by on-screen overlay worker
POSE_MIN_DET_CONF          = 0.5
POSE_MIN_TRACK_CONF        = 0.5
POSE_REALTIME_ENABLED      = False  # UI default; user opts-in per session
POSE_REALTIME_CAM_ID       = "C0"   # preferred live-overlay camera
# Directory for auto-downloaded MediaPipe task files (pose_landmarker_*.task)
POSE_MODEL_CACHE_DIR = RESOURCES_DIR / "mediapipe"

# ── NI DAQ (inherited from existing Force Plate Viewer) ───────────────────────
DAQ_DEVICE_NAME = "Dev1"
DAQ_CHANNEL_MAP = [
    "Dev1/ai6",   # Board1 TL
    "Dev1/ai7",   # Board1 TR
    "Dev1/ai5",   # Board1 BL
    "Dev1/ai4",   # Board1 BR
    "Dev1/ai0",   # Board2 TL
    "Dev1/ai1",   # Board2 TR
    "Dev1/ai3",   # Board2 BL
    "Dev1/ai2",   # Board2 BR
]
DAQ_ENCODER1_CHANNEL = "Dev1/ai8"   # Barbell / rod sensor #1
DAQ_ENCODER2_CHANNEL = "Dev1/ai9"   # Barbell / rod sensor #2

DAQ_VOLTAGE_MIN = -10.0
DAQ_VOLTAGE_MAX =  10.0
# Per-channel voltage -> kg conversion. Index order follows DAQ_CHANNEL_MAP:
#   [Board1 TL, TR, BL, BR,   Board2 TL, TR, BL, BR]
# A bare scalar (e.g. 206.0) is also accepted; DaqReader broadcasts it to all
# 8 channels. Update via scripts/calibrate_daq_scale.py (multi-point linear
# regression against known reference weights).
DAQ_VOLTAGE_SCALE = [
    200.771, 200.771, 200.771, 200.771,    # Board1 TL, TR, BL, BR  (calibrated 2026-04-22)
    200.430, 200.430, 200.430, 200.430,    # Board2 TL, TR, BL, BR  (calibrated 2026-04-22)
]
DAQ_VOLTAGE_OFFSET = 0.0
# Encoder voltage->mm calibration (left encoder, ai8)
# Verified 2026-04-23 with scripts/verify_encoders.py --auto using 4 tape
# reference points (0/500/1000/1500 mm): linear fit slope=0.25503,
# intercept=-0.80 mm, residual std=0.91 mm -> 1V ≈ 784.23 mm.
# If the sensor or wiring changes, re-run verify_encoders.py and update.
ENCODER_VOLTAGE_SCALE  = 784.23   # 1V = 784.23 mm
ENCODER_VOLTAGE_OFFSET = 0.0

# Encoder availability flags — flip to True once the hardware is repaired.
# Any consumer (EncoderBar UI, analyzers, report charts) respects these
# so a broken channel does not poison derived metrics.
ENCODER1_AVAILABLE     = True    # left encoder
ENCODER2_AVAILABLE     = False   # right encoder (rewind mechanism broken)
# Display range for the EncoderBar widget. Hardware itself supports up
# to ~4 m; fix to 2 m to match the test-protocol expectation.
ENCODER_MAX_DISPLAY_MM = 2000.0

SAMPLE_RATE_HZ      = 100
MOVING_AVERAGE_SAMPLES = 10   # 10 samples @ 100 Hz = 100 ms smoothing window
ZERO_CAL_SECONDS    = 5
CLIP_NEGATIVE_FORCE = False   # NOTE: new project does NOT clip (keeps sign for debugging)

# Live-display noise floor. When the trainer is NOT actively recording
# (between tests, before "측정 시작"), any VGRF below this threshold
# is clamped to 0 and the CoP marker is suppressed so the dashboard
# shows a clean "idle" state instead of streaming sensor noise.
# During recording (RecorderState.phase == "recording") this floor is
# ignored so jump-flight phases render the full signal.
# 10 kg * g = ~98 N — above typical noise (1-3 N) and small enough that
# even a light child stepping on registers immediately.
LIVE_DISPLAY_MIN_N = 10.0 * 9.80665

# Compatibility with existing TDMS files produced by Force Plate Viewer
TDMS_GROUP_NAME = "Force Data"
TDMS_CHANNEL_NAMES = [
    "Board1_TL", "Board1_TR", "Board1_BL", "Board1_BR",
    "Board2_TL", "Board2_TR", "Board2_BL", "Board2_BR",
    "Board1_Total", "Board2_Total",
    "COP_X", "COP_Y",
    "Encoder1_mm", "Encoder2_mm",
]

# ── ChArUco calibration board (A4 paper) ──────────────────────────────────────
# A4 = 210 x 297 mm. Leaving 15 mm margin each side → printable 180 x 267 mm.
# Board: 6 columns x 8 rows, 30 mm squares = 180 x 240 mm (portrait, fits A4).
CHARUCO_SQUARES_X     = 6
CHARUCO_SQUARES_Y     = 8
CHARUCO_SQUARE_LEN_MM = 30.0
CHARUCO_MARKER_LEN_MM = 22.0     # ~73% of square (OpenCV recommended)
CHARUCO_DICT          = "DICT_5X5_100"    # Smaller bits work better on A4

# Intrinsic calibration target
INTRINSIC_MIN_IMAGES     = 20
INTRINSIC_TARGET_RMS_PX  = 0.5    # aim for <0.5 px reprojection error

# ── Pose / biomech pipeline ───────────────────────────────────────────────────
# COCO-17 keypoints (standard RTMPose Body output)
KPT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Butterworth low-pass filter (post-triangulation)
FILTER_KPT_CUTOFF_HZ = 6.0   # standard for gait/jump
FILTER_GRF_CUTOFF_HZ = 50.0  # for force data

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE   = "Biomech MoCap — 3-Cam + Force Plate"
APP_VERSION = "0.1.0"

# ── Report branding (PDF deliverable) ─────────────────────────────────────────
# These values drive the PDF cover page, header/footer, and accent colors.
# Change here when deploying to a different clinic / rebranding — no code
# changes needed elsewhere.
REPORT_CLINIC_NAME     = "FITWIN MoCap Clinic"
REPORT_CLINIC_SUBTITLE = ""                      # optional tagline under name
# Absolute path to a PNG/JPG logo, or None to fall back to text-only branding.
# Recommended size: 400-800 px wide, transparent background, aspect ~ 3:1.
REPORT_LOGO_PATH: "str | None" = str(
    ROOT_DIR / "src" / "ui" / "resources" / "brand" / "Logo2.png")

# ── Accent colors: two variants ──────────────────────────────────────
# PDF pages are WHITE, so the #AAF500 (luminance ~83%) fails WCAG
# contrast for text/thin strokes (~1.8:1). Use it for FILL areas only
# (big header strip, cover page block, status bars on fills) via
# REPORT_ACCENT_FILL_HEX. For TEXT and thin lines on white, use the
# darker variant REPORT_ACCENT_TEXT_HEX (~5:1 contrast).
REPORT_ACCENT_FILL_HEX = "#AAF500"    # FITWIN logo primary (for fills)
REPORT_ACCENT_TEXT_HEX = "#5F8A00"    # Darker variant for text/thin strokes
REPORT_ACCENT_HEX      = REPORT_ACCENT_TEXT_HEX   # default alias (text-safe)
REPORT_SECONDARY_HEX   = "#0A84FF"
# Footer extras — contact line on every page.
REPORT_FOOTER_LINE   = ""                        # e.g. "T. 02-xxxx  |  contact@x"
# Set a non-None string (e.g. "DRAFT", "CONFIDENTIAL") to overlay a diagonal
# watermark on every page. None disables the watermark.
REPORT_WATERMARK: "str | None" = None

# Top row: 3 cameras + 3D skeleton view. Each tile aspect ~4:3.
TILE_WIDTH  = 360
TILE_HEIGHT = 270
PLOT_UPDATE_MS = 33     # ~30 fps UI
