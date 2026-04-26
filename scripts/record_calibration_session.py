"""
Record a synchronized calibration session.

Captures 3 cameras + NI DAQ (force plate + encoders) simultaneously, for
later use by scripts/calibrate_from_poses.py (extrinsics) and
scripts/align_world_from_force.py (world frame).

Features:
  - 5-second countdown (shown LARGE so a subject 2 m away can read it)
  - On-screen movement prompt timeline (T-pose / walk / turn / arms / squat /
    still) - the subject follows the on-screen cue
  - Audio beep at start, each protocol-stage change, and end (Windows winsound)
  - ESC or SPACE to cancel at any time (remote control)

Outputs (under data/calibration/session_YYYYMMDD_HHMMSS/):
    C0.mp4  C1.mp4  C2.mp4                 (30 fps MJPG-compatible H.264)
    C0.timestamps.csv ...                  (frame_idx, t_monotonic_ns, t_wall)
    forces.csv                             (if DAQ connected)
    session.json                           (protocol timeline + metadata)

Usage:
    python scripts/record_calibration_session.py
    python scripts/record_calibration_session.py --duration 60 --countdown 5
    python scripts/record_calibration_session.py --no-daq    (skip force plate)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.capture.camera_worker import MultiCameraCapture
from src.capture.daq_reader import DaqReader, DaqFrame
from src.capture.wait_for_stance import StabilityDetector, draw_wait_overlay


# ── Default protocol for a 60-second recording ────────────────────────────────
# Each entry: (start_second, prompt_text, prompt_color_BGR)
#
# Designed for a confined-plate setup: subject cannot step OFF the plate, so
# locomotion is done via march-in-place. The final segment (50-60s) must be
# a STILL T-pose - align_world_from_force.py uses it as the world anchor.
DEFAULT_PROTOCOL = [
    ( 0, "N-POSE   -   stand still, arms at sides, face forward",       (0, 255, 255)),
    ( 7, "T-POSE   -   arms out horizontally, stand still",             (0, 255, 200)),
    (14, "MARCH in place   -   lift feet alternately (stay on plate)",  (255, 200,   0)),
    (24, "MARCH + TURN LEFT 360 deg   -   slow, feet on plate",         (0, 200, 255)),
    (34, "ARMS UP / DOWN   -   repeat 3x, slow",                        (255,   0, 255)),
    (42, "SQUAT 3x   -   slow and controlled",                          (0, 100, 255)),
    (50, "STAND STILL   -   T-pose, do not move (world anchor)",        (0, 255,   0)),
]


def current_prompt(elapsed: float, protocol):
    prompt, color = protocol[0][1], protocol[0][2]
    next_change_in = protocol[1][0] - elapsed if len(protocol) > 1 else None
    for i, (t, p, c) in enumerate(protocol):
        if elapsed >= t:
            prompt, color = p, c
            next_change_in = None
            if i + 1 < len(protocol):
                next_change_in = protocol[i + 1][0] - elapsed
    return prompt, color, next_change_in


def beep(freq_hz: int = 1000, ms: int = 180):
    """Beep on Windows (non-blocking via thread)."""
    def _beep():
        try:
            import winsound
            winsound.Beep(freq_hz, ms)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


def draw_countdown(frame: np.ndarray, remaining: float):
    """Large countdown overlay visible from 2 m away."""
    h, w = frame.shape[:2]
    num = max(0, int(np.ceil(remaining)))
    text = str(num) if num > 0 else "GO!"
    # huge bold text, centered
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(w, h) / 110.0
    thickness = max(4, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx, cy = w // 2, h // 2
    # dark backdrop
    cv2.rectangle(frame, (cx - tw // 2 - 30, cy - th // 2 - 30),
                  (cx + tw // 2 + 30, cy + th // 2 + 30),
                  (0, 0, 0), -1)
    col = (0, 255, 0) if num == 0 else (0, 255, 255)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                font, scale, col, thickness, cv2.LINE_AA)
    cv2.putText(frame, "Get ready...", (cx - 160, cy - th // 2 - 60),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def draw_recording_ui(frame: np.ndarray, elapsed: float, total: float,
                      prompt: str, prompt_col: tuple, next_change_in):
    h, w = frame.shape[:2]
    # top banner with timer
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 70), (60, 60, 60), 1)
    remaining = max(0, total - elapsed)
    timer = f"REC  {int(elapsed):02d} / {int(total):02d} s  (remaining {int(remaining):02d}s)"
    cv2.putText(frame, timer, (20, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    # red recording circle
    cv2.circle(frame, (w - 40, 35), 12, (0, 0, 255), -1)
    # bottom prompt
    cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, prompt, (20, h - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, prompt_col, 2, cv2.LINE_AA)
    hint = "ESC or SPACE = cancel"
    if next_change_in is not None and next_change_in > 0:
        hint = f"next prompt in {int(np.ceil(next_change_in))}s   |   " + hint
    cv2.putText(frame, hint, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def side_by_side(frames, labels=None):
    """Stack frames horizontally for the preview window."""
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    h = max(f.shape[0] for f in frames)
    out = []
    for i, f in enumerate(frames):
        r = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
        if labels is not None and i < len(labels):
            cv2.putText(r, labels[i], (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        out.append(r)
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration",  type=float, default=60.0)
    ap.add_argument("--countdown", type=float, default=5.0,
                    help="fixed countdown (only used with --skip-wait)")
    ap.add_argument("--no-daq",    action="store_true",
                    help="skip NI DAQ recording (video only)")
    ap.add_argument("--name",      type=str, default=None,
                    help="custom session name suffix")
    # Smart-wait options (parity with record_session.py)
    ap.add_argument("--subject-kg",   type=float, default=90.0,
                    help="subject body weight in kg (for stability detection)")
    ap.add_argument("--wait-timeout", type=float, default=60.0,
                    help="max seconds to wait for stable stance")
    ap.add_argument("--skip-wait",    action="store_true",
                    help="skip smart stability wait; use fixed countdown")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    session_name = f"session_{ts}" + (f"_{args.name}" if args.name else "")
    session_dir  = config.CALIB_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[record] Session dir: {session_dir}", flush=True)

    # ── Start DAQ (optional) ─────────────────────────────────────────────────
    force_csv_path = session_dir / "forces.csv"
    daq_frames: list[DaqFrame] = []
    daq_latest: list[DaqFrame] = []
    daq_lock = threading.Lock()
    daq = DaqReader() if not args.no_daq else None
    daq_connected = False
    stability = None
    if not args.skip_wait and daq is not None:
        stability = StabilityDetector(
            subject_kg=args.subject_kg,
            stability_target_s=3.0,
            timeout_s=args.wait_timeout,
        )
    record_ready = threading.Event()
    if daq is not None:
        print("[record] Connecting to NI DAQ...", flush=True)
        if daq.connect():
            def _on_frame(f: DaqFrame):
                daq_latest.clear()
                daq_latest.append(f)
                if record_ready.is_set():
                    with daq_lock:
                        daq_frames.append(f)
            daq.set_callback(_on_frame)
            # NOTE: DaqReader.start() triggers its 5s zero-calibration first.
            daq.start()
            print("[record] DAQ streaming (zero-cal ~5s; keep OFF the plate)",
                  flush=True)
            daq_connected = True
        else:
            print("[record] DAQ not connected - proceeding video-only.",
                  flush=True)
            if stability is not None:
                print("[record] smart-wait disabled (no DAQ)", flush=True)
                stability = None

    # ── Start cameras ────────────────────────────────────────────────────────
    print("[record] Opening 3 cameras...", flush=True)
    cam = MultiCameraCapture(record_dir=session_dir)
    cam.start()
    time.sleep(1.0)   # let workers boot, camera settles

    win = "Calibration Recording - follow on-screen prompts"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 600)

    # Cache latest frame per cam for preview
    latest: dict[str, np.ndarray] = {}
    if stability is not None:
        state = "wait"
        stability_arm_at = time.monotonic() + 5.5
    else:
        state = "countdown"
        stability_arm_at = None
    t_phase_start = time.monotonic()
    last_prompt_text = ""
    record_start_ns = None
    record_start_wall = None
    wait_duration_s = 0.0
    cancelled = False

    try:
        while True:
            # Drain camera queue (keep only the most recent per cam)
            drained = 0
            while drained < 30:
                item = cam.get(timeout=0.003)
                if item is None:
                    break
                if "error" in item:
                    print(f"  [cam error] {item.get('cam_id')}: {item['error']}")
                    continue
                latest[item["cam_id"]] = item["bgr"]
                drained += 1

            # Construct preview
            ordered_ids = [c["id"] for c in config.CAMERAS]
            preview_tiles = []
            for cid in ordered_ids:
                f = latest.get(cid)
                if f is None:
                    f = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3),
                                 dtype=np.uint8)
                    cv2.putText(f, f"{cid}  (starting...)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (100, 100, 100), 2)
                preview_tiles.append(f.copy())

            now = time.monotonic()

            if state == "wait":
                if now >= stability_arm_at and daq_latest:
                    stab_state = stability.update(daq_latest[-1])
                    for tile in preview_tiles:
                        draw_wait_overlay(tile, stab_state, args.subject_kg)
                    if stab_state.status == "READY":
                        beep(1200, 250)
                        state = "recording"
                        wait_duration_s = now - t_phase_start
                        t_phase_start = now
                        record_start_ns = time.monotonic_ns()
                        record_start_wall = time.time()
                        record_ready.set()
                        with daq_lock:
                            daq_frames.clear()
                        print(f"[record] STABLE - RECORDING STARTED "
                              f"(waited {wait_duration_s:.1f} s)",
                              flush=True)
                    elif stab_state.status == "TIMEOUT":
                        print(f"[record] wait timed out after "
                              f"{args.wait_timeout:.0f}s - cancel",
                              flush=True)
                        cancelled = True
                        break
                else:
                    remaining = max(0.0, stability_arm_at - now)
                    for tile in preview_tiles:
                        h_ = tile.shape[0]
                        cv2.rectangle(tile, (0, 0), (tile.shape[1], 90),
                                      (0, 0, 0), -1)
                        cv2.putText(tile,
                                    f"DAQ zero-calibration: {remaining:.1f}s",
                                    (20, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.0,
                                    (255, 255, 0), 2)
                        cv2.putText(tile, "stay OFF the plate",
                                    (20, h_ - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 200, 0), 2)

            elif state == "countdown":
                remaining = args.countdown - (now - t_phase_start)
                for tile in preview_tiles:
                    draw_countdown(tile, remaining)
                if remaining <= 0:
                    state = "recording"
                    record_start_ns = time.monotonic_ns()
                    record_start_wall = time.time()
                    t_phase_start = now
                    record_ready.set()
                    with daq_lock:
                        daq_frames.clear()
                    beep(1200, 250)
                    print("[record] RECORDING STARTED", flush=True)

            elif state == "recording":
                elapsed = now - t_phase_start
                prompt, col, nxt = current_prompt(elapsed, DEFAULT_PROTOCOL)
                if prompt != last_prompt_text:
                    print(f"  t={elapsed:5.1f}s  -  {prompt}", flush=True)
                    beep(800, 120)
                    last_prompt_text = prompt
                for tile in preview_tiles:
                    draw_recording_ui(tile, elapsed, args.duration,
                                      prompt, col, nxt)
                if elapsed >= args.duration:
                    beep(600, 400)
                    state = "done"

            elif state == "done":
                break

            grid = side_by_side(preview_tiles, ordered_ids)
            cv2.imshow(win, grid)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord(' '):  # ESC or SPACE cancels
                print("[record] Cancelled by user.", flush=True)
                cancelled = True
                break

    finally:
        print("[record] Stopping cameras...", flush=True)
        cam.stop()
        if daq_connected and daq is not None:
            print("[record] Stopping DAQ...", flush=True)
            daq.stop()
        cv2.destroyAllWindows()

    # ── Persist DAQ data ─────────────────────────────────────────────────────
    if daq_connected and daq_frames:
        with open(force_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_ns", "t_wall",
                "b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
                "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N",
                "enc1_mm", "enc2_mm",
                "total_n", "cop_world_x_mm", "cop_world_y_mm",
            ])
            for fr in daq_frames:
                cx, cy = fr.cop_world_mm()       # already returned in mm
                cx_mm = "" if np.isnan(cx) else f"{cx:.2f}"
                cy_mm = "" if np.isnan(cy) else f"{cy:.2f}"
                w.writerow([
                    fr.t_ns, f"{fr.t_wall:.6f}",
                    *[f"{v:.3f}" for v in fr.forces_n],
                    f"{fr.enc1_mm:.3f}", f"{fr.enc2_mm:.3f}",
                    f"{fr.total_n:.3f}",
                    cx_mm, cy_mm,
                ])
        print(f"[record] Saved force data: {force_csv_path} "
              f"({len(daq_frames)} samples)", flush=True)

    # ── Session metadata ─────────────────────────────────────────────────────
    meta = {
        "session_name": session_name,
        "cancelled": cancelled,
        "duration_s": args.duration,
        "countdown_s": args.countdown,
        "protocol": [{"t": t, "prompt": p} for t, p, _ in DEFAULT_PROTOCOL],
        "cameras": config.CAMERAS,
        "camera_resolution": [config.CAMERA_WIDTH, config.CAMERA_HEIGHT],
        "camera_fps": config.CAMERA_FPS,
        "daq_connected": daq_connected,
        "daq_samples": len(daq_frames),
        "record_start_monotonic_ns": record_start_ns,
        "record_start_wall_s": record_start_wall,
        "wait_duration_s": wait_duration_s,
        "subject_kg": args.subject_kg,
        "smart_wait": stability is not None,
        "plate_total_width_mm": config.PLATE_TOTAL_WIDTH_MM,
        "plate_total_height_mm": config.PLATE_TOTAL_HEIGHT_MM,
    }
    meta_path = session_dir / "session.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"[record] Saved session.json: {meta_path}", flush=True)

    if cancelled:
        print("\n[record] Session was cancelled. Delete the folder if the "
              "recording is incomplete.", flush=True)
    else:
        print(f"\n[record] Recording complete:\n  {session_dir}\n", flush=True)
        print("Next step:\n"
              f"  python scripts/calibrate_from_poses.py --session "
              f"{session_dir.name}\n", flush=True)


if __name__ == "__main__":
    main()
