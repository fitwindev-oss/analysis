"""
Record an EXPERIMENT session (not calibration).

Difference from record_calibration_session.py:
  - No fixed movement protocol. The user chooses test type via --test and
    an appropriate on-screen prompt appears.
  - Optional reaction-time stimulus generator (beeps + flash) that writes
    a stimulus_log.csv.
  - Session folder goes under data/sessions/, not data/calibration/.
  - Output: videos (mp4), timestamps, forces.csv, session.json,
            plus stimulus_log.csv / proprio_log.csv where applicable.

Usage examples:
    python scripts/record_session.py --test balance --duration 30
    python scripts/record_session.py --test wba --duration 20
    python scripts/record_session.py --test cmj --duration 10
    python scripts/record_session.py --test encoder --duration 60 \\
           --encoder-prompt "5 reps back squat, 60kg"
    python scripts/record_session.py --test reaction --duration 60 \\
           --n-stimuli 15 --stim-min-gap 2 --stim-max-gap 5
    python scripts/record_session.py --test squat --duration 30
"""
from __future__ import annotations

import argparse
import csv
import json
import random
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


TEST_PROMPTS = {
    "balance_eo":  [
        ( 0, "EYES OPEN - stand still",                                (0, 255, 0)),
    ],
    "balance_ec":  [
        ( 0, "EYES CLOSED - stand still (close your eyes NOW)",        (0, 200, 255)),
    ],
    "cmj":      [
        ( 0, "CMJ - stand still on plate",                             (255, 255, 0)),
        ( 3, "PREPARE to jump",                                        (255, 200, 0)),
        ( 5, "JUMP as high as possible (CMJ, 1 attempt)",              (0, 255, 255)),
    ],
    "encoder":  [
        ( 0, "ENCODER - follow movement prompt below the screen",      (255, 0, 255)),
    ],
    "reaction": [
        ( 0, "REACTION - respond to stimulus shown on screen",         (0, 255, 0)),
    ],
    "squat":    [
        ( 0, "SQUAT - perform reps when ready",                        (255, 100, 0)),
    ],
    "overhead_squat": [
        ( 0, "OVERHEAD SQUAT - bar/dowel overhead, perform reps",      (255, 100, 0)),
    ],
    "proprio":  [
        ( 0, "PROPRIOCEPTION - follow target cue, then reproduce",     (0, 200, 255)),
    ],
}

# Balance variants get a suffix added to the prompt based on --stance
STANCE_LABEL = {
    "two":   "  -  BOTH feet on plates",
    "left":  "  -  LEFT foot only (Board1)",
    "right": "  -  RIGHT foot only (Board2)",
}

# Reaction response types and their display labels + (key, label, color)
REACTION_RESPONSES = {
    "left_shift":  ("LEFT SHIFT!",   (0, 200, 255)),
    "right_shift": ("RIGHT SHIFT!",  (0, 200, 255)),
    "jump":        ("JUMP!",         (0, 255, 0)),
}


def current_prompt(elapsed, proto):
    p, c = proto[0][1], proto[0][2]
    for t, pp, cc in proto:
        if elapsed >= t:
            p, c = pp, cc
    return p, c


def beep(freq=1000, ms=180):
    def _beep():
        try:
            import winsound
            winsound.Beep(freq, ms)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


def draw_countdown(frame, remaining):
    h, w = frame.shape[:2]
    num = max(0, int(np.ceil(remaining)))
    text = str(num) if num > 0 else "GO!"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(w, h) / 110.0
    thick = max(4, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cx, cy = w // 2, h // 2
    cv2.rectangle(frame, (cx - tw // 2 - 30, cy - th // 2 - 30),
                  (cx + tw // 2 + 30, cy + th // 2 + 30), (0, 0, 0), -1)
    col = (0, 255, 0) if num == 0 else (0, 255, 255)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                font, scale, col, thick, cv2.LINE_AA)


def draw_rec_ui(frame, elapsed, total, prompt, col, stim_banner=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"REC  {int(elapsed):02d} / {int(total):02d} s",
                (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    cv2.circle(frame, (w - 40, 35), 12, (0, 0, 255), -1)
    cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, prompt, (20, h - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
    cv2.putText(frame, "ESC / SPACE = cancel",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # Stimulus banner (big red flash for reaction-time trials)
    if stim_banner is not None:
        cv2.rectangle(frame, (0, 80), (w, h - 100), (0, 0, 255), -1)
        txt = stim_banner
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.0, 6)
        cv2.putText(frame, txt, (w // 2 - tw // 2, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), 6)


def side_by_side(frames, labels):
    if not frames:
        return np.zeros((480, 640, 3), np.uint8)
    h = max(f.shape[0] for f in frames)
    out = []
    for f, lab in zip(frames, labels):
        r = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
        cv2.putText(r, lab, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        out.append(r)
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",      required=True, choices=sorted(TEST_PROMPTS))
    ap.add_argument("--duration",  type=float, default=30.0)
    ap.add_argument("--countdown", type=float, default=5.0,
                    help="fixed countdown (only used when --skip-wait)")
    ap.add_argument("--name",      type=str, default=None)
    ap.add_argument("--no-daq",    action="store_true")
    ap.add_argument("--no-cam",    action="store_true",
                    help="force-only recording (skip all camera streams)")
    ap.add_argument("--encoder-prompt", type=str, default=None,
                    help="custom prompt text to display for encoder tests")
    # Balance options
    ap.add_argument("--stance", choices=["two", "left", "right"], default="two",
                    help="foot stance for balance_eo/balance_ec tests")
    # Reaction options
    ap.add_argument("--n-stimuli", type=int, default=10)
    ap.add_argument("--stim-min-gap", type=float, default=2.0)
    ap.add_argument("--stim-max-gap", type=float, default=5.0)
    ap.add_argument("--trigger", choices=["auto", "manual"], default="auto",
                    help="reaction trigger mode: auto-timed or operator keypress")
    ap.add_argument("--responses", type=str, default="random",
                    help="reaction response set: 'random', 'left_shift', "
                         "'right_shift', 'jump', or comma-separated subset "
                         "like 'left_shift,right_shift'")
    # Smart-wait options
    ap.add_argument("--subject-kg",   type=float, default=90.0,
                    help="subject body weight in kg (for stability detection)")
    ap.add_argument("--wait-timeout", type=float, default=60.0,
                    help="max seconds to wait for stable stance")
    ap.add_argument("--skip-wait",    action="store_true",
                    help="skip smart stability wait; use fixed countdown")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{args.test}_{ts}" + (f"_{args.name}" if args.name else "")
    session_dir = config.SESSIONS_DIR / name
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rec] session: {session_dir}", flush=True)

    # DAQ
    daq = None if args.no_daq else DaqReader()
    daq_frames: list[DaqFrame] = []           # frames stored during RECORDING
    daq_latest: list[DaqFrame] = []           # most-recent frame for wait UI
    daq_lock = threading.Lock()
    daq_connected = False
    stability = None
    # Determine stance mode: balance_eo/balance_ec use --stance, others = two
    if args.test in ("balance_eo", "balance_ec"):
        stance_mode = args.stance
    else:
        stance_mode = "two"
    if not args.skip_wait and daq is not None:
        stability = StabilityDetector(
            subject_kg=args.subject_kg,
            stability_target_s=3.0,
            timeout_s=args.wait_timeout,
            stance_mode=stance_mode,
        )
    record_ready = threading.Event()

    if daq is not None and daq.connect():
        def _on(frame):
            # Always keep latest frame for wait UI
            daq_latest.clear()
            daq_latest.append(frame)
            # Once recording has begun, persist frames
            if record_ready.is_set():
                with daq_lock:
                    daq_frames.append(frame)
        daq.set_callback(_on)
        daq.start()
        daq_connected = True
        print("[rec] DAQ streaming (zero-cal ~5s; keep OFF the plate)",
              flush=True)
    elif daq is not None:
        print("[rec] DAQ connect failed - video only", flush=True)
        if stability is not None:
            print("[rec] smart-wait disabled (no DAQ)", flush=True)
            stability = None

    # Cameras
    cam = None
    if not args.no_cam:
        cam = MultiCameraCapture(record_dir=session_dir)
        cam.start()
        time.sleep(1.0)

    win = f"Recording ({args.test})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 600)

    # Prompt schedule (balance_eo/balance_ec get stance hint appended)
    proto = list(TEST_PROMPTS[args.test])
    if args.test in ("balance_eo", "balance_ec"):
        suffix = STANCE_LABEL.get(stance_mode, "")
        proto = [(t, p + suffix, c) for (t, p, c) in proto]
    if args.test == "encoder" and args.encoder_prompt:
        proto = [(0, f"ENCODER - {args.encoder_prompt}", (255, 0, 255))]

    # Reaction: parse response pool + schedule stimuli (auto mode only)
    response_pool = []
    if args.test == "reaction":
        if args.responses == "random":
            response_pool = list(REACTION_RESPONSES.keys())
        else:
            response_pool = [r.strip() for r in args.responses.split(",")
                             if r.strip() in REACTION_RESPONSES]
            if not response_pool:
                raise RuntimeError(
                    f"--responses produced empty pool: {args.responses}")
        print(f"[rec] reaction response pool: {response_pool}", flush=True)
        print(f"[rec] reaction trigger: {args.trigger}", flush=True)
    stim_times = []   # planned times (auto mode only)
    if args.test == "reaction" and args.trigger == "auto":
        rng = random.Random()
        t_cur = 2.0
        for _ in range(args.n_stimuli):
            t_cur += rng.uniform(args.stim_min_gap, args.stim_max_gap)
            if t_cur >= args.duration - 1:
                break
            # pre-pick response type per stimulus
            stim_times.append(
                (t_cur, rng.choice(response_pool)))
        print(f"[rec] scheduled {len(stim_times)} stimuli",
              flush=True)

    stim_events = []   # list of dicts
    stim_rng = random.Random()   # used for manual random key

    latest: dict[str, np.ndarray] = {}
    # Initial state: wait for stable stance unless --skip-wait
    if stability is not None:
        state = "wait"
        # Give DAQ ~5s to finish zero-cal before evaluating stability
        stability_arm_at = time.monotonic() + 5.5
    else:
        state = "countdown"
        stability_arm_at = None
    t_phase = time.monotonic()
    wait_start_wall = time.time()
    rec_start_ns = None
    rec_start_wall = None
    wait_duration_s = 0.0
    stim_banner_until = 0.0
    stim_banner_type = None
    cancelled = False

    def _fire_stim(events_list, response_type, now_mono):
        t_wall = time.time()
        t_ns   = time.monotonic_ns()
        beep(1500, 120)
        events_list.append({
            "trial_idx": len(events_list),
            "t_wall":    t_wall,
            "t_ns":      t_ns,
            "stimulus_type":   "audio_visual",
            "response_type":   response_type,
        })
        rel_t = now_mono - t_phase
        print(f"  [STIM #{len(events_list)}]  "
              f"t={rel_t:.2f}s  response={response_type}", flush=True)

    try:
        while True:
            # Drain cam queue
            if cam is not None:
                drained = 0
                while drained < 30:
                    item = cam.get(timeout=0.003)
                    if item is None:
                        break
                    if "error" in item:
                        continue
                    latest[item["cam_id"]] = item["bgr"]
                    drained += 1

            ordered = [c["id"] for c in config.CAMERAS]
            tiles = []
            for cid in ordered:
                f = latest.get(cid)
                if f is None:
                    f = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3),
                                 np.uint8)
                    cv2.putText(f, f"{cid} (starting...)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                tiles.append(f.copy())

            now = time.monotonic()
            stim_banner = None

            if state == "wait":
                # Only arm stability detector after DAQ zero-cal finishes
                if now >= stability_arm_at and daq_latest:
                    stab_state = stability.update(daq_latest[-1])
                    # Overlay on every tile
                    for tl in tiles:
                        draw_wait_overlay(tl, stab_state, args.subject_kg,
                                          stance_mode=stance_mode)
                    if stab_state.status == "READY":
                        beep(1200, 220)
                        state = "recording"
                        wait_duration_s = now - t_phase
                        t_phase = now
                        rec_start_ns = time.monotonic_ns()
                        rec_start_wall = time.time()
                        record_ready.set()       # start persisting DAQ frames
                        with daq_lock:
                            daq_frames.clear()   # discard pre-recording frames
                        print(f"[rec] STABLE - RECORDING STARTED "
                              f"(waited {wait_duration_s:.1f} s)",
                              flush=True)
                    elif stab_state.status == "TIMEOUT":
                        print(f"[rec] wait timed out after "
                              f"{args.wait_timeout:.0f} s - cancel",
                              flush=True)
                        cancelled = True
                        break
                else:
                    # Still in DAQ zero-cal
                    remaining = max(0.0, stability_arm_at - now)
                    for tl in tiles:
                        cv2.rectangle(tl, (0, 0), (tl.shape[1], 90),
                                      (0, 0, 0), -1)
                        cv2.putText(tl,
                                    f"DAQ zero-calibration: {remaining:.1f}s",
                                    (20, 60),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.0,
                                    (255, 255, 0), 2)
                        cv2.putText(tl, "stay OFF the plate",
                                    (20, tl.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 200, 0), 2)

            elif state == "countdown":
                rem = args.countdown - (now - t_phase)
                for tl in tiles:
                    draw_countdown(tl, rem)
                if rem <= 0:
                    state = "recording"
                    rec_start_ns = time.monotonic_ns()
                    rec_start_wall = time.time()
                    t_phase = now
                    record_ready.set()
                    with daq_lock:
                        daq_frames.clear()
                    beep(1200, 220)
                    print("[rec] RECORDING STARTED", flush=True)

            elif state == "recording":
                elapsed = now - t_phase
                # Auto-timed reaction stimuli
                while stim_times and elapsed >= stim_times[0][0]:
                    _, chosen_type = stim_times.pop(0)
                    _fire_stim(stim_events, chosen_type, now)
                    stim_banner_until = now + 0.5
                    stim_banner_type = chosen_type
                if now < stim_banner_until:
                    stim_banner = REACTION_RESPONSES.get(
                        stim_banner_type, ("NOW!", (0, 0, 255)))[0]
                prompt, col = current_prompt(elapsed, proto)
                for tl in tiles:
                    draw_rec_ui(tl, elapsed, args.duration, prompt, col,
                                stim_banner)
                if elapsed >= args.duration:
                    beep(600, 300)
                    state = "done"

            elif state == "done":
                break

            grid = side_by_side(tiles, ordered)
            cv2.imshow(win, grid)
            key = cv2.waitKey(20) & 0xFF
            # Global ESC only (SPACE is used as reaction stimulus key now)
            if key == 27:
                print("[rec] cancelled", flush=True)
                cancelled = True
                break
            # SPACE only cancels OUTSIDE of manual-reaction sessions
            if key == ord(" ") and not (args.test == "reaction"
                                         and args.trigger == "manual"):
                if state == "wait":
                    print("[rec] cancelled during wait", flush=True)
                    cancelled = True
                    break

            # Manual-trigger reaction: hotkeys fire a stimulus + choose response
            if state == "recording" and args.test == "reaction" \
                    and args.trigger == "manual":
                chosen = None
                if key == ord("1") and "left_shift" in response_pool:
                    chosen = "left_shift"
                elif key == ord("2") and "right_shift" in response_pool:
                    chosen = "right_shift"
                elif key == ord("3") and "jump" in response_pool:
                    chosen = "jump"
                elif key == ord(" "):   # SPACE = random from pool
                    chosen = stim_rng.choice(response_pool)
                if chosen is not None and len(stim_events) < args.n_stimuli:
                    _fire_stim(stim_events, chosen, now)
                    stim_banner_until = now + 0.5
                    stim_banner_type = chosen
                if len(stim_events) >= args.n_stimuli and state == "recording":
                    # auto-end once we hit target count
                    beep(600, 300)
                    state = "done"
    finally:
        if cam is not None:
            cam.stop()
        if daq is not None and daq_connected:
            daq.stop()
        cv2.destroyAllWindows()

    # Persist force data
    force_csv = session_dir / "forces.csv"
    if daq_connected and daq_frames:
        with open(force_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_ns", "t_wall",
                "b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
                "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N",
                "enc1_mm", "enc2_mm",
                "total_n", "cop_world_x_mm", "cop_world_y_mm",
            ])
            for fr in daq_frames:
                cx, cy = fr.cop_world_mm()
                cx_mm = "" if np.isnan(cx) else f"{cx:.2f}"
                cy_mm = "" if np.isnan(cy) else f"{cy:.2f}"
                w.writerow([
                    fr.t_ns, f"{fr.t_wall:.6f}",
                    *[f"{v:.3f}" for v in fr.forces_n],
                    f"{fr.enc1_mm:.3f}", f"{fr.enc2_mm:.3f}",
                    f"{fr.total_n:.3f}", cx_mm, cy_mm,
                ])
        print(f"[rec] forces.csv: {len(daq_frames)} samples", flush=True)

    # Stimulus log
    if stim_events:
        path = session_dir / "stimulus_log.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial_idx", "t_wall", "t_ns",
                        "stimulus_type", "response_type"])
            for e in stim_events:
                w.writerow([e["trial_idx"], f"{e['t_wall']:.6f}",
                            e["t_ns"], e["stimulus_type"],
                            e.get("response_type", "")])
        print(f"[rec] stimulus_log.csv: {len(stim_events)} events", flush=True)

    # Session metadata
    vision = None
    if args.test == "balance_eo":
        vision = "open"
    elif args.test == "balance_ec":
        vision = "closed"
    meta = {
        "name": name,
        "test": args.test,
        "duration_s": args.duration,
        "cancelled": cancelled,
        "cameras": config.CAMERAS if cam is not None else [],
        "daq_connected": daq_connected,
        "n_daq_samples": len(daq_frames),
        "n_stimuli": len(stim_events),
        "record_start_monotonic_ns": rec_start_ns,
        "record_start_wall_s": rec_start_wall,
        "wait_duration_s": wait_duration_s,
        "subject_kg": args.subject_kg,
        "smart_wait": stability is not None,
        # New: stance + vision for balance variants
        "vision": vision,
        "stance": stance_mode,
        # New: reaction-test details
        "reaction_trigger":  args.trigger if args.test == "reaction" else None,
        "reaction_responses": response_pool if args.test == "reaction" else None,
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n[rec] complete:\n  {session_dir}\n", flush=True)
    print(f"Analyze with:\n  python scripts/analyze.py {args.test} "
          f"--session {session_dir.name}\n", flush=True)


if __name__ == "__main__":
    main()
