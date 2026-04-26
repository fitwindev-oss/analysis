"""
Multi-point DAQ voltage-scale calibration using known reference weights.

Protocol (cumulative loading, SYMMETRIC on both boards):

    Step 0:  empty plates                       (zero baseline re-check)
    Step 1:   5 kg on EACH board  (total 10 kg)
    Step 2:  10 kg on EACH board  (total 20 kg)
    Step 3:  15 kg on EACH board  (total 30 kg)
    Step 4:  20 kg on EACH board  (total 40 kg)
    Step 5:  all weights removed                (drift check)

At each step the user places the plates and presses SPACE. The script then
averages 3 seconds of samples (rejecting if std > 30 N - unstable) and
advances automatically.

Analysis:
  * Per-board linear regression  measured_N = slope * true_N
        new_board_scale = current_board_mean_scale / slope
  * Per-corner diagnostic regression (assuming even intra-board weight)
        measured_corner_N = slope_i * (true_board_N / 4)
        warn if any corner slope deviates > 15% from its board average
  * Zero-drift check: Step 5 measured total should be near 0 N
  * R^2 reported; < 0.99 -> warn

Output:
  data/calibration/daq_scale_<ts>/calibration_report.json
  Recommended 8-element DAQ_VOLTAGE_SCALE list printed.
  With --apply, config.py is rewritten in place (.bak saved).

Usage:
  python scripts/calibrate_daq_scale.py
  python scripts/calibrate_daq_scale.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.capture.daq_reader import DaqReader, DaqFrame, GRAVITY


CORNER_NAMES = [
    "b1_tl", "b1_tr", "b1_bl", "b1_br",
    "b2_tl", "b2_tr", "b2_bl", "b2_br",
]


# ── Protocol definition ──────────────────────────────────────────────────────
# (step_idx, title,    per_board_kg, drift_check)
STEPS = [
    (0, "Empty plates - baseline",                    0.0,  True ),
    (1, "Place  5 kg on EACH board  (total 10 kg)",   5.0,  False),
    (2, "Place 10 kg on EACH board  (total 20 kg)",  10.0,  False),
    (3, "Place 15 kg on EACH board  (total 30 kg)",  15.0,  False),
    (4, "Place 20 kg on EACH board  (total 40 kg)",  20.0,  False),
    (5, "Remove ALL weights  (drift check)",          0.0,  True ),
]

MEASURE_SECONDS = 3.0
MAX_STD_N = 30.0       # reject if instability exceeds this


# ── UI helpers ───────────────────────────────────────────────────────────────
def beep(freq=1200, ms=180):
    def _b():
        try:
            import winsound
            winsound.Beep(freq, ms)
        except Exception:
            pass
    threading.Thread(target=_b, daemon=True).start()


def draw_canvas(step_idx: int, title: str, per_board_kg: float,
                sub_status: str, status_color: tuple,
                mean_total_n: float, mean_b1_n: float, mean_b2_n: float,
                std_n: float, progress: float = 0.0,
                progress_target: float = MEASURE_SECONDS) -> np.ndarray:
    """Render a 1000x500 status canvas."""
    W, H = 1000, 500
    canvas = np.full((H, W, 3), 25, dtype=np.uint8)

    # Step banner
    cv2.rectangle(canvas, (0, 0), (W, 85), (40, 40, 40), -1)
    cv2.putText(canvas,
                f"STEP {step_idx} / {len(STEPS) - 1}",
                (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 255), 3, cv2.LINE_AA)

    # Title (action)
    cv2.putText(canvas, title, (20, 140),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Per-board expected
    expected_total = 2.0 * per_board_kg * GRAVITY
    cv2.putText(canvas,
                f"Expected total: {expected_total:6.1f} N   "
                f"(per board: {per_board_kg * GRAVITY:.1f} N)",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

    # Live readings box
    box_top = 210
    cv2.rectangle(canvas, (0, box_top), (W, box_top + 160), (0, 0, 0), -1)
    cv2.putText(canvas, "Live reading:", (20, box_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(canvas,
                f"Total: {mean_total_n:7.1f} N",
                (20, box_top + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    cv2.putText(canvas,
                f"B1: {mean_b1_n:6.1f} N    B2: {mean_b2_n:6.1f} N",
                (20, box_top + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    col_std = (0, 255, 0) if std_n < MAX_STD_N else (0, 0, 255)
    cv2.putText(canvas,
                f"Stability: std = {std_n:5.1f} N   (< {MAX_STD_N:.0f} N)",
                (20, box_top + 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_std, 2)

    # Status
    cv2.rectangle(canvas, (0, H - 100), (W, H - 50),
                  status_color, -1)
    cv2.putText(canvas, sub_status, (20, H - 63),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)

    # Progress bar (when measuring)
    if progress > 0:
        bar_x = 20; bar_y = H - 30; bar_w = W - 40; bar_h = 18
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), 1)
        frac = min(progress / progress_target, 1.0)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + int(bar_w * frac), bar_y + bar_h),
                      (0, 255, 0), -1)

    # Hint footer
    cv2.putText(canvas,
                "SPACE = start/confirm    |    R = retry    |    ESC = cancel",
                (20, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return canvas


# ── Analysis ─────────────────────────────────────────────────────────────────
def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope for y = slope * x, also returns R^2.
       Intercept is forced to zero (assumed after good zero-cal)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom = float((x * x).sum())
    if denom < 1e-9:
        return 1.0, 0.0
    slope = float((x * y).sum() / denom)
    # R^2 computed against true mean-y for reference
    y_pred = slope * x
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return slope, r2


# ── Apply ────────────────────────────────────────────────────────────────────
def update_config_scale(new_scale_list: list[float], config_path: Path) -> bool:
    """Rewrite DAQ_VOLTAGE_SCALE in config.py to a list form.
    Supports both scalar (legacy) and list forms."""
    text = config_path.read_text(encoding="utf-8")
    new_block = (
        f"DAQ_VOLTAGE_SCALE = [\n"
        f"    {new_scale_list[0]:.3f}, {new_scale_list[1]:.3f}, "
        f"{new_scale_list[2]:.3f}, {new_scale_list[3]:.3f},    # Board1 TL TR BL BR\n"
        f"    {new_scale_list[4]:.3f}, {new_scale_list[5]:.3f}, "
        f"{new_scale_list[6]:.3f}, {new_scale_list[7]:.3f},    # Board2 TL TR BL BR\n"
        f"]"
    )

    # Match either scalar assignment or list form (multi-line)
    list_pattern = re.compile(
        r"DAQ_VOLTAGE_SCALE\s*=\s*\[[^\]]*\]", re.DOTALL)
    scalar_pattern = re.compile(
        r"DAQ_VOLTAGE_SCALE\s*=\s*[0-9.]+\s*(?:#.*)?\n")

    if list_pattern.search(text):
        new_text = list_pattern.sub(new_block, text, count=1)
    elif scalar_pattern.search(text):
        new_text = scalar_pattern.sub(new_block + "\n", text, count=1)
    else:
        return False

    backup = config_path.with_suffix(".py.bak")
    backup.write_text(text, encoding="utf-8")
    config_path.write_text(new_text, encoding="utf-8")
    print(f"  [apply] config.py DAQ_VOLTAGE_SCALE updated "
          f"(backup: {backup.name})", flush=True)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write recommended scales to config.py")
    ap.add_argument("--measure-s", type=float, default=MEASURE_SECONDS,
                    help=f"measurement duration per step (default {MEASURE_SECONDS}s)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = config.CALIB_DIR / f"daq_scale_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cal] session: {out_dir}", flush=True)

    # ── DAQ ─────────────────────────────────────────────────────────────────
    daq = DaqReader()
    if not daq.connect():
        print("[cal] DAQ connect failed. Check USB-6210 and NI-DAQmx driver.",
              flush=True)
        return

    latest: list[DaqFrame] = []
    measuring: list[DaqFrame] = []
    measuring_flag = threading.Event()

    def _on_frame(f: DaqFrame):
        latest.clear()
        latest.append(f)
        if measuring_flag.is_set():
            measuring.append(f)
    daq.set_callback(_on_frame)
    daq.start()
    print("[cal] DAQ zero-cal 5s (STAY OFF the plate)", flush=True)

    # Zero-cal wait
    zero_cal_until = time.monotonic() + 6.0

    win = "DAQ scale calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1000, 500)

    step_idx = 0
    phase = "zero_cal"         # zero_cal | await_space | measuring | done
    measure_start: float = 0.0
    cancelled = False
    # Per-step results:   list of dict
    step_data: list[dict] = []

    try:
        while True:
            now = time.monotonic()

            # Compute live stats from buffer
            if latest:
                f = latest[-1]
                live_total = float(f.total_n)
                live_b1    = float(f.b1_total_n)
                live_b2    = float(f.b2_total_n)
            else:
                live_total = live_b1 = live_b2 = 0.0

            # Short-term std estimate from recent measuring window
            if measuring:
                recent = measuring[-min(len(measuring), 300):]
                totals = np.array([fr.total_n for fr in recent])
                live_std = float(totals.std()) if len(totals) > 5 else 0.0
            else:
                live_std = 0.0

            if phase == "zero_cal":
                remaining = max(0.0, zero_cal_until - now)
                canvas = draw_canvas(
                    step_idx=0,
                    title=f"DAQ zero-cal running: {remaining:.1f} s",
                    per_board_kg=0,
                    sub_status="STAY OFF THE PLATE",
                    status_color=(0, 200, 255),
                    mean_total_n=live_total, mean_b1_n=live_b1,
                    mean_b2_n=live_b2, std_n=0,
                )
                if remaining <= 0:
                    phase = "await_space"
                    beep(1000, 180)
                    print(f"[cal] ready for step {step_idx}", flush=True)

            elif phase == "await_space":
                s_idx, title, per_kg, drift = STEPS[step_idx]
                canvas = draw_canvas(
                    step_idx=s_idx, title=title, per_board_kg=per_kg,
                    sub_status="Place the weights - then press SPACE",
                    status_color=(0, 255, 255),
                    mean_total_n=live_total, mean_b1_n=live_b1,
                    mean_b2_n=live_b2, std_n=0,
                )

            elif phase == "measuring":
                s_idx, title, per_kg, drift = STEPS[step_idx]
                elapsed = now - measure_start
                canvas = draw_canvas(
                    step_idx=s_idx, title=title, per_board_kg=per_kg,
                    sub_status=f"MEASURING... {elapsed:.1f} / {args.measure_s:.1f} s",
                    status_color=(0, 255, 0),
                    mean_total_n=live_total, mean_b1_n=live_b1,
                    mean_b2_n=live_b2, std_n=live_std,
                    progress=elapsed, progress_target=args.measure_s,
                )
                if elapsed >= args.measure_s:
                    # Finish measuring
                    measuring_flag.clear()
                    frames = list(measuring)
                    totals = np.array([fr.total_n for fr in frames])
                    std_total = float(totals.std())
                    if std_total > MAX_STD_N:
                        # Reject
                        beep(400, 350)
                        print(f"[cal] step {step_idx} UNSTABLE "
                              f"(std={std_total:.1f} N > {MAX_STD_N}). "
                              f"Press SPACE to retry.", flush=True)
                        phase = "await_retry"
                    else:
                        # Accept
                        arr = np.stack([fr.forces_n for fr in frames])  # (N, 8)
                        mean_8 = arr.mean(axis=0)
                        step_data.append({
                            "step": step_idx,
                            "title": STEPS[step_idx][1],
                            "per_board_kg": per_kg,
                            "true_total_n": 2.0 * per_kg * GRAVITY,
                            "true_per_board_n": per_kg * GRAVITY,
                            "measured_total_n": float(totals.mean()),
                            "measured_total_std_n": std_total,
                            "measured_b1_n": float(np.array(
                                [fr.b1_total_n for fr in frames]).mean()),
                            "measured_b2_n": float(np.array(
                                [fr.b2_total_n for fr in frames]).mean()),
                            "measured_corners_n": [float(v) for v in mean_8],
                            "n_samples": len(frames),
                            "drift_check": drift,
                        })
                        beep(1500, 150)
                        print(f"[cal] step {step_idx} OK  "
                              f"total={totals.mean():7.1f} N  "
                              f"(std {std_total:4.1f})", flush=True)
                        step_idx += 1
                        if step_idx >= len(STEPS):
                            phase = "done"
                        else:
                            phase = "await_space"

            elif phase == "await_retry":
                s_idx, title, per_kg, drift = STEPS[step_idx]
                canvas = draw_canvas(
                    step_idx=s_idx, title=title, per_board_kg=per_kg,
                    sub_status="Unstable - stabilize plates, SPACE to retry",
                    status_color=(100, 100, 255),
                    mean_total_n=live_total, mean_b1_n=live_b1,
                    mean_b2_n=live_b2, std_n=0,
                )

            elif phase == "done":
                break

            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:               # ESC
                cancelled = True
                break
            elif key == ord(" "):
                if phase in ("await_space", "await_retry"):
                    measuring.clear()
                    measuring_flag.set()
                    measure_start = now
                    phase = "measuring"
            elif key == ord("r"):
                if phase in ("await_retry",):
                    measuring.clear()
                    measuring_flag.set()
                    measure_start = now
                    phase = "measuring"
    finally:
        daq.stop()
        cv2.destroyAllWindows()

    if cancelled or len(step_data) < 2:
        print("[cal] cancelled / insufficient data - no report written.",
              flush=True)
        return

    # ── Analysis ────────────────────────────────────────────────────────────
    # Per-board linear regression (through origin)
    load_b1 = np.array([d["true_per_board_n"]  for d in step_data])
    load_b2 = np.array([d["true_per_board_n"]  for d in step_data])
    meas_b1 = np.array([d["measured_b1_n"]     for d in step_data])
    meas_b2 = np.array([d["measured_b2_n"]     for d in step_data])
    meas_tot = np.array([d["measured_total_n"] for d in step_data])
    true_tot = np.array([d["true_total_n"]     for d in step_data])

    slope_b1, r2_b1 = fit_through_origin(load_b1, meas_b1)
    slope_b2, r2_b2 = fit_through_origin(load_b2, meas_b2)
    slope_tot, r2_tot = fit_through_origin(true_tot, meas_tot)

    # Current scale: resolve scalar vs list
    cur_raw = np.asarray(config.DAQ_VOLTAGE_SCALE, dtype=np.float64)
    if cur_raw.ndim == 0:
        cur_scale_8 = np.full(8, float(cur_raw))
    else:
        cur_scale_8 = cur_raw.astype(np.float64)
    cur_b1 = float(cur_scale_8[:4].mean())
    cur_b2 = float(cur_scale_8[4:].mean())

    new_b1 = cur_b1 / slope_b1 if slope_b1 > 1e-6 else cur_b1
    new_b2 = cur_b2 / slope_b2 if slope_b2 > 1e-6 else cur_b2
    new_scale_8 = [new_b1] * 4 + [new_b2] * 4

    # Per-corner diagnostic (assuming centered placement = uniform share)
    corner_slopes: list[float] = []
    for i in range(8):
        board_load_n = load_b1 if i < 4 else load_b2
        expected_corner_n = board_load_n / 4.0
        meas_corner = np.array([d["measured_corners_n"][i] for d in step_data])
        s, _ = fit_through_origin(expected_corner_n, meas_corner)
        corner_slopes.append(float(s))

    # Drift check: measured total at drift-check steps should be near 0
    drift_vals = [d["measured_total_n"] for d in step_data if d["drift_check"]]

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MULTI-POINT DAQ VOLTAGE-SCALE CALIBRATION REPORT")
    print("=" * 72)
    print("\n  Per-step measurements:")
    print(f"  {'step':>4s} {'true_total':>12s} {'meas_total':>12s}  "
          f"{'B1_true':>9s} {'B1_meas':>9s}  {'B2_true':>9s} {'B2_meas':>9s}")
    for d in step_data:
        print(f"  {d['step']:>4d} "
              f"{d['true_total_n']:>12.2f} "
              f"{d['measured_total_n']:>12.2f}  "
              f"{d['true_per_board_n']:>9.2f} {d['measured_b1_n']:>9.2f}  "
              f"{d['true_per_board_n']:>9.2f} {d['measured_b2_n']:>9.2f}")

    print("\n  Per-board linear fit (through origin):")
    print(f"    Board1:  slope = {slope_b1:.4f}   R2 = {r2_b1:.4f}")
    print(f"    Board2:  slope = {slope_b2:.4f}   R2 = {r2_b2:.4f}")
    print(f"    Total :  slope = {slope_tot:.4f}   R2 = {r2_tot:.4f}")

    print("\n  Current scale (mean per board):")
    print(f"    Board1: {cur_b1:7.3f}   Board2: {cur_b2:7.3f}")
    print("  Recommended new scale (per board):")
    print(f"    Board1: {new_b1:7.3f}   Board2: {new_b2:7.3f}")

    print("\n  8-element DAQ_VOLTAGE_SCALE (for config.py):")
    print("    [", ", ".join(f"{v:.3f}" for v in new_scale_8), "]", sep="")

    print("\n  Per-corner diagnostic slopes (even-distribution assumption):")
    for i, s in enumerate(corner_slopes):
        board_slope = slope_b1 if i < 4 else slope_b2
        dev_pct = 100.0 * abs(s - board_slope) / board_slope if board_slope > 1e-6 else 0
        flag = " *" if dev_pct > 15 else ""
        print(f"    {CORNER_NAMES[i]:8s}: slope = {s:.4f}   "
              f"dev from board mean = {dev_pct:5.1f}%{flag}")

    # Warnings
    warnings = []
    if r2_b1 < 0.99 or r2_b2 < 0.99:
        warnings.append(
            f"low R^2 (B1={r2_b1:.3f}, B2={r2_b2:.3f}) - response is not "
            f"linear. Check weights accuracy and stable placement.")
    if len(drift_vals) >= 2 and abs(drift_vals[-1]) > 30:
        warnings.append(
            f"drift detected: final empty-plate reading = "
            f"{drift_vals[-1]:.1f} N (should be near 0). The DAQ zero "
            f"drifted during the session.")
    if any(
        100.0 * abs(s - (slope_b1 if i < 4 else slope_b2))
              / (slope_b1 if i < 4 else slope_b2) > 15
        for i, s in enumerate(corner_slopes)
    ):
        warnings.append(
            "some corner slopes deviate > 15% from their board average. "
            "Weights may have been placed off-center OR those channels "
            "have individual calibration drift.")
    if warnings:
        print("\n  WARNINGS:")
        for w in warnings:
            print(f"    [!] {w}")

    print("\n" + "=" * 72)

    # ── Save JSON ───────────────────────────────────────────────────────────
    report = {
        "timestamp": ts,
        "steps": step_data,
        "per_board": {
            "slope_b1": float(slope_b1), "r2_b1": float(r2_b1),
            "slope_b2": float(slope_b2), "r2_b2": float(r2_b2),
            "current_scale_b1": float(cur_b1),
            "current_scale_b2": float(cur_b2),
            "recommended_scale_b1": float(new_b1),
            "recommended_scale_b2": float(new_b2),
        },
        "total_fit": {
            "slope": float(slope_tot), "r2": float(r2_tot),
        },
        "per_corner_slopes": corner_slopes,
        "recommended_scale_8": [float(x) for x in new_scale_8],
        "drift_values_n": drift_vals,
        "warnings": warnings,
    }
    report_path = out_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  saved: {report_path}", flush=True)

    # ── Apply ────────────────────────────────────────────────────────────────
    if args.apply:
        cfg = Path(__file__).resolve().parents[1] / "config.py"
        ok = update_config_scale(new_scale_8, cfg)
        if not ok:
            print("  [apply] could not locate DAQ_VOLTAGE_SCALE in config.py",
                  flush=True)
    else:
        print("\n  To write these values to config.py automatically, "
              "re-run with --apply.")


if __name__ == "__main__":
    main()
