"""
Benchmark the connected camera at various (resolution, fps, codec) combos.

Usage:
    python scripts/measure_camera_fps.py                 # test a small grid
    python scripts/measure_camera_fps.py --sec 20        # longer runs per combo
    python scripts/measure_camera_fps.py --index 0       # specific cam idx
    python scripts/measure_camera_fps.py --save          # also write mp4

Each combo measures:
    requested fps (config value)
    effective fps (actual frames / elapsed wall seconds)
    first-frame latency (time from open to first successful read)
    drops (number of failed cap.read calls)

Run this BEFORE picking final config.CAMERA_* values.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


DEFAULT_GRID = [
    # (width, height, fps, codec)
    (640,  480, 30, "MJPG"),
    (640,  480, 60, "MJPG"),
    (1280, 720, 30, "MJPG"),
    (1280, 720, 60, "MJPG"),
    (1920, 1080, 30, "MJPG"),
    (1920, 1080, 60, "MJPG"),
]


def run_one(index: int, width: int, height: int, fps_req: int,
            fourcc_str: str, duration_s: float,
            save_path: Path | None = None) -> dict:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return {
            "width": width, "height": height, "fps_req": fps_req,
            "codec": fourcc_str, "ok": False, "reason": "open_failed",
        }
    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*fourcc_str))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps_req)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   config.CAMERA_BUFFERSIZE)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.CAMERA_AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE,      config.CAMERA_EXPOSURE)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    # Warm-up: grab a few frames
    t_warm_start = time.monotonic()
    first_ok_at: float | None = None
    for _ in range(30):
        ok, _ = cap.read()
        if ok and first_ok_at is None:
            first_ok_at = time.monotonic() - t_warm_start
            break

    writer = None
    if save_path is not None:
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_req, (actual_w, actual_h),
        )

    # Measure
    n_ok = 0
    n_fail = 0
    ts_diffs: list[float] = []
    t_last: float | None = None
    t_start = time.monotonic()
    t_end = t_start + duration_s
    while time.monotonic() < t_end:
        ok, frame = cap.read()
        now = time.monotonic()
        if not ok:
            n_fail += 1
            time.sleep(0.001)
            continue
        n_ok += 1
        if t_last is not None:
            ts_diffs.append(now - t_last)
        t_last = now
        if writer is not None:
            writer.write(frame)

    elapsed = time.monotonic() - t_start
    cap.release()
    if writer is not None:
        writer.release()

    eff_fps = n_ok / elapsed if elapsed > 0 else 0.0
    if ts_diffs:
        dt_mean = sum(ts_diffs) / len(ts_diffs)
        dt_max  = max(ts_diffs)
        dt_min  = min(ts_diffs)
    else:
        dt_mean = dt_max = dt_min = 0.0

    return {
        "width":       width,       "height":      height,
        "fps_req":     fps_req,     "codec":       fourcc_str,
        "actual_w":    actual_w,    "actual_h":    actual_h,
        "driver_fps":  actual_fps,  "eff_fps":     eff_fps,
        "frames":      n_ok,        "fails":       n_fail,
        "elapsed_s":   elapsed,
        "dt_mean_ms":  dt_mean * 1000,
        "dt_max_ms":   dt_max  * 1000,
        "dt_min_ms":   dt_min  * 1000,
        "first_ok_ms": (first_ok_at or -1) * 1000,
        "ok":          True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0,
                    help="camera index (default: 0)")
    ap.add_argument("--sec", type=float, default=10.0,
                    help="seconds per combo (default: 10)")
    ap.add_argument("--save", action="store_true",
                    help="save an mp4 per combo (data/benchmarks/)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset like \"1280x720@60\"")
    args = ap.parse_args()

    grid = DEFAULT_GRID
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        filtered = []
        for w, h, f, c in grid:
            key = f"{w}x{h}@{f}"
            if key in wanted:
                filtered.append((w, h, f, c))
        grid = filtered or grid

    out_dir = Path(config.DATA_DIR) / "benchmarks"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] camera index={args.index}  {args.sec:.0f}s per combo\n")
    header = (f"{'Resolution':11s} {'Req':>4s} {'Drv':>4s} "
              f"{'Eff':>5s} {'Frm':>4s} {'Fail':>4s} "
              f"{'dt_mean':>8s} {'dt_max':>7s} {'lat':>5s}")
    print(header)
    print("-" * len(header))
    results = []
    for w, h, f, c in grid:
        save_path = (out_dir / f"bench_{w}x{h}_{f}_{c}.mp4"
                     if args.save else None)
        r = run_one(args.index, w, h, f, c, args.sec, save_path)
        if not r["ok"]:
            print(f"{w}x{h}       {f:>4d} {'fail':>4s} -- open failed")
            continue
        line = (
            f"{r['actual_w']}x{r['actual_h']:<5d} "
            f"{r['fps_req']:>4d} "
            f"{r['driver_fps']:>4.0f} "
            f"{r['eff_fps']:>5.1f} "
            f"{r['frames']:>4d} "
            f"{r['fails']:>4d} "
            f"{r['dt_mean_ms']:>7.1f}ms "
            f"{r['dt_max_ms']:>6.1f}ms "
            f"{r['first_ok_ms']:>4.0f}ms"
        )
        print(line)
        results.append(r)

    if results:
        best = max(results, key=lambda x: x["eff_fps"])
        print(f"\nhighest effective fps: {best['eff_fps']:.1f} at "
              f"{best['actual_w']}x{best['actual_h']} req {best['fps_req']} fps")


if __name__ == "__main__":
    main()
