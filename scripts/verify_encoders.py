"""
Live verification tool for the two linear encoders (Dev1/ai8, Dev1/ai9).

Purpose
-------
  1. Confirm that each encoder responds to physical motion at all.
  2. Confirm that ``config.ENCODER_VOLTAGE_SCALE`` (default 200 mm/V) still
     matches the hardware — walk the slider to known positions along a
     measuring tape and record the delta.
  3. Help decide when ``ENCODER2_AVAILABLE`` can be flipped back to True
     once the rewind mechanism is repaired.

Usage
-----
  Manual (SPACE to mark reference):
    python scripts/verify_encoders.py

  Semi-auto (single-operator — SPACE arms the measurement, script captures
  once the reading stabilizes; then it waits for the next SPACE):
    python scripts/verify_encoders.py --auto
    python scripts/verify_encoders.py --auto --channel 2
    python scripts/verify_encoders.py --auto --targets 0,500,1000,1500

Display
-------
  Manual mode:
    ch         live_V   live_mm   min_mm   max_mm   span_mm   available
    enc1        +1.523   +304.6    +0.3    +1996.5   +1996.2   YES
    enc2        +0.002     +0.4    -0.1      +0.4      +0.5   no (config)

  Semi-auto mode (per-target overwrite-in-place):
    Target 2/5: +500.0 mm   (move encoder to position, then press SPACE)
      live enc1 = +498.2 mm

    ... after SPACE pressed ...
    Target 2/5: +500.0 mm   HOLDING  1.4s / 3.0s
      live enc1 = +498.2 mm   window mean = +498.5   std =  0.47 mm  (keep still)

Controls
--------
  Manual    : SPACE=mark ref, r=reset min/max, q=quit
  Semi-auto : SPACE=start measuring current target, s=skip, q=quit
  Both      : writes ``data/calibration/encoder_verify_<ts>.json`` on exit

Notes
-----
Even if ``ENCODER2_AVAILABLE=False`` in config, this tool still streams the
raw voltage from ai9 so you can check whether the channel is dead, floating,
or producing stale values. Once mechanical repair is done, set the config
flag back to True and re-run this script to confirm.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.capture.daq_reader import DaqReader, DaqFrame


# ── Live state ────────────────────────────────────────────────────────────────
@dataclass
class ChannelState:
    name: str
    available: bool
    last_mm: float = 0.0
    last_v:  float = 0.0
    min_mm:  float = float("inf")
    max_mm:  float = float("-inf")

    def update(self, mm: float) -> None:
        self.last_mm = float(mm)
        self.last_v  = float(mm) / float(config.ENCODER_VOLTAGE_SCALE)
        if mm < self.min_mm: self.min_mm = float(mm)
        if mm > self.max_mm: self.max_mm = float(mm)

    def reset_range(self) -> None:
        self.min_mm = float("inf")
        self.max_mm = float("-inf")


@dataclass
class RefPoint:
    """One user-marked reference reading."""
    ch: str            # "enc1" | "enc2"
    true_mm: float     # typed by operator (known tape measurement)
    read_mm: float     # what the DAQ returned at that instant
    delta_mm: float    # read_mm - true_mm


def _fmt(mm: float) -> str:
    """Format mm value, handling +inf / -inf gracefully."""
    if mm == float("inf") or mm == float("-inf"):
        return "    —"
    return f"{mm:+8.1f}"


def _render(states: list[ChannelState]) -> str:
    header = (f"{'ch':<6}{'live_V':>10}{'live_mm':>10}"
              f"{'min_mm':>10}{'max_mm':>10}{'span_mm':>10}  available")
    lines = [header, "-" * len(header)]
    for s in states:
        span = (s.max_mm - s.min_mm
                if s.min_mm != float("inf") and s.max_mm != float("-inf")
                else 0.0)
        flag = "YES" if s.available else "no (config)"
        lines.append(
            f"{s.name:<6}{s.last_v:+10.3f}{s.last_mm:+10.1f}"
            f"{_fmt(s.min_mm):>10}{_fmt(s.max_mm):>10}"
            f"{span:+10.1f}  {flag}"
        )
    return "\n".join(lines)


# ── Keyboard handling (cross-platform, simple) ────────────────────────────────
def _read_key_nonblocking() -> Optional[str]:
    """Return a single pressed key without blocking, or None.

    Windows: msvcrt.kbhit / getwch. POSIX: termios + select.
    Falls back to None if stdin is not a tty (e.g. piped).
    """
    try:
        if sys.platform == "win32":
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                return ch
            return None
        else:
            import select
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                return sys.stdin.read(1)
            return None
    except Exception:
        return None


def _prompt_true_mm(default_ch: str) -> Optional[tuple[str, float]]:
    """Blocking prompt — returns (ch, true_mm) or None if user cancels.

    Format:  '1 500'  -> ch=enc1, true=500 mm
             '2 1000' -> ch=enc2, true=1000 mm
             '500'    -> defaults to `default_ch`, true=500 mm
             empty / 'c' -> cancel
    """
    print(f"\n[ref] type '1 <mm>' / '2 <mm>' / '<mm>' (default {default_ch})"
          f", enter to cancel:")
    try:
        raw = input("  > ").strip()
    except EOFError:
        return None
    if not raw or raw.lower() in ("c", "cancel"):
        return None
    parts = raw.split()
    try:
        if len(parts) == 1:
            return default_ch, float(parts[0])
        if len(parts) >= 2:
            ch_tok = parts[0].lower().replace("enc", "")
            ch = "enc1" if ch_tok == "1" else "enc2"
            return ch, float(parts[1])
    except ValueError:
        pass
    print("  (bad input, ignored)")
    return None


# ── Manual loop (SPACE-to-mark) ────────────────────────────────────────────
def _manual_loop(states: list[ChannelState], latest: dict,
                 lock: threading.Lock, ref_points: list[RefPoint],
                 duration_s: Optional[float]) -> None:
    t0 = time.time()
    last_draw = 0.0
    last_paint_nl = 0
    while True:
        if duration_s is not None and (time.time() - t0) > duration_s:
            return

        key = _read_key_nonblocking()
        if key is not None:
            k = key.lower()
            if k in ("q", "\x03", "\x1b"):           # q / Ctrl-C / ESC
                return
            if k == "r":
                for s in states:
                    s.reset_range()
                print("\n[verify] min/max reset")
            if k in (" ", "\r", "\n"):
                # Which channel? default to whichever moved more recently
                # (wider span). Operator can override via the prompt.
                default_ch = "enc1"
                if (states[1].max_mm - states[1].min_mm
                        > states[0].max_mm - states[0].min_mm):
                    default_ch = "enc2"
                resp = _prompt_true_mm(default_ch)
                if resp is not None:
                    ch, true_mm = resp
                    with lock:
                        read_mm = latest[ch]
                    delta = read_mm - true_mm
                    ref_points.append(RefPoint(ch, true_mm, read_mm, delta))
                    print(f"  [{ch}]  true={true_mm:+8.1f}  "
                          f"read={read_mm:+8.1f}  delta={delta:+7.2f} mm")

        now = time.time()
        if now - last_draw > 0.1:    # ~10 Hz paint
            with lock:
                block = _render(states)
            if last_paint_nl:
                sys.stdout.write(f"\x1b[{last_paint_nl}A")
            sys.stdout.write(block + "\n")
            sys.stdout.flush()
            last_paint_nl = block.count("\n") + 1
            last_draw = now
        time.sleep(0.02)


# ── Auto-capture parameters ────────────────────────────────────────────────
# These are tuned for a human holding a tape-measure reference. The encoder
# streams at SAMPLE_RATE_HZ (100 Hz) so a 2s window has ~200 samples, which
# gives a reliable std estimate.
_AUTO_STABILITY_WINDOW_S = 2.0   # rolling window for std measurement
_AUTO_STABILITY_STD_MM   = 2.0   # "stable" := std over window < this
_AUTO_HOLD_REQUIRED_S    = 3.0   # must remain stable this long before capture
_AUTO_ARM_MOTION_MM      = 80.0  # after SPACE, must move at least this far
                                  # before stability watch begins — prevents
                                  # capturing at the arm-press position if
                                  # the operator happened to be still there


def _auto_loop(history: collections.deque, latest: dict,
               states: list[ChannelState], lock: threading.Lock,
               ref_points: list[RefPoint],
               targets: list[float], channel: int) -> None:
    """Semi-auto capture loop — key-gated per target.

    Flow for each target:
      1. Script shows the target, live reading, and "press SPACE when ready".
         Operator physically moves the encoder to that position.
      2. When operator presses SPACE, stability watch starts.
      3. Script waits for std<threshold to hold for HOLD_REQUIRED_S, then
         captures mean as the reading and auto-advances the display to the
         next target (which again waits for SPACE).
      4. s=skip current target, q=quit.

    This avoids false auto-captures if the operator moves through a range
    on the way to the intended target.
    """
    ch_key = f"enc{channel}"
    last_paint_nl = 0

    def _paint(lines: list[str]) -> None:
        """Overwrite-in-place painter."""
        nonlocal last_paint_nl
        block = "\n".join(lines)
        if last_paint_nl:
            sys.stdout.write(f"\x1b[{last_paint_nl}A")
        sys.stdout.write("\x1b[0J")            # erase from cursor downward
        sys.stdout.write(block + "\n")
        sys.stdout.flush()
        last_paint_nl = block.count("\n") + 1

    def _wait_for_arm(target: float, idx: int, total: int) -> str:
        """Block until operator signals readiness. Returns 'arm'/'skip'/'quit'."""
        while True:
            with lock:
                cur_mm = latest[ch_key]
            _paint([
                f"Target {idx}/{total}: {target:+.1f} mm   "
                f"(move encoder to position, then press SPACE)",
                f"  live {ch_key} = {cur_mm:+8.1f} mm",
                "  keys: SPACE=start measuring, s=skip, q=quit",
            ])
            k = _read_key_nonblocking()
            if k is not None:
                kl = k.lower()
                if kl in ("q", "\x03", "\x1b"):
                    return "quit"
                if kl == "s":
                    return "skip"
                if kl in (" ", "\r", "\n"):
                    return "arm"
            time.sleep(0.05)

    for idx, target in enumerate(targets, 1):
        # Phase 1 — arming prompt
        action = _wait_for_arm(target, idx, len(targets))
        if action == "quit":
            print("\n[auto] aborted by user")
            return
        if action == "skip":
            _paint([
                f"Target {idx}/{len(targets)}: {target:+.1f} mm   SKIPPED",
            ])
            last_paint_nl = 0
            print()
            continue

        # Phase 2a — motion gate. Operator presses SPACE before moving, so
        # we must wait until they actually reach the target. Without this
        # gate the stability watch would fire immediately at the arm-press
        # position since the encoder was already still there.
        with lock:
            arm_mm = latest[ch_key]
        moved_enough = False
        skipped = False
        while not moved_enough and not skipped:
            k = _read_key_nonblocking()
            if k is not None:
                kl = k.lower()
                if kl in ("q", "\x03", "\x1b"):
                    print("\n[auto] aborted by user")
                    return
                if kl == "s":
                    skipped = True
                    break
            with lock:
                cur_mm = latest[ch_key]
            moved = abs(cur_mm - arm_mm)
            if moved >= _AUTO_ARM_MOTION_MM:
                moved_enough = True
                break
            _paint([
                f"Target {idx}/{len(targets)}: {target:+.1f} mm   "
                f"(move now - waiting for motion)",
                f"  live {ch_key} = {cur_mm:+8.1f} mm   "
                f"armed at {arm_mm:+8.1f} mm   moved {moved:5.1f} mm "
                f"(need >= {_AUTO_ARM_MOTION_MM:.0f})",
                "  keys: s=skip, q=quit",
            ])
            time.sleep(0.05)

        if skipped:
            _paint([
                f"Target {idx}/{len(targets)}: {target:+.1f} mm   SKIPPED",
            ])
            last_paint_nl = 0
            print()
            continue

        # Phase 2b — stability watch and capture
        hold_start: Optional[float] = None
        captured = False
        skipped  = False
        while not captured and not skipped:
            k = _read_key_nonblocking()
            if k is not None:
                kl = k.lower()
                if kl in ("q", "\x03", "\x1b"):
                    print("\n[auto] aborted by user")
                    return
                if kl == "s":
                    skipped = True
                    break

            now = time.monotonic()
            with lock:
                cutoff = now - _AUTO_STABILITY_WINDOW_S
                window = [mm for (t, mm) in history
                          if t >= cutoff and not np.isnan(mm)]
                cur_mm = latest[ch_key]

            if len(window) < int(_AUTO_STABILITY_WINDOW_S * 50):
                _paint([
                    f"Target {idx}/{len(targets)}: {target:+.1f} mm   "
                    f"(collecting samples...)",
                    f"  live {ch_key} = {cur_mm:+8.1f} mm",
                    "  keys: s=skip, q=quit",
                ])
                time.sleep(0.05)
                continue

            arr = np.asarray(window, dtype=np.float64)
            mean_mm = float(arr.mean())
            std_mm  = float(arr.std())

            stable = std_mm < _AUTO_STABILITY_STD_MM
            if stable:
                if hold_start is None:
                    hold_start = now
                held_s = now - hold_start
                if held_s >= _AUTO_HOLD_REQUIRED_S:
                    read_mm = mean_mm
                    delta = read_mm - target
                    ref_points.append(RefPoint(
                        ch=ch_key, true_mm=float(target),
                        read_mm=float(read_mm), delta_mm=float(delta),
                    ))
                    _paint([
                        f"Target {idx}/{len(targets)}: {target:+.1f} mm   "
                        f"CAPTURED",
                        f"  read = {read_mm:+8.1f} mm   "
                        f"delta = {delta:+7.2f} mm   "
                        f"(window std={std_mm:.2f} mm, n={len(arr)})",
                    ])
                    last_paint_nl = 0
                    print()
                    captured = True
                    break
                else:
                    _paint([
                        f"Target {idx}/{len(targets)}: {target:+.1f} mm   "
                        f"HOLDING {held_s:4.1f}s / {_AUTO_HOLD_REQUIRED_S:.1f}s",
                        f"  live {ch_key} = {cur_mm:+8.1f} mm   "
                        f"window mean = {mean_mm:+8.1f}   "
                        f"std = {std_mm:5.2f} mm  (keep still)",
                        "  keys: s=skip, q=quit",
                    ])
            else:
                hold_start = None
                _paint([
                    f"Target {idx}/{len(targets)}: {target:+.1f} mm   "
                    f"(unstable - still settling)",
                    f"  live {ch_key} = {cur_mm:+8.1f} mm   "
                    f"window mean = {mean_mm:+8.1f}   "
                    f"std = {std_mm:5.2f} mm  "
                    f"(need <{_AUTO_STABILITY_STD_MM:.1f} mm)",
                    "  keys: s=skip, q=quit",
                ])
            time.sleep(0.05)

        if skipped:
            _paint([
                f"Target {idx}/{len(targets)}: {target:+.1f} mm   SKIPPED",
            ])
            last_paint_nl = 0
            print()


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(duration_s: Optional[float] = None,
        auto_targets: Optional[list[float]] = None,
        auto_channel: int = 1) -> int:
    states = [
        ChannelState("enc1", bool(getattr(config, "ENCODER1_AVAILABLE", True))),
        ChannelState("enc2", bool(getattr(config, "ENCODER2_AVAILABLE", False))),
    ]
    ref_points: list[RefPoint] = []
    latest: dict[str, float] = {"enc1": 0.0, "enc2": 0.0}
    # Rolling history for stability detection in auto mode. At 100 Hz and a
    # 3s stability window, 500 entries (= 5 s) is more than enough.
    history: collections.deque = collections.deque(maxlen=500)

    lock = threading.Lock()

    def on_frame(fr: DaqFrame) -> None:
        with lock:
            latest["enc1"] = fr.enc1_mm
            latest["enc2"] = fr.enc2_mm
            states[0].update(fr.enc1_mm)
            states[1].update(fr.enc2_mm)
            # Store only the channel being auto-verified; manual mode
            # doesn't read the history.
            mm = fr.enc1_mm if auto_channel == 1 else fr.enc2_mm
            history.append((time.monotonic(), float(mm)))

    daq = DaqReader()
    if not daq.connect():
        print("[verify] DAQ not connected - is the USB-6210 plugged in?")
        return 2
    daq.set_callback(on_frame)
    daq.start()

    print(f"[verify] streaming. SCALE = {config.ENCODER_VOLTAGE_SCALE} mm/V, "
          f"OFFSET = {config.ENCODER_VOLTAGE_OFFSET} mm")
    if auto_targets is not None:
        print(f"[verify] SEMI-AUTO mode: channel=enc{auto_channel}, "
              f"targets={[round(t, 1) for t in auto_targets]} mm")
        print(f"[verify] for each target: move -> SPACE -> hold still "
              f"{_AUTO_HOLD_REQUIRED_S:.1f}s (std<{_AUTO_STABILITY_STD_MM:.1f} mm) "
              f"-> auto-capture.")
    else:
        print("[verify] MANUAL mode: SPACE=mark ref, r=reset min/max, q=quit")
    print("[verify] zero calibration starting - keep everything still "
          f"for {config.ZERO_CAL_SECONDS}s...\n")

    try:
        if auto_targets is not None:
            _auto_loop(history, latest, states, lock, ref_points,
                       auto_targets, auto_channel)
        else:
            _manual_loop(states, latest, lock, ref_points, duration_s)
    except KeyboardInterrupt:
        pass
    finally:
        daq.stop()

    # ── Write report ─────────────────────────────────────────────────────────
    out_dir = config.CALIB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"encoder_verify_{ts}.json"
    report = {
        "timestamp":  ts,
        "voltage_scale_mm_per_v": float(config.ENCODER_VOLTAGE_SCALE),
        "voltage_offset_mm":      float(config.ENCODER_VOLTAGE_OFFSET),
        "channels": {
            s.name: {
                "available":     s.available,
                "final_mm":      float(s.last_mm),
                "final_volts":   float(s.last_v),
                "min_mm":        (None if s.min_mm == float("inf")
                                  else float(s.min_mm)),
                "max_mm":        (None if s.max_mm == float("-inf")
                                  else float(s.max_mm)),
                "span_mm":       (float(s.max_mm - s.min_mm)
                                  if s.min_mm != float("inf")
                                  and s.max_mm != float("-inf") else None),
            } for s in states
        },
        "ref_points": [
            {"ch": r.ch, "true_mm": r.true_mm,
             "read_mm": r.read_mm, "delta_mm": r.delta_mm}
            for r in ref_points
        ],
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[verify] report saved -> {out_path}")
    if ref_points:
        # Per-channel stats + linear fit (if >= 2 points): read_mm = a * true_mm + b
        # - slope ~1.0 indicates the voltage-scale is correct
        # - intercept ~0 indicates no offset
        # - residual std-dev indicates measurement precision
        for ch in ("enc1", "enc2"):
            ch_pts = [r for r in ref_points if r.ch == ch]
            if not ch_pts:
                continue
            deltas = np.array([r.delta_mm for r in ch_pts], dtype=np.float64)
            mean = float(deltas.mean())
            mae  = float(np.abs(deltas).mean())
            print(f"  {ch}: {len(ch_pts)} refs, "
                  f"mean_delta={mean:+.2f} mm, MAE={mae:.2f} mm")
            if len(ch_pts) >= 2:
                trues = np.array([r.true_mm for r in ch_pts], dtype=np.float64)
                reads = np.array([r.read_mm for r in ch_pts], dtype=np.float64)
                slope, intercept = np.polyfit(trues, reads, 1)
                residuals = reads - (slope * trues + intercept)
                res_std = float(residuals.std())
                suggested_scale = (float(config.ENCODER_VOLTAGE_SCALE) / float(slope)
                                    if slope != 0 else float("nan"))
                print(f"       linear fit: slope={slope:.5f}, "
                      f"intercept={intercept:+.2f} mm, res_std={res_std:.2f} mm")
                print(f"       -> suggested ENCODER_VOLTAGE_SCALE = "
                      f"{suggested_scale:.3f} mm/V "
                      f"(currently {config.ENCODER_VOLTAGE_SCALE:.3f})")
                if abs(slope - 1.0) < 0.01 and abs(intercept) < 5.0:
                    print(f"       VERDICT: calibration looks correct "
                          f"(slope within 1%, offset <5 mm).")
                elif abs(slope - 1.0) > 0.05:
                    print(f"       VERDICT: scale off by "
                          f"{(slope - 1) * 100:+.1f}% - update "
                          f"ENCODER_VOLTAGE_SCALE in config.py.")
                else:
                    print(f"       VERDICT: minor drift; acceptable if MAE "
                          f"under your tolerance (typical <5 mm).")
    return 0


def _parse_targets(s: str) -> list[float]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if len(out) < 2:
        raise argparse.ArgumentTypeError(
            "--targets needs at least two values (e.g. '0,500,1000,1500')")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=None,
                    help="auto-stop after N seconds (manual mode only)")
    ap.add_argument("--auto", action="store_true",
                    help="semi-auto mode: SPACE arms, script auto-captures "
                         f"after {_AUTO_HOLD_REQUIRED_S:.0f}s of stable hold")
    ap.add_argument("--targets", type=_parse_targets,
                    default="0,500,1000,1500",
                    help="comma-separated target positions in mm for --auto "
                         "(default: 0,500,1000,1500)")
    ap.add_argument("--channel", type=int, choices=(1, 2), default=1,
                    help="encoder channel to verify in --auto mode (default 1)")
    args = ap.parse_args()

    # Allow --targets with or without --auto; --auto is what actually picks
    # the loop. Normalize the targets list.
    targets = (args.targets if isinstance(args.targets, list)
                else _parse_targets(str(args.targets)))
    return run(
        duration_s=args.duration,
        auto_targets=(targets if args.auto else None),
        auto_channel=args.channel,
    )


if __name__ == "__main__":
    sys.exit(main())
