"""
verify_recorder_alignment.py — Phase U3-2 quality check.

Measures the alignment between force end and camera end for one or
more session directories. After the U3-2 fix in SessionRecorder._finalize,
the gap should be ≤ 30 ms (one DAQ sample at 100 Hz). Pre-fix sessions
will show a ~2 s gap matching the camera-stop join timeout.

Usage:
    # Check the most recently recorded session
    python tools/verify_recorder_alignment.py --latest

    # Check a specific session
    python tools/verify_recorder_alignment.py data/sessions/cmj_2026...

    # Check every session under data/sessions/
    python tools/verify_recorder_alignment.py --all

    # Threshold customisation (default 30 ms)
    python tools/verify_recorder_alignment.py --latest --gap-ms 50

Exit codes:
    0 = all checked sessions PASS
    1 = at least one session FAIL
    2 = bad arguments / no session found
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


# Make project root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_walls(session_dir: Path, cam_id: str) -> Optional[list[float]]:
    p = session_dir / f"{cam_id}.timestamps.csv"
    if not p.exists():
        return None
    walls: list[float] = []
    with open(p, "r") as f:
        for row in csv.DictReader(f):
            walls.append(float(row["t_wall_s"]))
    return walls or None


def _load_force_end_wall(session_dir: Path) -> Optional[float]:
    p = session_dir / "forces.csv"
    if not p.exists():
        return None
    last_wall: Optional[float] = None
    with open(p, "r") as f:
        for row in csv.DictReader(f):
            try:
                last_wall = float(row["t_wall"])
            except (KeyError, ValueError):
                continue
    return last_wall


def _load_force_first_wall(session_dir: Path) -> Optional[float]:
    p = session_dir / "forces.csv"
    if not p.exists():
        return None
    with open(p, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                return float(row["t_wall"])
            except (KeyError, ValueError):
                return None
    return None


def check_session(session_dir: Path, gap_threshold_ms: float = 30.0) -> dict:
    """Compute alignment metrics for one session.

    Returns a dict with the metric values + a 'pass' bool.
    """
    result: dict = {
        "session": session_dir.name,
        "pass": False,
        "errors": [],
    }
    meta_p = session_dir / "session.json"
    if not meta_p.exists():
        result["errors"].append("missing session.json")
        return result
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception as e:
        result["errors"].append(f"bad session.json: {e}")
        return result

    target_dur = float(meta.get("duration_s") or 0.0)
    rec_start  = meta.get("record_start_wall_s")
    rec_end    = meta.get("record_end_wall_s")
    n_samples  = int(meta.get("n_daq_samples") or 0)

    if rec_start is None:
        result["errors"].append("no record_start_wall_s in meta")
        return result

    rec_start = float(rec_start)
    rec_end_present = rec_end is not None
    rec_end_f = float(rec_end) if rec_end_present else None

    # Force end (relative to rec_start)
    force_last = _load_force_end_wall(session_dir)
    if force_last is None:
        result["errors"].append("no forces.csv or empty")
        return result
    force_end_rel = force_last - rec_start

    # Camera end — try every cam in cameras list
    cams = meta.get("cameras") or [{"id": "C0"}]
    cam_results: list[dict] = []
    for cam in cams:
        cam_id = cam.get("id") if isinstance(cam, dict) else str(cam)
        walls = _load_walls(session_dir, cam_id)
        if not walls:
            cam_results.append({"cam": cam_id, "skipped": True})
            continue
        cam_end_rel = walls[-1] - rec_start
        gap_s = abs(force_end_rel - cam_end_rel)
        cam_results.append({
            "cam":           cam_id,
            "frames":        len(walls),
            "cam_end_rel":   cam_end_rel,
            "gap_s":         gap_s,
        })

    result.update({
        "target_dur":      target_dur,
        "n_daq_samples":   n_samples,
        "force_end_rel":   force_end_rel,
        "rec_end_present": rec_end_present,
        "rec_end_rel":     (rec_end_f - rec_start) if rec_end_f else None,
        "cams":            cam_results,
    })

    # PASS condition:
    #   1. Every camera with timestamps shows |force_end - cam_end| ≤ threshold
    #   2. record_end_wall_s is present in meta (post-fix sessions)
    #      — this is informational, not strictly required for old sessions
    valid_cams = [c for c in cam_results if not c.get("skipped")]
    if not valid_cams:
        result["errors"].append("no camera timestamps found")
        return result
    max_gap = max(c["gap_s"] for c in valid_cams)
    result["max_gap_s"] = max_gap
    result["gap_threshold_s"] = gap_threshold_ms / 1000.0
    result["pass"] = max_gap <= gap_threshold_ms / 1000.0
    return result


def _format_row(r: dict) -> str:
    if r.get("errors"):
        return f"  {r['session']:<46}  ERR: {'; '.join(r['errors'])}"
    parts = [
        f"{r['session']:<46}",
        f"target={r['target_dur']:5.1f}s",
        f"force_end={r['force_end_rel']:6.3f}s",
    ]
    for c in r["cams"]:
        if c.get("skipped"):
            parts.append(f"{c['cam']}=skip")
        else:
            parts.append(f"{c['cam']}_end={c['cam_end_rel']:6.3f}s "
                         f"gap={c['gap_s']*1000:6.1f}ms")
    if r.get("rec_end_present"):
        parts.append(f"rec_end={r['rec_end_rel']:6.3f}s")
    else:
        parts.append("rec_end=---")
    parts.append("PASS" if r["pass"] else "FAIL")
    return "  " + "  ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify SessionRecorder force/camera alignment.")
    ap.add_argument("session", nargs="?",
                    help="Session directory path (or use --latest / --all)")
    ap.add_argument("--latest", action="store_true",
                    help="Check the most recent session under data/sessions/")
    ap.add_argument("--all", action="store_true",
                    help="Check every session under data/sessions/")
    ap.add_argument("--gap-ms", type=float, default=30.0,
                    help="Pass threshold for |force_end - cam_end| (default 30 ms)")
    args = ap.parse_args()

    sessions: list[Path] = []
    if args.session:
        sessions = [Path(args.session)]
    elif args.latest or args.all:
        root = Path("data/sessions")
        if not root.exists():
            print(f"data/sessions not found", file=sys.stderr)
            return 2
        all_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True)
        sessions = [all_dirs[0]] if (args.latest and all_dirs) else all_dirs
        if not sessions:
            print(f"no sessions found in {root}", file=sys.stderr)
            return 2
    else:
        ap.print_help()
        return 2

    print(f"Threshold for PASS: |force_end - cam_end| ≤ {args.gap_ms:.0f} ms")
    print("-" * 100)

    n_pass = 0
    n_fail = 0
    for sd in sessions:
        r = check_session(sd, gap_threshold_ms=args.gap_ms)
        print(_format_row(r))
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1

    print("-" * 100)
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL out of {len(sessions)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
