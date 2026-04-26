"""
Unified CLI for running any analysis on a recorded session.

Usage:
    python scripts/analyze.py balance   --session balance_20260421_...
    python scripts/analyze.py wba       --session wba_...
    python scripts/analyze.py cmj       --session cmj_...
    python scripts/analyze.py encoder   --session ... --channel 1 --bar-kg 60
    python scripts/analyze.py reaction  --session reaction_...
    python scripts/analyze.py squat     --session squat_... [--encoder 1]
    python scripts/analyze.py proprio   --session proprio_... [--signal joint --joint 16]

  Add --json <out.json> to dump the full result as JSON.
  Add --plot to save a diagnostic PNG under the session folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.analysis.balance   import analyze_balance_file
from src.analysis.cmj       import analyze_cmj_file
from src.analysis.encoder   import analyze_encoder_file
from src.analysis.reaction  import analyze_reaction_file
from src.analysis.squat     import analyze_squat_file
from src.analysis.proprio   import analyze_proprio_file
# Note: WBA subcommand was removed; balance output already contains
# mean_board1_pct / mean_board2_pct which cover weight-bearing asymmetry.
# src.analysis.wba module is still available as a library import.


def _resolve_session(name: str) -> Path:
    """Look for the session folder under data/sessions/ (or absolute path)."""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for base in (config.SESSIONS_DIR, config.CALIB_DIR):
        candidate = base / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"session '{name}' not found under "
                            f"{config.SESSIONS_DIR} or {config.CALIB_DIR}")


def _print_result(title: str, d: dict):
    print(f"\n===== {title} =====")
    def _walk(k, v, indent=0):
        pad = "  " * indent
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            for kk, vv in v.items():
                _walk(kk, vv, indent + 1)
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                print(f"{pad}{k}: ({len(v)} items)")
                for i, item in enumerate(v):
                    print(f"{pad}  [{i}]")
                    for kk, vv in item.items():
                        _walk(kk, vv, indent + 2)
            else:
                print(f"{pad}{k}: {v}")
        elif isinstance(v, float):
            print(f"{pad}{k}: {v:.4f}")
        else:
            print(f"{pad}{k}: {v}")
    for k, v in d.items():
        _walk(k, v)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)

    for name in ["balance", "balance_eo", "balance_ec",
                 "cmj", "reaction", "squat", "proprio"]:
        s = sub.add_parser(name)
        s.add_argument("--session", required=True)
        s.add_argument("--json",    type=str, default=None)
        s.add_argument("--plot",    action="store_true")
        if name in ("balance", "balance_eo", "balance_ec"):
            s.add_argument("--t-start", type=float, default=None)
            s.add_argument("--t-end",   type=float, default=None)
            s.add_argument("--cutoff-hz", type=float, default=3.0,
                           help="LPF cutoff Hz (3-5 Hz standard; lower = "
                                "less noise in path length)")
        if name == "cmj":
            s.add_argument("--bw-kg", type=float, default=None)
        if name == "squat":
            s.add_argument("--encoder", type=int, default=0,
                           help="0=off, 1=use encoder1, 2=use encoder2")
        if name == "proprio":
            s.add_argument("--signal", choices=["cop", "joint"], default="cop")
            s.add_argument("--joint",  type=int, default=16)

    # encoder has extra args
    s_enc = sub.add_parser("encoder")
    s_enc.add_argument("--session",   required=True)
    s_enc.add_argument("--channel",   type=int, default=1, choices=[1, 2])
    s_enc.add_argument("--bar-kg",    type=float, default=20.0)
    s_enc.add_argument("--min-rom-mm", type=float, default=150.0)
    s_enc.add_argument("--json",      type=str, default=None)
    s_enc.add_argument("--plot",      action="store_true")

    args = ap.parse_args()
    session_dir = _resolve_session(args.session)
    print(f"[analyze] {args.kind}  session: {session_dir}", flush=True)

    result = None
    if args.kind in ("balance", "balance_eo", "balance_ec"):
        result = analyze_balance_file(
            session_dir, args.t_start, args.t_end, cutoff_hz=args.cutoff_hz)
    elif args.kind == "cmj":
        result = analyze_cmj_file(session_dir, bw_override_kg=args.bw_kg)
    elif args.kind == "encoder":
        result = analyze_encoder_file(
            session_dir, channel=args.channel,
            bar_mass_kg=args.bar_kg, min_rom_mm=args.min_rom_mm)
    elif args.kind == "reaction":
        result = analyze_reaction_file(session_dir)
    elif args.kind == "squat":
        result = analyze_squat_file(session_dir, use_encoder=args.encoder)
    elif args.kind == "proprio":
        result = analyze_proprio_file(session_dir, signal=args.signal,
                                      joint_idx=args.joint)

    d = result.to_dict()
    _print_result(args.kind.upper(), d)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"\n[analyze] JSON saved: {args.json}", flush=True)

    if args.plot:
        _maybe_plot(args.kind, session_dir, result)


def _maybe_plot(kind: str, session_dir: Path, result):
    """Save a simple diagnostic PNG for each analysis type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed; skipping plot", flush=True)
        return

    from src.analysis.common import load_force_session
    force = load_force_session(session_dir)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    if kind in ("balance", "balance_eo", "balance_ec"):
        axes[0].plot(force.t_s, force.cop_x, label="CoP X (ML)")
        axes[0].plot(force.t_s, force.cop_y, label="CoP Y (AP)")
        axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("mm")
        axes[0].legend(); axes[0].set_title("CoP trajectory")
        axes[1].plot(force.cop_x, force.cop_y)
        axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Y (mm)")
        axes[1].set_aspect("equal"); axes[1].set_title("2D CoP path")
    elif kind == "cmj":
        axes[0].plot(force.t_s, force.total, label="vGRF")
        if hasattr(result, "t_unweight_onset_s"):
            for name, tv in [("unweight", result.t_unweight_onset_s),
                              ("takeoff", result.t_takeoff_s),
                              ("landing", result.t_landing_s)]:
                axes[0].axvline(tv, ls="--", alpha=0.5, label=name)
        axes[0].legend(); axes[0].set_ylabel("N"); axes[0].set_title("CMJ vGRF")
        axes[1].text(
            0.05, 0.5,
            f"Jump height (impulse): {result.jump_height_m_impulse*1000:.0f} mm\n"
            f"Jump height (flight):  {result.jump_height_m_flight*1000:.0f} mm\n"
            f"Peak force: {result.peak_force_n:.0f} N ({result.peak_force_bw:.2f}xBW)\n"
            f"Peak RFD: {result.peak_rfd_n_s:.0f} N/s\n"
            f"Peak power: {result.peak_power_w:.0f} W",
            family="monospace", fontsize=12,
        )
        axes[1].set_axis_off()
        axes[0].set_xlabel("time (s)")
    elif kind == "encoder":
        ch = result.channel
        disp = force.enc1 if ch == 1 else force.enc2
        axes[0].plot(force.t_s, disp); axes[0].set_ylabel("mm")
        axes[0].set_title(f"Encoder {ch} displacement")
        for r in result.reps:
            axes[0].axvspan(r.t_start_s, r.t_end_s, alpha=0.1, color="green")
            axes[0].axvline(r.t_bottom_s, ls="--", color="red", alpha=0.5)
        axes[1].text(
            0.02, 0.5,
            f"Reps: {result.n_reps}\n"
            f"Mean ROM: {result.mean_rom_mm:.1f} mm\n"
            f"Mean concentric velocity: {result.mean_con_vel_m_s:.3f} m/s\n"
            f"Peak concentric velocity: {result.peak_con_vel_m_s:.3f} m/s\n"
            f"Mean concentric power: {result.mean_con_power_w:.1f} W\n"
            f"Peak concentric power: {result.peak_con_power_w:.1f} W",
            family="monospace", fontsize=12,
        )
        axes[1].set_axis_off()
        axes[0].set_xlabel("time (s)")
    else:
        plt.close(fig)
        return

    out = session_dir / f"{kind}_report.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[analyze] plot saved: {out}", flush=True)


if __name__ == "__main__":
    main()
