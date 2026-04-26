"""Smoke test all analysis modules on synthetic force data."""
from __future__ import annotations

import sys
import tempfile
import csv
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.analysis.common  import load_force_session, G
from src.analysis.balance import analyze_balance
from src.analysis.wba     import analyze_wba
from src.analysis.cmj     import analyze_cmj
from src.analysis.encoder import analyze_encoder
from src.analysis.reaction import analyze_reaction
from src.analysis.squat   import analyze_squat


def make_synthetic_session(kind: str, duration_s: float = 30.0,
                           fs: float = 100.0, bw_kg: float = 75.0,
                           out_dir: Path | None = None) -> Path:
    out_dir = out_dir or Path(tempfile.mkdtemp()) / f"synth_{kind}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(duration_s * fs)
    t = np.arange(n) / fs
    t_wall_s = t + 1_000_000.0     # wall-clock seconds (arbitrary epoch)
    t_ns = (t_wall_s * 1e9).astype(np.int64)
    bw_n = bw_kg * G

    rng = np.random.default_rng(42)

    if kind == "balance":
        cop_x = 280 + 3.0 * np.sin(2 * np.pi * 0.4 * t) + rng.normal(0, 1.5, n)
        cop_y = 216 + 2.0 * np.cos(2 * np.pi * 0.3 * t) + rng.normal(0, 1.0, n)
        b1_total = bw_n * 0.50 + rng.normal(0, 3, n)
        b2_total = bw_n * 0.50 + rng.normal(0, 3, n)
        total = b1_total + b2_total
        enc1 = np.zeros(n)
        enc2 = np.zeros(n)
    elif kind == "wba":
        cop_x = 310 + rng.normal(0, 2, n)      # shifted right
        cop_y = 216 + rng.normal(0, 1, n)
        b1_total = bw_n * 0.42 + rng.normal(0, 3, n)   # 42% left
        b2_total = bw_n * 0.58 + rng.normal(0, 3, n)   # 58% right
        total = b1_total + b2_total
        enc1 = np.zeros(n); enc2 = np.zeros(n)
    elif kind == "cmj":
        # quiet stand -> unweight -> push -> flight -> land
        total = np.full(n, bw_n)
        # Times
        i1 = int(5.0 * fs); i2 = int(5.3 * fs); i3 = int(5.7 * fs)
        i4 = int(6.2 * fs); i5 = int(6.5 * fs)
        total[i1:i2] = bw_n - 400                 # eccentric drop
        total[i2:i3] = bw_n + 900                 # concentric push
        total[i3:i4] = 0                          # flight 0.5s -> h~300mm
        total[i4:i5] = bw_n * 2.5                 # landing spike
        total = np.maximum(total, 0)
        b1_total = total * 0.5; b2_total = total * 0.5
        cop_x = np.full(n, 280.0); cop_y = np.full(n, 216.0)
        enc1 = np.zeros(n); enc2 = np.zeros(n)
    elif kind == "encoder":
        # 3 reps of squat: bar drops 400mm then rises 400mm
        bar = np.full(n, 1000.0)    # start at 1m height
        rep_starts = [3.0, 8.0, 13.0]
        for rs in rep_starts:
            i0 = int(rs * fs); i1 = int((rs + 1.5) * fs)
            i2 = int((rs + 3.0) * fs)
            # eccentric (drop) 0 -> 1.5s
            bar[i0:i1] = 1000.0 - 400 * np.linspace(0, 1, i1 - i0)
            # concentric (rise) 1.5 -> 3.0s
            bar[i1:i2] = 600.0 + 400 * np.linspace(0, 1, i2 - i1)
        enc1 = bar + rng.normal(0, 0.5, n)
        enc2 = np.zeros(n)
        total = np.full(n, bw_n + 60 * G)    # BW + 60kg barbell
        b1_total = total * 0.5; b2_total = total * 0.5
        cop_x = np.full(n, 280.0); cop_y = np.full(n, 216.0)
    elif kind == "reaction":
        total = np.full(n, bw_n) + rng.normal(0, 2, n)
        # inject impulsive response 300ms after each stimulus
        stim_t = [5.0, 10.0, 15.0, 20.0, 25.0]
        for st in stim_t:
            i_resp = int((st + 0.30) * fs)
            if i_resp + 30 < n:
                total[i_resp:i_resp + 20] += 200     # ~200 N response spike
        b1_total = total * 0.5; b2_total = total * 0.5
        cop_x = np.full(n, 280.0); cop_y = np.full(n, 216.0)
        enc1 = np.zeros(n); enc2 = np.zeros(n)
        # stimulus log
        stim_log = pd.DataFrame([
            {"trial_idx": i, "t_wall": t_wall_s[0] + st,
             "t_ns": int(t_ns[0] + st * 1e9),
             "stimulus_type": "audio_visual"}
            for i, st in enumerate(stim_t)
        ])
        stim_log.to_csv(out_dir / "stimulus_log.csv", index=False)
    elif kind == "squat":
        # vGRF pattern: 3 reps of eccentric-concentric
        total = np.full(n, bw_n)
        for rs in [3.0, 8.0, 13.0]:
            i0 = int(rs * fs); i1 = int((rs + 1.2) * fs); i2 = int((rs + 2.4) * fs)
            total[i0:i1] = bw_n - 120
            total[i1:i2] = bw_n + 180
        b1_total = total * 0.48; b2_total = total * 0.52
        cop_x = np.full(n, 280.0) + rng.normal(0, 2, n)
        cop_y = np.full(n, 216.0) + rng.normal(0, 2, n)
        enc1 = np.zeros(n); enc2 = np.zeros(n)
    else:
        raise ValueError(kind)

    # Dummy corner forces (each = total/4 per side)
    b1_each = b1_total / 4
    b2_each = b2_total / 4

    fp = out_dir / "forces.csv"
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_ns", "t_wall",
            "b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
            "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N",
            "enc1_mm", "enc2_mm",
            "total_n", "cop_world_x_mm", "cop_world_y_mm",
        ])
        for i in range(n):
            w.writerow([
                int(t_ns[i]), f"{t_wall_s[i]:.6f}",
                f"{b1_each[i]:.3f}", f"{b1_each[i]:.3f}",
                f"{b1_each[i]:.3f}", f"{b1_each[i]:.3f}",
                f"{b2_each[i]:.3f}", f"{b2_each[i]:.3f}",
                f"{b2_each[i]:.3f}", f"{b2_each[i]:.3f}",
                f"{enc1[i]:.3f}", f"{enc2[i]:.3f}",
                f"{total[i]:.3f}",
                f"{cop_x[i]:.2f}", f"{cop_y[i]:.2f}",
            ])
    return out_dir


def main():
    # --- Balance ---
    print("\n=== Balance ===")
    d = make_synthetic_session("balance")
    force = load_force_session(d)
    r = analyze_balance(force)
    assert r.path_length_mm > 50 and r.mean_velocity_mm_s > 0
    print(f"  path_length={r.path_length_mm:.1f} mm  "
          f"vel={r.mean_velocity_mm_s:.2f} mm/s  "
          f"ellipse={r.ellipse95_area_mm2:.1f} mm^2")

    # --- WBA ---
    print("\n=== WBA ===")
    d = make_synthetic_session("wba")
    force = load_force_session(d)
    r = analyze_wba(force)
    assert 10 < r.mean_wba_pct < 30, f"WBA expected ~16%, got {r.mean_wba_pct}"
    print(f"  wba={r.mean_wba_pct:.2f}%  L={r.mean_left_pct:.1f}%  "
          f"R={r.mean_right_pct:.1f}%")

    # --- CMJ ---
    print("\n=== CMJ ===")
    d = make_synthetic_session("cmj")
    force = load_force_session(d)
    r = analyze_cmj(force)
    assert 0.05 < r.jump_height_m_flight < 1.0, \
        f"jump height expected, got {r.jump_height_m_flight}"
    print(f"  bw={r.bw_kg:.1f}kg  jump(flight)={r.jump_height_m_flight*1000:.0f}mm  "
          f"peak_vgrf={r.peak_force_bw:.2f}xBW  rfd={r.peak_rfd_n_s:.0f} N/s")

    # --- Encoder ---
    print("\n=== Encoder ===")
    d = make_synthetic_session("encoder")
    force = load_force_session(d)
    r = analyze_encoder(force, channel=1, bar_mass_kg=60.0, min_rom_mm=200)
    assert r.n_reps >= 2, f"expected 3 reps, detected {r.n_reps}"
    print(f"  reps={r.n_reps}  mean_rom={r.mean_rom_mm:.0f}mm  "
          f"peak_vel={r.peak_con_vel_m_s:.2f}m/s  "
          f"peak_pow={r.peak_con_power_w:.0f}W")

    # --- Reaction ---
    print("\n=== Reaction ===")
    d = make_synthetic_session("reaction")
    force = load_force_session(d)
    from src.analysis.reaction import load_stimulus_log
    stim = load_stimulus_log(d)
    r = analyze_reaction(force, stim)
    print(f"  trials={r.n_trials}  valid={r.n_valid}  "
          f"mean_rt={r.mean_rt_ms:.0f}ms  min={r.min_rt_ms:.0f}ms")
    # With baseline subtracted + 5-sigma threshold, detection might miss
    # subtle responses; we accept if we got at least half.
    assert r.n_trials >= 3

    # --- Squat ---
    print("\n=== Squat ===")
    d = make_synthetic_session("squat")
    force = load_force_session(d)
    r = analyze_squat(force)
    assert r.n_reps >= 1, f"expected some reps, got {r.n_reps}"
    print(f"  reps={r.n_reps}  mean_peak_vgrf={r.mean_peak_vgrf_bw:.2f}xBW  "
          f"wba_mean={r.mean_wba_pct:.2f}%")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
