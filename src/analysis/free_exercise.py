"""
Free-exercise (자유 운동 측정) analyzer.

Thin wrapper around the existing encoder rep-detection pipeline. The
operator names the exercise freely (e.g. "데드리프트", "푸쉬업") and sets a
load in kg; this analyzer runs the same per-rep ROM/velocity/power metrics
as the encoder test but tags the result with the exercise label and load so
the report viewer and history trends can group sessions by exercise name.

Load semantics
--------------
  - If session.json meta has ``use_bodyweight_load=True``, ``load_kg`` has
    already been overridden to the subject's bodyweight at record time
    (see RecorderConfig.__post_init__). This analyzer just uses ``load_kg``
    as-is for power computation.
  - If load_kg is 0 or missing, power fields will be 0 (velocity/ROM still
    valid). We still report them so the caller can see the zero and act.

Channel
-------
  Always uses encoder 1 (only the left encoder is currently active on this
  hardware — see config.ENCODER*_AVAILABLE). If channel 1 is flagged
  unavailable, analyze_encoder will emit a RuntimeWarning but still run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .common import load_force_session
from .encoder import EncoderResult, analyze_encoder


# Min ROM is looser than the encoder test default (150 mm) because free
# exercises include low-amplitude movements (push-ups, arm curls). Callers
# can override via analyze_free_exercise_file(min_rom_mm=...).
DEFAULT_MIN_ROM_MM = 80.0


@dataclass
class FreeExerciseResult:
    exercise_name: Optional[str]
    load_kg: float
    load_source: str            # "external" | "bodyweight" | "none"
    encoder: EncoderResult      # nested — full per-rep breakdown

    # Convenience scalars promoted from encoder — makes key_metrics.py
    # simpler and keeps the top-level dict flat for the report HTML.
    n_reps: int = 0
    duration_s: float = 0.0
    mean_rom_mm: float = 0.0
    mean_con_vel_m_s: float = 0.0
    peak_con_vel_m_s: float = 0.0
    mean_con_power_w: float = 0.0
    peak_con_power_w: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["encoder"] = self.encoder.to_dict()
        return d


def analyze_free_exercise_file(session_dir,
                               min_rom_mm: float = DEFAULT_MIN_ROM_MM,
                               ) -> FreeExerciseResult:
    """Load the session and run the encoder rep analyzer with the session's
    recorded load."""
    session_dir = Path(session_dir)

    # Pull exercise_name / load_kg from session.json (written by the
    # recorder with bodyweight already resolved).
    exercise_name: Optional[str] = None
    load_kg: float = 0.0
    use_bw: bool = False
    meta_path = session_dir / "session.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            exercise_name = meta.get("exercise_name")
            load_kg = float(meta.get("load_kg") or 0.0)
            use_bw  = bool(meta.get("use_bodyweight_load") or False)
        except Exception:
            pass

    load_source = ("bodyweight" if use_bw
                    else ("external" if load_kg > 0 else "none"))

    force = load_force_session(session_dir)
    enc = analyze_encoder(force, channel=1,
                          bar_mass_kg=max(0.0, load_kg),
                          min_rom_mm=float(min_rom_mm))

    return FreeExerciseResult(
        exercise_name=exercise_name,
        load_kg=float(load_kg),
        load_source=load_source,
        encoder=enc,
        n_reps=enc.n_reps,
        duration_s=enc.duration_s,
        mean_rom_mm=enc.mean_rom_mm,
        mean_con_vel_m_s=enc.mean_con_vel_m_s,
        peak_con_vel_m_s=enc.peak_con_vel_m_s,
        mean_con_power_w=enc.mean_con_power_w,
        peak_con_power_w=enc.peak_con_power_w,
    )
