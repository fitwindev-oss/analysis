"""
Cognitive reaction analysis (Phase V6).

The recorder shows positional cues (one of N/NE/E/.../NW) on screen and
the subject must reach the cued spot with a specified body part. We
extract three metrics from the per-camera 2D pose stream:

    reaction_time_ms      time from stimulus to motion onset
    movement_time_ms      time from motion onset to closest approach
    total_time_ms         RT + MT (subject lands on target)
    spatial_error_norm    distance between final body-part position and
                          the cued target, in image-normalised units
                          (0 = perfect hit, 1 = full image diagonal)
    hit                   spatial_error_norm <= ``hit_tolerance_norm``

Multi-cam sessions get cross-camera means; one camera is enough to run
the analyzer (the others just contribute to noise reduction).

stimulus_log.csv columns (V6+):
    trial_idx, t_wall, t_ns, stimulus_type, response_type,
    target_x_norm, target_y_norm, target_label

session.json keys consumed:
    cog_track_body_part   "right_hand" | "left_hand" | "right_foot" | "left_foot"
    cog_n_positions       4 | 8
    cog_positions         [[label, x_norm, y_norm], ...]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import ForceSession, load_force_session
from .pose2d import (
    Pose2DSeries, load_session_pose2d, resolve_pose_frame,
)
from src.pose.mediapipe_backend import MP33


# ─────────────────────────────────────────────────────────────────────────────
# Body-part → MP33 keypoint name resolution
# ─────────────────────────────────────────────────────────────────────────────

# Each tracked body part can map to multiple MP33 keypoints; the first
# one with valid data wins. ``right_hand`` → wrist if visible, else index
# fingertip — the index is more distal but disappears more often.
BODY_PART_TO_KEYPOINTS: dict[str, list[str]] = {
    "right_hand": ["right_wrist", "right_index"],
    "left_hand":  ["left_wrist",  "left_index"],
    "right_foot": ["right_foot_index", "right_ankle"],
    "left_foot":  ["left_foot_index",  "left_ankle"],
}


def _kpt_index_for_body_part(body_part: str) -> int:
    """Return the primary MP33 keypoint index for ``body_part``.

    Raises ValueError if the body part is unknown — the recorder only
    emits the four canonical names so this should never fire from a
    well-formed session.
    """
    cands = BODY_PART_TO_KEYPOINTS.get(body_part)
    if not cands:
        raise ValueError(f"unknown body_part for cognitive_reaction: {body_part!r}")
    return MP33[cands[0]]


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CogTrial:
    trial_idx: int
    target_label: str
    target_x_norm: float
    target_y_norm: float
    t_stim_s: float
    # Per-cam diagnostics — list of dicts {cam_id, rt_ms, mt_ms, err_norm}
    per_cam: list = field(default_factory=list)
    # Cross-camera means (None if no cam produced valid metrics)
    rt_ms:               Optional[float] = None
    mt_ms:               Optional[float] = None
    total_ms:            Optional[float] = None
    spatial_error_norm:  Optional[float] = None
    hit: bool = False
    no_response: bool = False


@dataclass
class CognitiveReactionResult:
    n_trials: int
    n_valid: int
    n_no_response: int
    n_hit: int
    hit_rate_pct: float

    mean_rt_ms:    float
    median_rt_ms:  float
    std_rt_ms:     float
    min_rt_ms:     float
    max_rt_ms:     float

    mean_mt_ms:    float
    mean_total_ms: float

    mean_spatial_error_norm: float

    body_part:     str
    n_positions:   int

    # Per target-direction breakdown
    per_target: dict = field(default_factory=dict)
    trials:     list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trials"] = [asdict(t) for t in self.trials]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Per-cam trajectory analysis
# ─────────────────────────────────────────────────────────────────────────────

def _per_cam_metrics(
    pose: Pose2DSeries,
    session_dir: Path,
    t_stim_s: float,
    target_x_norm: float,
    target_y_norm: float,
    kpt_index: int,
    *,
    max_response_s: float,
    min_motion_speed_px: float,
    onset_baseline_s: float,
    onset_sigma: float,
) -> Optional[dict]:
    """Compute (rt_ms, mt_ms, err_norm) for one camera, or None if data
    is unusable (keypoint missing, target outside frame, etc.).

    Strategy:
      1. Map t_stim to a starting video-frame index using session timestamps.
      2. Read the body-part trajectory for ``max_response_s`` after stim.
      3. Detect motion onset = first frame where instantaneous speed
         exceeds (baseline_speed + onset_sigma * baseline_std).
      4. End-of-reach = frame of closest approach to the target after onset.
      5. Spatial error = euclidean distance(end_pos, target) in normalised
         image-diagonal units.
    """
    img_w, img_h = pose.image_size
    if img_w <= 0 or img_h <= 0:
        return None

    diag = float(np.hypot(img_w, img_h))
    target_px = np.array([target_x_norm * img_w, target_y_norm * img_h],
                         dtype=np.float32)

    fps = float(pose.fps) or 30.0
    i_stim = resolve_pose_frame(t_stim_s, session_dir, pose.cam_id, fps)
    n_post = max(2, int(max_response_s * fps))
    n_base = max(3, int(onset_baseline_s * fps))

    n_total = pose.kpts_mp33.shape[0]
    if i_stim >= n_total:
        return None

    # Baseline window — pre-stimulus speed (used for onset threshold).
    # If we don't have enough frames before stim (e.g. stim near t=0),
    # use a flat low default for the threshold.
    i_base0 = max(0, i_stim - n_base)
    pre_kpts = pose.kpts_mp33[i_base0:i_stim, kpt_index, :]
    if len(pre_kpts) >= 3:
        d_pre = np.diff(pre_kpts, axis=0)
        speed_pre = np.linalg.norm(d_pre, axis=1)
        speed_pre = speed_pre[~np.isnan(speed_pre)]
    else:
        speed_pre = np.array([], dtype=np.float32)

    if len(speed_pre) >= 3:
        bl_mean = float(np.mean(speed_pre))
        bl_std  = float(np.std(speed_pre))
        thr = max(min_motion_speed_px,
                  bl_mean + onset_sigma * bl_std)
    else:
        thr = float(min_motion_speed_px)

    # Post-stimulus window
    i_end = min(n_total, i_stim + n_post)
    post_kpts = pose.kpts_mp33[i_stim:i_end, kpt_index, :]
    if len(post_kpts) < 3:
        return None

    # Per-frame instantaneous speed (px / frame). NaN preserved so
    # invisible frames don't trigger false onsets.
    d_post = np.diff(post_kpts, axis=0)
    speed_post = np.linalg.norm(d_post, axis=1)

    onset_rel: Optional[int] = None
    for k, v in enumerate(speed_post):
        if np.isnan(v):
            continue
        if v >= thr:
            onset_rel = k
            break
    if onset_rel is None:
        return None

    # End of reach = closest approach to target on or after onset.
    # (NaN-bearing frames give NaN distance and are ignored by argmin.)
    valid = ~np.isnan(post_kpts).any(axis=1)
    if not valid.any():
        return None
    dists = np.full(len(post_kpts), np.inf, dtype=np.float32)
    dists[valid] = np.linalg.norm(
        post_kpts[valid] - target_px, axis=1)
    # Restrict the search to onset onward so a noisy pre-onset frame
    # accidentally near the target can't win.
    dists_after = np.full_like(dists, np.inf)
    dists_after[onset_rel:] = dists[onset_rel:]
    end_rel = int(np.argmin(dists_after))
    if not np.isfinite(dists_after[end_rel]):
        return None

    rt_ms = onset_rel / fps * 1000.0
    mt_ms = max(0, (end_rel - onset_rel)) / fps * 1000.0
    err_px = float(dists[end_rel])
    err_norm = err_px / diag

    return {
        "cam_id":   pose.cam_id,
        "rt_ms":    float(rt_ms),
        "mt_ms":    float(mt_ms),
        "err_norm": float(err_norm),
        "onset_frame": int(i_stim + onset_rel + 1),
        "end_frame":   int(i_stim + end_rel),
        "fps":      float(fps),
    }


def _aggregate_cam_metrics(per_cam: list[dict]) -> dict:
    """Mean RT/MT/err across cameras. Empty list → NaNs."""
    if not per_cam:
        return {"rt_ms": None, "mt_ms": None, "err_norm": None}
    rt = [c["rt_ms"]    for c in per_cam if c.get("rt_ms")    is not None]
    mt = [c["mt_ms"]    for c in per_cam if c.get("mt_ms")    is not None]
    er = [c["err_norm"] for c in per_cam if c.get("err_norm") is not None]
    return {
        "rt_ms":    float(np.mean(rt)) if rt else None,
        "mt_ms":    float(np.mean(mt)) if mt else None,
        "err_norm": float(np.mean(er)) if er else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def load_stimulus_log(session_dir: Path) -> pd.DataFrame:
    fp = Path(session_dir) / "stimulus_log.csv"
    if not fp.exists():
        raise FileNotFoundError(f"missing stimulus log: {fp}")
    return pd.read_csv(fp)


def _read_session_meta(session_dir: Path) -> dict:
    p = Path(session_dir) / "session.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def analyze_cognitive_reaction(
    force: ForceSession,
    stimuli: pd.DataFrame,
    *,
    body_part: str = "right_hand",
    n_positions: int = 4,
    max_response_s: float = 1.5,
    min_motion_speed_px: float = 4.0,
    onset_baseline_s: float = 0.4,
    onset_sigma: float = 3.0,
    hit_tolerance_norm: float = 0.12,
) -> CognitiveReactionResult:
    """Run the cognitive-reaction analysis over one session.

    ``force`` is required only for time-base alignment (record_start_wall).
    All actual metrics come from the per-cam pose streams.
    """
    session_dir = Path(force.session_dir)
    pose_by_cam = load_session_pose2d(session_dir)

    # Resolve body part once
    try:
        kpt_index = _kpt_index_for_body_part(body_part)
    except ValueError:
        # Unknown body part → fall back to right_wrist; flag in result via
        # ``body_part`` field so the report shows what we actually used.
        body_part = "right_hand"
        kpt_index = MP33["right_wrist"]

    # Stim time alignment — match the reaction.py protocol
    if "t_wall" in stimuli.columns:
        t_stim_wall = stimuli["t_wall"].to_numpy(np.float64)
        forces_file = session_dir / "forces.csv"
        df_force = pd.read_csv(forces_file, usecols=["t_wall"], nrows=1)
        t0_wall = float(df_force["t_wall"].iloc[0])
        t_stim_arr = t_stim_wall - t0_wall
    elif "t_ns" in stimuli.columns:
        t_stim_arr = stimuli["t_ns"].to_numpy(np.int64) / 1e9
    else:
        raise RuntimeError("stimulus log needs t_wall or t_ns column")

    # Target XY: prefer the inline columns written by V6+ recordings;
    # fall back to looking up by response_type via session.json positions.
    if {"target_x_norm", "target_y_norm", "target_label"}.issubset(
            stimuli.columns):
        tx_arr = stimuli["target_x_norm"].to_numpy(np.float64)
        ty_arr = stimuli["target_y_norm"].to_numpy(np.float64)
        labels = stimuli["target_label"].fillna("").to_numpy()
    else:
        meta = _read_session_meta(session_dir)
        pos_table = meta.get("cog_positions") or []
        lookup = {row[0]: (float(row[1]), float(row[2])) for row in pos_table}
        resp = stimuli.get("response_type",
                           pd.Series([""] * len(stimuli))).to_numpy()
        tx_arr = np.array([lookup.get(r, (np.nan, np.nan))[0] for r in resp])
        ty_arr = np.array([lookup.get(r, (np.nan, np.nan))[1] for r in resp])
        labels = resp

    trials: list[CogTrial] = []
    for i, ts in enumerate(t_stim_arr):
        tx = float(tx_arr[i]) if i < len(tx_arr) else float("nan")
        ty = float(ty_arr[i]) if i < len(ty_arr) else float("nan")
        label = str(labels[i]) if i < len(labels) else ""

        per_cam: list[dict] = []
        if not (np.isnan(tx) or np.isnan(ty)):
            for ps in pose_by_cam.values():
                m = _per_cam_metrics(
                    ps, session_dir, float(ts), tx, ty, kpt_index,
                    max_response_s=max_response_s,
                    min_motion_speed_px=min_motion_speed_px,
                    onset_baseline_s=onset_baseline_s,
                    onset_sigma=onset_sigma,
                )
                if m is not None:
                    per_cam.append(m)
        agg = _aggregate_cam_metrics(per_cam)

        rt   = agg.get("rt_ms")
        mt   = agg.get("mt_ms")
        err  = agg.get("err_norm")
        no_resp = (rt is None)
        total = (rt + mt) if (rt is not None and mt is not None) else None
        hit  = bool(err is not None and err <= hit_tolerance_norm)

        trials.append(CogTrial(
            trial_idx=i,
            target_label=label,
            target_x_norm=tx,
            target_y_norm=ty,
            t_stim_s=float(ts),
            per_cam=per_cam,
            rt_ms=rt, mt_ms=mt, total_ms=total,
            spatial_error_norm=err,
            hit=hit,
            no_response=no_resp,
        ))

    # Aggregates
    valid = [t for t in trials if not t.no_response]
    rts = np.array([t.rt_ms for t in valid if t.rt_ms is not None],
                   dtype=np.float64)
    mts = np.array([t.mt_ms for t in valid if t.mt_ms is not None],
                   dtype=np.float64)
    tots = np.array([t.total_ms for t in valid if t.total_ms is not None],
                    dtype=np.float64)
    errs = np.array([t.spatial_error_norm for t in valid
                     if t.spatial_error_norm is not None], dtype=np.float64)

    if len(rts) == 0:
        mean_rt = median_rt = std_rt = min_rt = max_rt = float("nan")
    else:
        mean_rt   = float(rts.mean())
        median_rt = float(np.median(rts))
        std_rt    = float(rts.std())
        min_rt    = float(rts.min())
        max_rt    = float(rts.max())

    mean_mt    = float(mts.mean())  if len(mts)  else float("nan")
    mean_total = float(tots.mean()) if len(tots) else float("nan")
    mean_err   = float(errs.mean()) if len(errs) else float("nan")

    n_hit = sum(1 for t in valid if t.hit)
    hit_rate_pct = (100.0 * n_hit / len(valid)) if valid else 0.0

    # Per-target breakdown
    per_target: dict = {}
    by_label: dict[str, list[CogTrial]] = {}
    for t in valid:
        by_label.setdefault(t.target_label or "unknown", []).append(t)
    for key, group in by_label.items():
        rt_vals  = [x.rt_ms              for x in group if x.rt_ms is not None]
        mt_vals  = [x.mt_ms              for x in group if x.mt_ms is not None]
        err_vals = [x.spatial_error_norm for x in group
                    if x.spatial_error_norm is not None]
        n_hit_grp = sum(1 for x in group if x.hit)
        per_target[key] = {
            "n":              int(len(group)),
            "n_hit":          int(n_hit_grp),
            "mean_rt_ms":     float(np.mean(rt_vals))  if rt_vals  else float("nan"),
            "mean_mt_ms":     float(np.mean(mt_vals))  if mt_vals  else float("nan"),
            "mean_err_norm":  float(np.mean(err_vals)) if err_vals else float("nan"),
        }

    return CognitiveReactionResult(
        n_trials=len(trials),
        n_valid=len(valid),
        n_no_response=sum(1 for t in trials if t.no_response),
        n_hit=n_hit,
        hit_rate_pct=float(hit_rate_pct),
        mean_rt_ms=mean_rt,
        median_rt_ms=median_rt,
        std_rt_ms=std_rt,
        min_rt_ms=min_rt,
        max_rt_ms=max_rt,
        mean_mt_ms=mean_mt,
        mean_total_ms=mean_total,
        mean_spatial_error_norm=mean_err,
        body_part=body_part,
        n_positions=int(n_positions),
        per_target=per_target,
        trials=trials,
    )


def analyze_cognitive_reaction_file(session_dir, **kw) -> CognitiveReactionResult:
    """Convenience wrapper that loads everything from a session folder.

    Reads body_part / n_positions from session.json so callers don't
    need to pass them explicitly.
    """
    session_dir = Path(session_dir)
    force = load_force_session(session_dir)
    stim = load_stimulus_log(session_dir)
    meta = _read_session_meta(session_dir)
    body_part = kw.pop(
        "body_part", meta.get("cog_track_body_part") or "right_hand")
    n_positions = kw.pop(
        "n_positions", int(meta.get("cog_n_positions") or 4))
    return analyze_cognitive_reaction(
        force, stim, body_part=body_part, n_positions=n_positions, **kw)
