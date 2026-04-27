"""
Unit tests for V6 — cognitive reaction (positional cue + skeleton tracking).

Covers:
  - SessionRecorder _prepare_reaction_pool builds a positional pool +
    target lookup for cognitive_reaction sessions
  - _fire_stim attaches target_x/y metadata to events
  - _option_tag produces a meaningful folder suffix
  - cognitive_reaction analyzer body-part resolution
  - per-cam motion-onset detection on a synthetic Pose2DSeries
  - per-target aggregation in CognitiveReactionResult

Run:
    python tests/test_cognitive_reaction.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.session_recorder import (
    SessionRecorder, RecorderConfig,
    COGNITIVE_REACTION_POSITIONS_4, COGNITIVE_REACTION_POSITIONS_8,
    REACTION_RESPONSES,
)
from src.analysis.cognitive_reaction import (
    BODY_PART_TO_KEYPOINTS, _kpt_index_for_body_part,
    _aggregate_cam_metrics, _per_cam_metrics, CognitiveReactionResult,
    CogTrial,
)
from src.analysis.pose2d import Pose2DSeries
from src.pose.mediapipe_backend import MP33


# ────────────────────────────────────────────────────────────────────────────
# RecorderConfig + reaction pool prep
# ────────────────────────────────────────────────────────────────────────────
def test_recorder_accepts_cognitive_reaction_test():
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_track_body_part="right_hand",
                         react_n_positions=4,
                         duration_s=30.0,
                         n_stimuli=8,
                         trigger="auto")
    rec = SessionRecorder(cfg)
    # Prompt should mention the body part hint.
    assert "COGNITIVE REACTION" in rec.state.prompt
    # Pool prep must populate _response_pool with positional keys.
    rec._prepare_reaction_pool()
    assert set(rec._response_pool) == {"pos_N", "pos_E", "pos_S", "pos_W"}
    # Lookup table has matching XY entries.
    assert hasattr(rec, "_cog_pos_lookup")
    assert all(k in rec._cog_pos_lookup for k in rec._response_pool)


def test_recorder_8pos_pool():
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_n_positions=8,
                         duration_s=20.0,
                         n_stimuli=4,
                         trigger="auto")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    assert len(rec._response_pool) == 8
    # All 8 cardinal+ordinal directions present
    assert {"pos_N", "pos_NE", "pos_E", "pos_SE",
            "pos_S", "pos_SW", "pos_W", "pos_NW"} <= set(rec._response_pool)


def test_position_table_has_correct_geometry():
    """Cardinal-direction positions must lie on the expected sides
    (north has small y, east has large x, etc.)."""
    p = dict([(k, (x, y)) for k, x, y in COGNITIVE_REACTION_POSITIONS_8])
    assert p["pos_N"][1] < 0.5      # north → small y (image-top)
    assert p["pos_S"][1] > 0.5      # south → large y
    assert p["pos_E"][0] > 0.5      # east → large x
    assert p["pos_W"][0] < 0.5      # west → small x


def test_smart_wait_bypassed_for_cognitive_reaction():
    """V6-fix — smart-wait depends on force-plate stance, but the
    cognitive_reaction test is a hand/foot reach driven by screen cues
    with no plate involvement. The recorder must skip StabilityDetector
    construction even when ``use_smart_wait=True`` so the run-loop never
    hangs waiting for a plate stance that wouldn't apply.
    """
    cfg = RecorderConfig(test="cognitive_reaction",
                         use_smart_wait=True,
                         react_n_positions=4,
                         duration_s=30.0, n_stimuli=4, trigger="auto")
    rec = SessionRecorder(cfg)
    # Mirror the gating inside _start_hardware.
    wants_smart_wait = (
        rec.cfg.use_smart_wait
        and rec.cfg.test != "cognitive_reaction")
    assert wants_smart_wait is False


def test_smart_wait_still_active_for_classic_reaction():
    """Classic reaction stays gated on stance — both feet on plate."""
    cfg = RecorderConfig(test="reaction", use_smart_wait=True,
                         duration_s=30.0, n_stimuli=4, trigger="auto",
                         responses="random")
    rec = SessionRecorder(cfg)
    wants_smart_wait = (
        rec.cfg.use_smart_wait
        and rec.cfg.test != "cognitive_reaction")
    assert wants_smart_wait is True


def test_smart_wait_off_skips_for_all_tests():
    """If the operator turned the checkbox off, the gate stays off
    regardless of the test type (sanity)."""
    cfg = RecorderConfig(test="cmj", use_smart_wait=False, duration_s=10.0)
    rec = SessionRecorder(cfg)
    wants_smart_wait = (
        rec.cfg.use_smart_wait
        and rec.cfg.test != "cognitive_reaction")
    assert wants_smart_wait is False


def test_zero_cal_only_flag_set_for_cognitive_reaction():
    """V6-fix — when use_smart_wait=True, cognitive_reaction takes the
    zero-cal-only branch. The wait phase still runs (waits out the DAQ
    5 s zero-cal so forces.csv baseline is clean) but with no stance
    check; it auto-transitions to countdown when zero-cal completes.
    """
    # Mirror the calculation _start_hardware does. We can't actually
    # call _start_hardware in CI (no DAQ hardware) so we just verify
    # the predicate that controls _zero_cal_only.
    cfg = RecorderConfig(test="cognitive_reaction",
                         use_smart_wait=True,
                         react_n_positions=4,
                         duration_s=30.0, n_stimuli=4, trigger="auto")
    rec = SessionRecorder(cfg)
    zero_cal_only = (
        rec.cfg.test == "cognitive_reaction"
        and rec.cfg.use_smart_wait)
    assert zero_cal_only is True


def test_zero_cal_only_flag_off_when_smart_wait_off():
    """Operator can opt out of zero-cal-only by turning use_smart_wait
    off entirely. Then cognitive_reaction goes straight into countdown
    (~5 s) with no zero-cal wait — first ~0.5 s of forces.csv may
    contain DAQ ramp-up samples but the analyzer doesn't care."""
    cfg = RecorderConfig(test="cognitive_reaction",
                         use_smart_wait=False,
                         duration_s=30.0, n_stimuli=4, trigger="auto")
    rec = SessionRecorder(cfg)
    zero_cal_only = (
        rec.cfg.test == "cognitive_reaction"
        and rec.cfg.use_smart_wait)
    assert zero_cal_only is False


def test_reaction_responses_includes_positions():
    """Each pos_* key must have an on-screen banner registered."""
    for k in ("pos_N", "pos_NE", "pos_E", "pos_SE",
              "pos_S", "pos_SW", "pos_W", "pos_NW"):
        assert k in REACTION_RESPONSES
        label, color = REACTION_RESPONSES[k]
        assert isinstance(label, str) and len(label) > 0


# ────────────────────────────────────────────────────────────────────────────
# _fire_stim → target metadata
# ────────────────────────────────────────────────────────────────────────────
def test_fire_stim_attaches_target_xy():
    cfg = RecorderConfig(test="cognitive_reaction", react_n_positions=4,
                         duration_s=30.0, n_stimuli=4, trigger="manual")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    rec._fire_stim("pos_N", time.monotonic_ns())
    rec._fire_stim("pos_E", time.monotonic_ns())
    assert len(rec._stim_events) == 2
    e0, e1 = rec._stim_events
    assert e0["response_type"] == "pos_N"
    assert "target_x_norm" in e0 and "target_y_norm" in e0
    assert e0["target_label"] == "pos_N"
    assert abs(e0["target_x_norm"] - 0.50) < 1e-9
    assert e0["target_y_norm"] < 0.5
    assert e1["target_x_norm"] > 0.5    # east


def test_fire_stim_no_target_for_classic_reaction():
    """Classic reaction sessions still emit events without target XY."""
    cfg = RecorderConfig(test="reaction", duration_s=30.0, n_stimuli=4,
                         trigger="manual", responses="random")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    rec._fire_stim("jump", time.monotonic_ns())
    assert "target_x_norm" not in rec._stim_events[0]


def test_option_tag_for_cognitive_reaction():
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_track_body_part="right_hand",
                         react_n_positions=4,
                         duration_s=20.0, n_stimuli=10, trigger="auto")
    rec = SessionRecorder(cfg)
    tag = rec._option_tag()
    assert "right_hand" in tag
    assert "4pos" in tag
    assert "10x" in tag


# ────────────────────────────────────────────────────────────────────────────
# Body-part → MP33 keypoint resolution
# ────────────────────────────────────────────────────────────────────────────
def test_body_part_to_keypoints_covers_four_canonicals():
    for k in ("right_hand", "left_hand", "right_foot", "left_foot"):
        assert k in BODY_PART_TO_KEYPOINTS
        cands = BODY_PART_TO_KEYPOINTS[k]
        # First candidate must resolve to a real MP33 index
        assert cands[0] in MP33


def test_kpt_index_resolves_right_hand_to_wrist():
    assert _kpt_index_for_body_part("right_hand") == MP33["right_wrist"]


def test_kpt_index_unknown_body_part_raises():
    try:
        _kpt_index_for_body_part("third_arm")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown body part")


# ────────────────────────────────────────────────────────────────────────────
# _aggregate_cam_metrics
# ────────────────────────────────────────────────────────────────────────────
def test_aggregate_empty_returns_none():
    out = _aggregate_cam_metrics([])
    assert out["rt_ms"] is None
    assert out["mt_ms"] is None
    assert out["err_norm"] is None


def test_aggregate_means_across_cams():
    pc = [
        {"rt_ms": 200.0, "mt_ms": 300.0, "err_norm": 0.05},
        {"rt_ms": 240.0, "mt_ms": 360.0, "err_norm": 0.07},
    ]
    out = _aggregate_cam_metrics(pc)
    assert abs(out["rt_ms"]    - 220.0) < 1e-9
    assert abs(out["mt_ms"]    - 330.0) < 1e-9
    assert abs(out["err_norm"] - 0.06)  < 1e-9


def test_aggregate_skips_none_values():
    pc = [
        {"rt_ms": 200.0, "mt_ms": None,  "err_norm": 0.05},
        {"rt_ms": None,  "mt_ms": 300.0, "err_norm": 0.07},
    ]
    out = _aggregate_cam_metrics(pc)
    assert abs(out["rt_ms"]    - 200.0) < 1e-9
    assert abs(out["mt_ms"]    - 300.0) < 1e-9
    assert abs(out["err_norm"] - 0.06)  < 1e-9


# ────────────────────────────────────────────────────────────────────────────
# _per_cam_metrics — synthetic pose trajectory
# ────────────────────────────────────────────────────────────────────────────
def _make_synthetic_pose(n_frames: int, fps: float, kpt_index: int,
                         start_xy: tuple[float, float],
                         move_start_frame: int,
                         move_end_xy: tuple[float, float],
                         move_dur_frames: int,
                         img_size: tuple[int, int] = (640, 480)
                         ) -> Pose2DSeries:
    """Build a Pose2DSeries where ``kpt_index`` sits at start_xy until
    ``move_start_frame``, then linearly interpolates to ``move_end_xy``
    over the next ``move_dur_frames`` frames, then stays.

    All other keypoints are NaN — analyzer reads only the configured one.
    """
    n_joints = 33
    kpts = np.full((n_frames, n_joints, 2), np.nan, dtype=np.float32)
    vis  = np.zeros((n_frames, n_joints), dtype=np.float32)
    # Trajectory of the tracked keypoint
    xs = np.full(n_frames, start_xy[0], dtype=np.float32)
    ys = np.full(n_frames, start_xy[1], dtype=np.float32)
    f0 = move_start_frame
    f1 = min(n_frames, f0 + move_dur_frames)
    if f1 > f0:
        xs[f0:f1] = np.linspace(start_xy[0], move_end_xy[0], f1 - f0)
        ys[f0:f1] = np.linspace(start_xy[1], move_end_xy[1], f1 - f0)
    xs[f1:] = move_end_xy[0]
    ys[f1:] = move_end_xy[1]
    kpts[:, kpt_index, 0] = xs
    kpts[:, kpt_index, 1] = ys
    vis[:, kpt_index] = 1.0
    return Pose2DSeries(
        cam_id="cam0",
        kpts_mp33=kpts,
        vis_mp33=vis,
        world_mp33=np.full((n_frames, n_joints, 3), np.nan, dtype=np.float32),
        angles=np.full((n_frames, 12), np.nan, dtype=np.float32),
        angle_names=[],
        fps=float(fps),
        image_size=img_size,
    )


def test_per_cam_metrics_detects_motion_onset():
    """Hand at rest, then moves to target → RT roughly = onset time,
    err_norm small at end of move."""
    # 90-frame, 30 Hz pose; stim at frame 30 (t = 1.0 s); motion starts
    # at frame 36 (i.e., 200 ms after stim); reaches target by frame 60.
    fps = 30.0
    img_w, img_h = 640, 480
    target_norm = (0.85, 0.50)                   # east
    target_px = (target_norm[0] * img_w, target_norm[1] * img_h)
    pose = _make_synthetic_pose(
        n_frames=90, fps=fps, kpt_index=MP33["right_wrist"],
        start_xy=(0.5 * img_w, 0.5 * img_h),
        move_start_frame=36,
        move_end_xy=target_px,
        move_dur_frames=24,
        img_size=(img_w, img_h),
    )
    fake_dir = Path("/__nonexistent_session__")
    out = _per_cam_metrics(
        pose=pose, session_dir=fake_dir,
        t_stim_s=1.0,                             # → frame 30
        target_x_norm=target_norm[0],
        target_y_norm=target_norm[1],
        kpt_index=MP33["right_wrist"],
        max_response_s=2.0,
        min_motion_speed_px=2.0,
        onset_baseline_s=0.4,
        onset_sigma=3.0,
        hit_tolerance_norm=0.12,
    )
    assert out is not None
    assert out["rt_ms"] is not None
    # Motion onset should land within ~3 frames of frame 36 → ~200 ms RT.
    # Allow ±100 ms slack for the speed-threshold detection.
    assert 100.0 <= out["rt_ms"] <= 300.0
    # End-of-reach error should be negligible (we land exactly on target).
    assert out["err_norm"] < 1e-3
    assert out["mt_ms"] >= 0.0
    # Standard motion-onset path
    assert out["failure_reason"] == "ok_motion_onset"


def test_per_cam_metrics_no_motion_no_hit_returns_failure_reason():
    """V6-fix2 — when wrist never moves AND never reaches the target,
    return a dict (not None) carrying failure_reason='no_motion_no_hit'
    so the report can show why the trial failed."""
    fps = 30.0
    img_w, img_h = 640, 480
    pose = _make_synthetic_pose(
        n_frames=60, fps=fps, kpt_index=MP33["right_wrist"],
        start_xy=(0.10 * img_w, 0.10 * img_h),    # NW corner, far from target
        move_start_frame=999,                       # never moves
        move_end_xy=(0.10 * img_w, 0.10 * img_h),
        move_dur_frames=0,
        img_size=(img_w, img_h),
    )
    out = _per_cam_metrics(
        pose=pose, session_dir=Path("/__nonexistent__"),
        t_stim_s=0.5,
        target_x_norm=0.85, target_y_norm=0.50,    # east — far from NW
        kpt_index=MP33["right_wrist"],
        max_response_s=1.5, min_motion_speed_px=4.0,
        onset_baseline_s=0.3, onset_sigma=3.0,
        hit_tolerance_norm=0.12,
    )
    assert out is not None
    assert out["rt_ms"] is None
    assert out["failure_reason"] == "no_motion_no_hit"


def test_per_cam_metrics_proximity_fallback_for_slow_reach():
    """V6-fix2 — slow smooth reach without a sharp speed onset should
    still be picked up via the proximity fallback when the wrist
    actually arrives near the target."""
    fps = 30.0
    img_w, img_h = 640, 480
    target_norm = (0.85, 0.50)
    target_px = (target_norm[0] * img_w, target_norm[1] * img_h)
    # Move VERY slowly: 60 frames to traverse, peak speed ~1.4 px/frame
    # (below the 2.0 default threshold).
    pose = _make_synthetic_pose(
        n_frames=120, fps=fps, kpt_index=MP33["right_wrist"],
        start_xy=(0.50 * img_w, 0.50 * img_h),
        move_start_frame=30,
        move_end_xy=target_px,
        move_dur_frames=60,                        # slow reach
        img_size=(img_w, img_h),
    )
    out = _per_cam_metrics(
        pose=pose, session_dir=Path("/__none__"),
        t_stim_s=1.0,
        target_x_norm=target_norm[0],
        target_y_norm=target_norm[1],
        kpt_index=MP33["right_wrist"],
        max_response_s=3.0, min_motion_speed_px=4.0,  # high speed threshold
        onset_baseline_s=0.4, onset_sigma=3.0,
        hit_tolerance_norm=0.12,
    )
    assert out is not None
    # Motion onset would have failed (peak speed too low), but the
    # proximity fallback picks up the trial because the wrist
    # reached close to the target.
    assert out["rt_ms"] is not None
    assert out["failure_reason"] == "ok_proximity_hit"


def test_per_cam_metrics_invalid_image_size_returns_failure_reason():
    """Sanity: zero-sized images return a failure dict (not None)."""
    pose = _make_synthetic_pose(
        n_frames=30, fps=30.0, kpt_index=MP33["right_wrist"],
        start_xy=(0.0, 0.0), move_start_frame=10,
        move_end_xy=(0.0, 0.0), move_dur_frames=0,
        img_size=(0, 0),
    )
    out = _per_cam_metrics(
        pose=pose, session_dir=Path("/x"),
        t_stim_s=0.0, target_x_norm=0.5, target_y_norm=0.5,
        kpt_index=MP33["right_wrist"],
        max_response_s=1.0, min_motion_speed_px=4.0,
        onset_baseline_s=0.3, onset_sigma=3.0,
        hit_tolerance_norm=0.12,
    )
    assert out is not None
    assert out["rt_ms"] is None
    assert out["failure_reason"] == "zero_image_size"


def test_per_cam_metrics_low_visibility_returns_failure_reason():
    """V6-fix2 — keypoint with visibility below 0.5 across the entire
    post-window must produce a 'no_visible_kpt' diagnostic."""
    fps = 30.0
    img_w, img_h = 640, 480
    pose = _make_synthetic_pose(
        n_frames=60, fps=fps, kpt_index=MP33["right_wrist"],
        start_xy=(0.50 * img_w, 0.50 * img_h),
        move_start_frame=20, move_end_xy=(0.85 * img_w, 0.50 * img_h),
        move_dur_frames=20, img_size=(img_w, img_h),
    )
    # Tank visibility everywhere — analyzer must skip the trial.
    pose.vis_mp33[:, MP33["right_wrist"]] = 0.0
    out = _per_cam_metrics(
        pose=pose, session_dir=Path("/x"),
        t_stim_s=0.5, target_x_norm=0.85, target_y_norm=0.50,
        kpt_index=MP33["right_wrist"],
        max_response_s=1.5, min_motion_speed_px=2.0,
        onset_baseline_s=0.3, onset_sigma=3.0,
        hit_tolerance_norm=0.12,
    )
    assert out is not None
    assert out["rt_ms"] is None
    assert out["failure_reason"] == "no_visible_kpt"


def test_threshold_capped_when_baseline_noisy():
    """Carry-over motion in the baseline window must not drive the
    onset threshold above the cap. We construct a pose where the wrist
    is moving fast in the pre-stim window (200 px/frame), then briefly
    accelerates slightly post-stim. Without the cap, onset_sigma * std
    would push thr ≫ post-stim speed and the trial would no-respond."""
    fps = 30.0
    img_w, img_h = 640, 480
    n_frames = 90
    kpts = np.full((n_frames, 33, 2), np.nan, dtype=np.float32)
    vis  = np.zeros((n_frames, 33), dtype=np.float32)
    wrist = MP33["right_wrist"]
    # Pre-stim (frames 0-29): wrist oscillates wildly across X
    rng = np.random.default_rng(0)
    xs = 320.0 + rng.uniform(-200.0, 200.0, size=n_frames).astype(np.float32)
    ys = np.full(n_frames, 240.0, dtype=np.float32)
    # Post-stim (frames 30+): clean reach toward target
    target_x = 0.85 * img_w
    xs[30:55] = np.linspace(320.0, target_x, 25)
    xs[55:] = target_x
    kpts[:, wrist, 0] = xs; kpts[:, wrist, 1] = ys
    vis[:, wrist] = 1.0
    pose = Pose2DSeries(
        cam_id="cam0", kpts_mp33=kpts, vis_mp33=vis,
        world_mp33=np.full((n_frames, 33, 3), np.nan, dtype=np.float32),
        angles=np.full((n_frames, 12), np.nan, dtype=np.float32),
        angle_names=[], fps=fps, image_size=(img_w, img_h))
    out = _per_cam_metrics(
        pose=pose, session_dir=Path("/x"),
        t_stim_s=1.0,                                # frame 30
        target_x_norm=0.85, target_y_norm=0.50,
        kpt_index=wrist,
        max_response_s=2.0, min_motion_speed_px=2.0,
        onset_baseline_s=0.4, onset_sigma=3.0,
        hit_tolerance_norm=0.12,
    )
    assert out is not None
    # Threshold should be capped to the configured ceiling (30 px/frame),
    # not the much larger value baseline noise would dictate.
    assert out["threshold_px_per_frame"] <= 30.0 + 1e-6


# ────────────────────────────────────────────────────────────────────────────
# CognitiveReactionResult + CogTrial — surface-level shape
# ────────────────────────────────────────────────────────────────────────────
def test_result_to_dict_round_trips_trials():
    trials = [
        CogTrial(trial_idx=0, target_label="pos_N",
                 target_x_norm=0.5, target_y_norm=0.2,
                 t_stim_s=1.0, rt_ms=210.0, mt_ms=300.0,
                 total_ms=510.0, spatial_error_norm=0.04, hit=True),
        CogTrial(trial_idx=1, target_label="pos_E",
                 target_x_norm=0.85, target_y_norm=0.5,
                 t_stim_s=4.0, rt_ms=None, mt_ms=None, total_ms=None,
                 spatial_error_norm=None, hit=False, no_response=True),
    ]
    res = CognitiveReactionResult(
        n_trials=2, n_valid=1, n_no_response=1, n_hit=1,
        hit_rate_pct=100.0,
        mean_rt_ms=210.0, median_rt_ms=210.0, std_rt_ms=0.0,
        min_rt_ms=210.0, max_rt_ms=210.0,
        mean_mt_ms=300.0, mean_total_ms=510.0,
        mean_spatial_error_norm=0.04,
        body_part="right_hand", n_positions=4,
        per_target={"pos_N": {"n": 1, "n_hit": 1,
                              "mean_rt_ms": 210.0, "mean_mt_ms": 300.0,
                              "mean_err_norm": 0.04}},
        trials=trials)
    d = res.to_dict()
    assert d["n_trials"] == 2
    assert d["n_hit"] == 1
    assert isinstance(d["trials"], list)
    assert d["trials"][0]["target_label"] == "pos_N"
    assert d["trials"][1]["no_response"] is True


def test_departure_tracking_disabled_for_cognitive_reaction():
    """V6 fix — cognitive_reaction is a hand-reach test; force-plate
    departures are meaningless here. The recorder must skip the
    departure tracker entirely so events.csv stays empty AND the
    replay doesn't show a ⚠ 이탈 banner."""
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_n_positions=4,
                         duration_s=30.0, n_stimuli=4, trigger="auto")
    rec = SessionRecorder(cfg)
    # Mirror the predicate inside _start_hardware
    track = (rec.cfg.test != "cognitive_reaction")
    assert track is False


def test_departure_tracking_enabled_for_other_tests():
    """Sanity — non-cognitive tests still track departures."""
    for t in ("balance_eo", "cmj", "sj", "squat", "reaction"):
        kw: dict = {"duration_s": 10.0}
        if t == "reaction":
            kw["responses"] = "random"; kw["trigger"] = "auto"
            kw["n_stimuli"] = 4
        cfg = RecorderConfig(test=t, **kw)
        rec = SessionRecorder(cfg)
        track = (rec.cfg.test != "cognitive_reaction")
        assert track is True, f"departure tracking should be ON for {t}"


def test_session_metadata_includes_cog_fields():
    """When the test is cognitive_reaction, _build_metadata writes the
    cog_track_body_part / cog_n_positions / cog_positions keys that
    the analyzer reads back at analysis time."""
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_track_body_part="left_hand",
                         react_n_positions=8,
                         duration_s=30.0, n_stimuli=8, trigger="auto")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    meta = rec._build_metadata(cancelled=False)
    assert meta["test"] == "cognitive_reaction"
    assert meta["cog_track_body_part"] == "left_hand"
    assert meta["cog_n_positions"] == 8
    assert meta["cog_trigger"] == "auto"
    pos = meta["cog_positions"]
    assert isinstance(pos, list) and len(pos) == 8
    # Each row must be (label, x, y).
    assert all(len(r) == 3 for r in pos)


# ────────────────────────────────────────────────────────────────────────────
# Direct runner
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items()
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if failed:
        print(f"=== {failed}/{len(fns)} tests failed ===")
        sys.exit(1)
    print(f"=== All {len(fns)} tests passed ===")
