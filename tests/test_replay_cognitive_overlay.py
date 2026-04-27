"""
Tests for V6 replay-time cognitive_reaction overlay.

Covers:
  - VideoPlayerWidget reads stimulus_log.csv + session.json on load
    and converts each stim's wall time to record-relative seconds
  - _active_cue_at returns the right cue inside its display window
    and None outside
  - _is_keypoint_in_cue agrees with the live overlay's hit predicate
  - _draw_positional_cue produces visibly different pixels in the
    "hit" state vs the resting cue (same expectation as the live one)

Replay tests don't open any real video — we drive the loader through
its file-reading paths and exercise the pure helpers directly.

Run:
    python tests/test_replay_cognitive_overlay.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication

from src.ui.widgets.video_player import (
    VideoPlayerWidget,
    _is_keypoint_in_cue, _draw_positional_cue,
    _BODY_PART_TO_KP_INDEX,
)
from src.pose.mediapipe_backend import MP33


# Headless QApplication for widget tests
_APP = QApplication.instance() or QApplication([])


def _make_session(tmp: Path, *, with_cog: bool = True,
                   stim_walls_offsets: list[tuple[float, str, float, float]] = None,
                   rec_start_wall: float = 1000.0) -> Path:
    """Build a minimal session folder with session.json + stimulus_log.csv +
    forces.csv (header only). Returns the folder path."""
    sd = tmp / "session"
    sd.mkdir()
    meta: dict = {
        "test": "cognitive_reaction" if with_cog else "balance_eo",
        "record_start_wall_s": rec_start_wall,
        "cog_track_body_part": "right_hand",
        "cog_n_positions": 4,
    }
    (sd / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    # Minimal forces.csv so resolve_pose_frame's t0_wall fallback works
    (sd / "forces.csv").write_text(
        f"t_ns,t_wall,total_n\n0,{rec_start_wall:.6f},800\n",
        encoding="utf-8")
    if with_cog and stim_walls_offsets is not None:
        rows = ["trial_idx,t_wall,t_ns,stimulus_type,response_type,"
                "target_x_norm,target_y_norm,target_label"]
        for i, (offs, lbl, tx, ty) in enumerate(stim_walls_offsets):
            rows.append(
                f"{i},{(rec_start_wall + offs):.6f},{i},audio_visual,{lbl},"
                f"{tx:.4f},{ty:.4f},{lbl}")
        (sd / "stimulus_log.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
    return sd


# ────────────────────────────────────────────────────────────────────────────
# Cue loading
# ────────────────────────────────────────────────────────────────────────────
def test_load_cognitive_cues_populates_events():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_session(
            Path(tmp), with_cog=True,
            stim_walls_offsets=[
                (3.0, "pos_N", 0.50, 0.20),
                (8.0, "pos_E", 0.85, 0.50),
            ])
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        assert w._is_cognitive_reaction is True
        assert len(w._cue_events) == 2
        # First cue at t=3 s, second at t=8 s (record-relative)
        assert abs(w._cue_events[0]["t_stim_s"] - 3.0) < 1e-6
        assert abs(w._cue_events[1]["t_stim_s"] - 8.0) < 1e-6
        assert w._cue_events[0]["label"] == "pos_N"
        assert w._cue_track_kpt_idx == MP33["right_wrist"]


def test_load_cognitive_cues_skipped_for_other_tests():
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_session(Path(tmp), with_cog=False)
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        assert w._is_cognitive_reaction is False
        assert w._cue_events == []
        assert w._cue_track_kpt_idx is None


def test_active_cue_window_inclusive():
    """Cue is shown for ``_CUE_HOLD_S`` (1.5 s) starting at t_stim."""
    with tempfile.TemporaryDirectory() as tmp:
        sd = _make_session(
            Path(tmp), with_cog=True,
            stim_walls_offsets=[(5.0, "pos_S", 0.50, 0.80)])
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        # Before stim — no cue
        assert w._active_cue_at(4.99) is None
        # At stim onset — cue active
        cue = w._active_cue_at(5.0)
        assert cue is not None and cue["label"] == "pos_S"
        # 1.4 s into hold — still active
        assert w._active_cue_at(6.4) is not None
        # 1.5 s exactly — boundary still on (inclusive)
        assert w._active_cue_at(6.5) is not None
        # Just past hold — gone
        assert w._active_cue_at(6.51) is None


# ────────────────────────────────────────────────────────────────────────────
# Hit detection
# ────────────────────────────────────────────────────────────────────────────
def test_is_keypoint_in_cue_hits_within_tolerance():
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis  = np.zeros(33, dtype=np.float32)
    idx = MP33["right_wrist"]
    # Wrist exactly on the cue
    kpts[idx] = [0.50 * (w - 1), 0.50 * (h - 1)]
    vis[idx] = 1.0
    assert _is_keypoint_in_cue(kpts, vis, idx, 0.50, 0.50, w, h, 0.12) is True


def test_is_keypoint_in_cue_misses_outside_tolerance():
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis  = np.zeros(33, dtype=np.float32)
    idx = MP33["right_wrist"]
    # Wrist far from cue
    kpts[idx] = [0.10 * (w - 1), 0.10 * (h - 1)]
    vis[idx] = 1.0
    assert _is_keypoint_in_cue(kpts, vis, idx, 0.85, 0.85, w, h, 0.12) is False


def test_is_keypoint_in_cue_low_visibility_returns_false():
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis  = np.zeros(33, dtype=np.float32)
    idx = MP33["right_wrist"]
    kpts[idx] = [0.50 * (w - 1), 0.50 * (h - 1)]
    vis[idx] = 0.1   # below threshold
    assert _is_keypoint_in_cue(kpts, vis, idx, 0.50, 0.50, w, h, 0.12) is False


def test_is_keypoint_in_cue_nan_returns_false():
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis  = np.full(33, 1.0, dtype=np.float32)
    idx = MP33["right_wrist"]
    # NaN coords even with full visibility
    assert _is_keypoint_in_cue(kpts, vis, idx, 0.50, 0.50, w, h, 0.12) is False


def test_body_part_to_kp_index_matches_camera_view():
    """Replay's body-part mapping must match the live one so live and
    replay highlight the same joint."""
    from src.ui.widgets.camera_view import (
        BODY_PART_TO_KP_INDEX as LIVE_MAP,
    )
    for k, v in _BODY_PART_TO_KP_INDEX.items():
        assert LIVE_MAP[k] == v


# ────────────────────────────────────────────────────────────────────────────
# Cue drawing
# ────────────────────────────────────────────────────────────────────────────
def test_draw_positional_cue_modifies_pixels():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    _draw_positional_cue(bgr, (0.5, 0.5), "pos_C", phase=0, hit=False)
    assert (bgr != before).any()


def test_draw_positional_cue_hit_state_differs_from_rest():
    bgr_rest = np.full((480, 640, 3), 30, dtype=np.uint8)
    bgr_hit  = np.full((480, 640, 3), 30, dtype=np.uint8)
    _draw_positional_cue(bgr_rest, (0.5, 0.5), "pos_C", phase=10, hit=False)
    _draw_positional_cue(bgr_hit,  (0.5, 0.5), "pos_C", phase=10, hit=True)
    assert np.abs(bgr_rest.astype(int) - bgr_hit.astype(int)).sum() > 0


def test_draw_positional_cue_handles_zero_size():
    """Defensive — degenerate frame size must not raise."""
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    _draw_positional_cue(bgr, (0.5, 0.5), "pos_C", phase=0, hit=False)
    # Should be a no-op (no exception)


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
