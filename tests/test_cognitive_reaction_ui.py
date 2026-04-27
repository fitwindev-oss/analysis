"""
V6-UI tests — TestOptionsPanel + CameraView positional cue overlay.

Headless: launches QApplication with the offscreen platform plugin so
the test runs without a display.

Run:
    python tests/test_cognitive_reaction_ui.py
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication

from src.capture.session_recorder import (
    SessionRecorder, RecorderConfig, RecorderState,
)
from src.ui.widgets.test_options_panel import TestOptionsPanel, TESTS_KO
from src.ui.widgets.camera_view import _SingleCamTile, CameraView


_app = QApplication.instance() or QApplication([])


# ────────────────────────────────────────────────────────────────────────────
# TestOptionsPanel
# ────────────────────────────────────────────────────────────────────────────
def test_tests_ko_includes_cognitive_reaction():
    keys = [k for k, _, _ in TESTS_KO]
    assert "cognitive_reaction" in keys


def test_options_panel_emits_cognitive_reaction_kwargs():
    panel = TestOptionsPanel()
    idx = panel._combo.findData("cognitive_reaction")
    assert idx >= 0
    panel._combo.setCurrentIndex(idx)
    opts = panel.options()
    # Must carry all the V6-specific RecorderConfig fields
    assert opts["test"] == "cognitive_reaction"
    assert opts["react_track_body_part"] in (
        "right_hand", "left_hand", "right_foot", "left_foot")
    assert opts["react_n_positions"] in (4, 8)
    # And the timing fields the recorder reuses from reaction
    assert opts["n_stimuli"] >= 1
    assert opts["stim_min_gap"] > 0
    assert opts["stim_max_gap"] >= opts["stim_min_gap"]
    assert opts["trigger"] in ("auto", "manual")


def test_options_panel_8pos_radio_emits_8():
    panel = TestOptionsPanel()
    idx = panel._combo.findData("cognitive_reaction")
    panel._combo.setCurrentIndex(idx)
    panel._cog_pos_8.setChecked(True)
    opts = panel.options()
    assert opts["react_n_positions"] == 8


def test_options_panel_left_hand_emits_left_hand():
    panel = TestOptionsPanel()
    idx = panel._combo.findData("cognitive_reaction")
    panel._combo.setCurrentIndex(idx)
    bidx = panel._cog_body_part.findData("left_hand")
    assert bidx >= 0
    panel._cog_body_part.setCurrentIndex(bidx)
    assert panel.options()["react_track_body_part"] == "left_hand"


def test_options_panel_visibility_toggles_on_test_change():
    panel = TestOptionsPanel()
    # Start at the first (balance_eo) test — cognitive box should be hidden
    idx_bal = panel._combo.findData("balance_eo")
    panel._combo.setCurrentIndex(idx_bal)
    assert not panel._cognitive_box.isVisibleTo(panel)
    # Switch to cognitive_reaction — cognitive box appears, others hidden
    idx_cog = panel._combo.findData("cognitive_reaction")
    panel._combo.setCurrentIndex(idx_cog)
    assert panel._cognitive_box.isVisibleTo(panel)
    assert not panel._reaction_box.isVisibleTo(panel)
    assert not panel._balance_box.isVisibleTo(panel)


def test_options_panel_forces_auto_pose_for_cognitive_reaction():
    """V6-fix — pose processing is mandatory for cognitive_reaction
    because the analyzer reads body-part trajectories from poses_*.npz.
    options() must emit ``_auto_pose=True`` for cognitive_reaction even
    if the operator left the auto-pose checkbox cleared."""
    panel = TestOptionsPanel()
    # First select something else and uncheck auto-pose
    panel._combo.setCurrentIndex(panel._combo.findData("balance_eo"))
    panel._auto_pose.setChecked(False)
    # Now switch to cognitive_reaction
    panel._combo.setCurrentIndex(panel._combo.findData("cognitive_reaction"))
    opts = panel.options()
    assert opts["_auto_pose"] is True
    # And the checkbox should be locked on (visual cue for the operator).
    assert panel._auto_pose.isChecked() is True
    assert panel._auto_pose.isEnabled() is False


def test_auto_pose_unlocks_when_leaving_cognitive_reaction():
    """Switching off cognitive_reaction must re-enable the auto-pose
    checkbox so other tests can opt out as before."""
    panel = TestOptionsPanel()
    panel._combo.setCurrentIndex(panel._combo.findData("cognitive_reaction"))
    assert panel._auto_pose.isEnabled() is False
    panel._combo.setCurrentIndex(panel._combo.findData("cmj"))
    assert panel._auto_pose.isEnabled() is True


def test_options_panel_kwargs_pass_through_recorderconfig():
    """The dict that TestOptionsPanel.options() returns (after stripping
    the UI-only ``_*`` keys) must construct a valid RecorderConfig."""
    panel = TestOptionsPanel()
    panel._combo.setCurrentIndex(panel._combo.findData("cognitive_reaction"))
    opts = panel.options()
    # Strip UI-only keys (those that start with underscore) — same step
    # MeasureTab._start_next_in_queue does before calling RecorderConfig.
    cfg_kwargs = {k: v for k, v in opts.items() if not k.startswith("_")}
    cfg = RecorderConfig(subject_kg=80.0, **cfg_kwargs)
    assert cfg.test == "cognitive_reaction"
    assert cfg.react_track_body_part == opts["react_track_body_part"]
    assert cfg.react_n_positions     == opts["react_n_positions"]


# ────────────────────────────────────────────────────────────────────────────
# RecorderState — V6 cue fields
# ────────────────────────────────────────────────────────────────────────────
def test_recorder_state_has_cue_fields():
    st = RecorderState()
    for f in ("cog_target_x_norm", "cog_target_y_norm", "cog_target_label"):
        assert hasattr(st, f), f
        assert getattr(st, f) is None


def test_fire_stim_populates_state_cue_fields():
    cfg = RecorderConfig(test="cognitive_reaction", react_n_positions=4,
                         n_stimuli=4, trigger="manual", duration_s=30.0)
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    import time
    rec._fire_stim("pos_E", time.monotonic_ns())
    assert rec.state.cog_target_label == "pos_E"
    assert rec.state.cog_target_x_norm is not None
    assert rec.state.cog_target_y_norm is not None
    # East = large x
    assert rec.state.cog_target_x_norm > 0.5


def test_classic_reaction_does_not_set_cue_fields():
    cfg = RecorderConfig(test="reaction", n_stimuli=4, trigger="manual",
                         duration_s=30.0, responses="random")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    import time
    rec._fire_stim("jump", time.monotonic_ns())
    assert rec.state.cog_target_x_norm is None
    assert rec.state.cog_target_label is None


# ────────────────────────────────────────────────────────────────────────────
# CameraView — positional cue overlay
# ────────────────────────────────────────────────────────────────────────────
def test_tile_set_positional_cue_stores_state():
    tile = _SingleCamTile("camA", "test")
    tile.set_positional_cue(0.85, 0.50, "pos_E")
    assert tile._cue_xy == (0.85, 0.50)
    assert tile._cue_label == "pos_E"


def test_tile_clear_positional_cue_with_none():
    tile = _SingleCamTile("camA", "test")
    tile.set_positional_cue(0.5, 0.5, "pos_C")
    tile.set_positional_cue(None, None)
    assert tile._cue_xy is None
    assert tile._cue_label is None


def test_draw_positional_cue_modifies_pixels():
    """The cue draw routine must change pixels at the cued position
    relative to the original frame."""
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)   # dark grey
    base = bgr.copy()
    _SingleCamTile._draw_positional_cue(
        bgr, (0.5, 0.5), "pos_C", phase=0)
    # Center pixel must differ from baseline (cue painted there).
    assert not np.array_equal(bgr[240, 320], base[240, 320])
    # A pixel far from the center, beyond the cue radius, must stay
    # untouched (the halo is finite — a corner is well outside it).
    assert np.array_equal(bgr[5, 5], base[5, 5])


def test_draw_positional_cue_offcenter():
    """Drawing at (0.85, 0.50) should change pixels near east edge."""
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    base = bgr.copy()
    _SingleCamTile._draw_positional_cue(
        bgr, (0.85, 0.50), "pos_E", phase=10)
    # Pixel at the cued spot (0.85*639, 0.50*479) ≈ (543, 239)
    assert not np.array_equal(bgr[239, 543], base[239, 543])
    # Center should NOT have changed (cue is far from center).
    assert np.array_equal(bgr[239, 320], base[239, 320])


def test_draw_positional_cue_handles_zero_size():
    """Don't crash on a 1×1 frame."""
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    _SingleCamTile._draw_positional_cue(bgr, (0.5, 0.5), "x", 0)
    # Just shouldn't have raised.
    assert bgr.shape == (1, 1, 3)


def test_camera_view_set_positional_cue_broadcasts():
    """CameraView.set_positional_cue with cam_id=None should mirror
    onto every tile."""
    cv = CameraView()
    cv.set_positional_cue(0.5, 0.5, "pos_C")
    for t in cv._tiles.values():
        assert t._cue_xy == (0.5, 0.5)
    cv.set_positional_cue(None, None)
    for t in cv._tiles.values():
        assert t._cue_xy is None


# ────────────────────────────────────────────────────────────────────────────
# V6-hit — live "체크" feedback when tracked body part reaches the cue
# ────────────────────────────────────────────────────────────────────────────
def test_set_track_body_part_resolves_to_mp33_index():
    """set_track_body_part must map the body-part string to the right
    MP33 keypoint index so the GUI's hit detector and the offline
    analyzer agree on which joint they're watching."""
    from src.ui.widgets.camera_view import BODY_PART_TO_KP_INDEX
    from src.pose.mediapipe_backend import MP33
    cv = CameraView()
    cv.set_track_body_part("right_hand")
    for t in cv._tiles.values():
        assert t._track_kpt_idx == MP33["right_wrist"]
    cv.set_track_body_part("left_foot")
    for t in cv._tiles.values():
        assert t._track_kpt_idx == MP33["left_foot_index"]
    cv.set_track_body_part(None)
    for t in cv._tiles.values():
        assert t._track_kpt_idx is None


def test_evaluate_hit_inside_tolerance():
    """Tracked keypoint within tolerance ring → _evaluate_hit True."""
    from src.pose.mediapipe_backend import MP33
    tile = _SingleCamTile("camA", "test")
    tile.set_track_body_part("right_hand")
    tile.set_positional_cue(0.50, 0.50, "pos_C")
    # Build a synthetic kpts33 array with right_wrist exactly on the cue.
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    wrist_idx = MP33["right_wrist"]
    kpts[wrist_idx] = [0.50 * (w - 1), 0.50 * (h - 1)]
    vis[wrist_idx] = 1.0
    tile._kpts = kpts; tile._vis = vis
    assert tile._evaluate_hit(w, h) is True


def test_evaluate_hit_outside_tolerance():
    """Tracked keypoint far from cue → not a hit."""
    from src.pose.mediapipe_backend import MP33
    tile = _SingleCamTile("camA", "test")
    tile.set_track_body_part("right_hand")
    tile.set_positional_cue(0.20, 0.20, "pos_NW")
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    wrist_idx = MP33["right_wrist"]
    # Wrist on the opposite side of the frame
    kpts[wrist_idx] = [0.85 * (w - 1), 0.85 * (h - 1)]
    vis[wrist_idx] = 1.0
    tile._kpts = kpts; tile._vis = vis
    assert tile._evaluate_hit(w, h) is False


def test_evaluate_hit_low_visibility_returns_false():
    """If the tracked keypoint has visibility below threshold, treat
    as 'no hit' rather than a false positive."""
    from src.pose.mediapipe_backend import MP33
    tile = _SingleCamTile("camA", "test")
    tile.set_track_body_part("right_hand")
    tile.set_positional_cue(0.50, 0.50, "pos_C")
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    wrist_idx = MP33["right_wrist"]
    kpts[wrist_idx] = [0.50 * (w - 1), 0.50 * (h - 1)]
    vis[wrist_idx] = 0.1   # below _VIS_THRESH
    tile._kpts = kpts; tile._vis = vis
    assert tile._evaluate_hit(w, h) is False


def test_evaluate_hit_no_tracking_disabled():
    """Without set_track_body_part call, hit detection always False."""
    tile = _SingleCamTile("camA", "test")
    tile.set_positional_cue(0.5, 0.5, "pos_C")
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis = np.zeros(33, dtype=np.float32)
    tile._kpts = kpts; tile._vis = vis
    assert tile._evaluate_hit(w, h) is False


def test_draw_positional_cue_hit_state_differs_from_rest():
    """Hit-state cue should produce visibly different pixels than
    the resting cue at the same center (different colors, checkmark)."""
    bgr_rest = np.full((480, 640, 3), 30, dtype=np.uint8)
    bgr_hit  = np.full((480, 640, 3), 30, dtype=np.uint8)
    _SingleCamTile._draw_positional_cue(
        bgr_rest, (0.5, 0.5), "pos_C", phase=10, hit=False)
    _SingleCamTile._draw_positional_cue(
        bgr_hit, (0.5, 0.5), "pos_C", phase=10, hit=True)
    # Pixels at the cue center should differ — hit has the checkmark
    # tick lines plus a different core color.
    diff = np.abs(bgr_rest.astype(int) - bgr_hit.astype(int)).sum()
    assert diff > 0


def test_options_panel_forces_live_pose_for_cognitive_reaction():
    """V6-hit — live pose overlay is required for the on-screen "체크"
    effect, so cognitive_reaction must lock _live_pose=True."""
    panel = TestOptionsPanel()
    # Switch away first and clear live_pose
    panel._combo.setCurrentIndex(panel._combo.findData("balance_eo"))
    panel._live_pose.setChecked(False)
    panel._combo.setCurrentIndex(panel._combo.findData("cognitive_reaction"))
    opts = panel.options()
    assert opts["_live_pose"] is True
    assert panel._live_pose.isChecked() is True
    assert panel._live_pose.isEnabled() is False


def test_hit_latch_holds_after_brief_overshoot():
    """The hit-hold counter keeps the cue in 'ack' state for a few
    repaints after the keypoint leaves the tolerance ring, so brief
    jitter doesn't flicker the visual feedback."""
    from src.pose.mediapipe_backend import MP33
    tile = _SingleCamTile("camA", "test")
    tile.set_track_body_part("right_hand")
    tile.set_positional_cue(0.5, 0.5, "pos_C")
    w, h = 640, 480
    kpts = np.full((33, 2), np.nan, dtype=np.float32)
    vis  = np.zeros(33, dtype=np.float32)
    wrist_idx = MP33["right_wrist"]
    vis[wrist_idx] = 1.0
    # Frame 1: in the hit zone
    kpts[wrist_idx] = [0.50 * (w - 1), 0.50 * (h - 1)]
    tile._kpts = kpts.copy(); tile._vis = vis.copy()
    assert tile._evaluate_hit(w, h) is True
    # Now simulate a frame where wrist briefly leaves the zone — by
    # itself _evaluate_hit returns False, but the latch in
    # repaint_if_dirty would still keep showing "hit". We verify the
    # latch field exists and the threshold is sane (>=4 repaints).
    assert tile._HIT_HOLD_FRAMES >= 4


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
