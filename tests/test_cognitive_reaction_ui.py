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
