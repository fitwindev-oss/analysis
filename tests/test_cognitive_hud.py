"""
Unit tests for V6-G2 cognitive_hud — pure BGR drawing functions
shared by the live CameraView and the replay VideoPlayerWidget.

Tests verify pixels CHANGE in the expected places and degenerate
inputs (tiny frames, unknown grade, expired age) are no-ops.

Run:
    python tests/test_cognitive_hud.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.widgets.cognitive_hud import (
    draw_progress_bar, draw_grade_message, draw_grade_counters,
    draw_full_hud,
    GRADE_BGR, GRADE_ORDER, GRADE_MSG_HOLD_FRAMES,
)


# ────────────────────────────────────────────────────────────────────────────
# Progress bar
# ────────────────────────────────────────────────────────────────────────────
def test_progress_bar_modifies_pixels_when_in_progress():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_progress_bar(bgr, n_done=3, n_total=10, cri_live=72.0)
    assert (bgr != before).any()
    # Expect pixel changes near the top of the frame
    assert (bgr[5:50, :, :] != before[5:50, :, :]).any()


def test_progress_bar_handles_zero_total():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    draw_progress_bar(bgr, n_done=0, n_total=0)   # must not raise


def test_progress_bar_full_progress():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    draw_progress_bar(bgr, n_done=10, n_total=10, cri_live=88.0)
    # CRI badge should color in the GREAT range (gold)
    assert (bgr != 30).any()


def test_progress_bar_tiny_frame_no_op():
    bgr = np.full((40, 40, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_progress_bar(bgr, n_done=1, n_total=2, cri_live=50.0)
    # Frame too small → silently no-op (no exceptions)
    assert (bgr == before).all()


# ────────────────────────────────────────────────────────────────────────────
# Grade message
# ────────────────────────────────────────────────────────────────────────────
def test_grade_message_great_writes_pixels_at_age_zero():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_grade_message(bgr, "great", rt_ms=300.0, age_frames=0)
    assert (bgr != before).any()


def test_grade_message_fades_to_invisible_after_hold():
    """Past the hold window the function silently no-ops."""
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_grade_message(bgr, "great", rt_ms=300.0,
                        age_frames=GRADE_MSG_HOLD_FRAMES + 5)
    assert (bgr == before).all()


def test_grade_message_unknown_grade_no_op():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_grade_message(bgr, "stellar", rt_ms=300.0, age_frames=0)
    assert (bgr == before).all()


def test_grade_message_none_grade_no_op():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_grade_message(bgr, None, rt_ms=None, age_frames=0)
    assert (bgr == before).all()


def test_grade_message_great_pixels_differ_from_bad():
    """Different grades produce different on-screen colors so the user
    can tell them apart at a glance."""
    bgr_great = np.full((480, 640, 3), 30, dtype=np.uint8)
    bgr_bad   = np.full((480, 640, 3), 30, dtype=np.uint8)
    draw_grade_message(bgr_great, "great", rt_ms=200.0, age_frames=4)
    draw_grade_message(bgr_bad,   "bad",   rt_ms=900.0, age_frames=4)
    assert np.abs(bgr_great.astype(int) - bgr_bad.astype(int)).sum() > 0


def test_grade_message_alpha_decays_with_age():
    """A late-stage frame should have smaller pixel deltas than a
    fresh frame (linear fade)."""
    base = np.full((480, 640, 3), 30, dtype=np.uint8)
    bgr_fresh = base.copy()
    bgr_late  = base.copy()
    draw_grade_message(bgr_fresh, "good", 450.0, age_frames=0)
    draw_grade_message(bgr_late,  "good", 450.0,
                        age_frames=GRADE_MSG_HOLD_FRAMES - 1)
    delta_fresh = int(np.abs(bgr_fresh.astype(int) - base.astype(int)).sum())
    delta_late  = int(np.abs(bgr_late.astype(int) - base.astype(int)).sum())
    assert delta_fresh > delta_late


# ────────────────────────────────────────────────────────────────────────────
# Grade counters
# ────────────────────────────────────────────────────────────────────────────
def test_grade_counters_modifies_pixels():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_grade_counters(bgr, {"great": 3, "good": 2, "normal": 1,
                               "bad": 0, "miss": 0})
    # Expect changes in the bottom strip
    assert (bgr[-50:, :, :] != before[-50:, :, :]).any()


def test_grade_counters_empty_dict_renders_layout():
    """No counts yet — chips still draw with 0 values."""
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_grade_counters(bgr, {})
    # Layout still drawn (chips with 0). Test that bottom strip changed.
    assert (bgr != before).any()


def test_grade_counters_all_grades_have_color():
    """Sanity: every grade key in GRADE_ORDER has a BGR color."""
    for g in GRADE_ORDER:
        assert g in GRADE_BGR
        c = GRADE_BGR[g]
        assert isinstance(c, tuple) and len(c) == 3


# ────────────────────────────────────────────────────────────────────────────
# Full HUD wrapper
# ────────────────────────────────────────────────────────────────────────────
def test_draw_full_hud_calls_all_three_layers():
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_full_hud(bgr,
                   n_done=4, n_total=10,
                   recent_grade="great", recent_rt_ms=300.0,
                   recent_age_frames=2,
                   grade_counts={"great": 1, "good": 2,
                                  "normal": 1, "bad": 0, "miss": 0},
                   live_cri=78.5)
    # Top + center + bottom should all have changed
    assert (bgr[5:50, :, :] != before[5:50, :, :]).any()       # progress
    assert (bgr[200:280, :, :] != before[200:280, :, :]).any() # message
    assert (bgr[-50:, :, :] != before[-50:, :, :]).any()       # counters


def test_draw_full_hud_no_recent_grade_skips_center():
    """When recent_grade is None the message layer must not write
    pixels (the burst already faded)."""
    bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
    before = bgr.copy()
    draw_full_hud(bgr,
                   n_done=4, n_total=10,
                   recent_grade=None, recent_rt_ms=None,
                   recent_age_frames=99,
                   grade_counts={},
                   live_cri=78.5)
    # Only top + bottom strips should have changed; mid section clean
    assert (bgr[200:280, 50:-50, :] == before[200:280, 50:-50, :]).all()


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
