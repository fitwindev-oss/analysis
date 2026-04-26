"""
Unit tests for the strength_3lift multi-set state machine
(Phase V1-D).

These exercise SessionRecorder's set-tracking + inter_set_rest phase
WITHOUT spinning up real hardware. The internal helpers are called
directly so we can assert on state transitions deterministically.

Run from project root:
    python tests/test_strength_3lift_state.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.session_recorder import (
    SessionRecorder, RecorderConfig,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def make_recorder(**overrides) -> SessionRecorder:
    """Build a strength_3lift recorder without touching hardware."""
    cfg_kwargs = dict(
        test="strength_3lift",
        exercise="bench_press",
        n_sets=3,
        target_reps=12,
        load_kg=80.0,
        rest_s=30.0,
        warmup_set=True,
        subject_kg=80.0,
    )
    cfg_kwargs.update(overrides)
    return SessionRecorder(RecorderConfig(**cfg_kwargs))


def simulate_recording_start(rec: SessionRecorder) -> None:
    """Bypass the real wait/countdown and put the recorder directly in
    a state equivalent to "recording, set 0 in progress" so transition
    tests can run without DAQ."""
    rec._rec_start_ns = time.monotonic_ns()
    rec._rec_start_wall = time.time()
    rec._record_ready.set()
    # Mimic _transition_to_recording's bookkeeping for the first set.
    rec._state.phase = "recording"
    rec._state.current_set_idx = 0
    rec._state.n_sets = rec.cfg.n_sets
    rec._t_phase_ns = time.monotonic_ns()
    rec._set_t_start_s = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Config validation
# ────────────────────────────────────────────────────────────────────────────
def test_config_accepts_valid_strength_3lift():
    cfg = RecorderConfig(
        test="strength_3lift",
        exercise="bench_press",
        n_sets=3,
        rest_s=30.0,
        subject_kg=80.0,
    )
    assert cfg.test == "strength_3lift"
    assert cfg.exercise == "bench_press"
    assert cfg.n_sets == 3


def test_config_rejects_unknown_exercise():
    try:
        RecorderConfig(
            test="strength_3lift", exercise="leg_press",
            n_sets=3, rest_s=30.0, subject_kg=80.0,
        )
        assert False, "should reject unknown exercise"
    except ValueError as e:
        assert "exercise" in str(e).lower()


def test_config_rejects_n_sets_below_3():
    try:
        RecorderConfig(
            test="strength_3lift", exercise="bench_press",
            n_sets=2, rest_s=30.0, subject_kg=80.0,
        )
        assert False, "should reject n_sets=2"
    except ValueError as e:
        assert "n_sets" in str(e).lower()


def test_config_rejects_n_sets_above_5():
    try:
        RecorderConfig(
            test="strength_3lift", exercise="bench_press",
            n_sets=6, rest_s=30.0, subject_kg=80.0,
        )
        assert False, "should reject n_sets=6"
    except ValueError as e:
        assert "n_sets" in str(e).lower()


def test_config_accepts_3_4_5_sets():
    """The plan allows 3, 4, or 5 sets."""
    for n in (3, 4, 5):
        RecorderConfig(test="strength_3lift", exercise="bench_press",
                       n_sets=n, rest_s=30.0, subject_kg=80.0)


def test_config_rejects_short_rest():
    try:
        RecorderConfig(
            test="strength_3lift", exercise="bench_press",
            n_sets=3, rest_s=0.5, subject_kg=80.0,
        )
        assert False, "should reject rest_s < 1"
    except ValueError as e:
        assert "rest" in str(e).lower()


def test_config_does_not_require_strength_fields_for_other_tests():
    """Other tests don't need exercise/n_sets to be set."""
    cfg = RecorderConfig(test="cmj", subject_kg=80.0)
    assert cfg.exercise is None
    assert cfg.n_sets == 3   # default; ignored for non-strength tests


# ────────────────────────────────────────────────────────────────────────────
# Public control methods exist
# ────────────────────────────────────────────────────────────────────────────
def test_recorder_exposes_multiset_controls():
    rec = make_recorder()
    for name in ("end_set", "pause_rest", "resume_rest",
                 "skip_rest", "end_session"):
        fn = getattr(rec, name, None)
        assert callable(fn), f"missing public method: {name}"


def test_initial_state_for_multiset():
    rec = make_recorder()
    s = rec._state
    assert s.phase == "idle"
    assert s.current_set_idx == 0
    assert s.n_sets == 0           # Set on first transition_to_recording
    assert s.rest_remaining_s == 0.0
    assert s.rest_paused is False


# ────────────────────────────────────────────────────────────────────────────
# Set-end → inter_set_rest transition
# ────────────────────────────────────────────────────────────────────────────
def test_close_current_set_appends_boundary():
    rec = make_recorder()
    simulate_recording_start(rec)
    # Force the boundary to be measurable regardless of OS sleep
    # granularity — on Windows ``time.sleep(0.01)`` can return in
    # near-zero wall time when the timer is coarse, which would round
    # t_end_s to 0.0 and fail the > 0 assertion. Synthesise a known
    # gap by rewinding _rec_start_ns 100 ms into the past instead.
    rec._rec_start_ns -= 100_000_000        # 100 ms in nanoseconds
    rec._close_current_set(phase_s=5.0)
    assert len(rec._sets) == 1
    s = rec._sets[0]
    assert s["set_idx"] == 0
    assert s["warmup"] is True       # cfg.warmup_set=True and idx=0
    assert s["exercise"] == "bench_press"
    assert s["load_kg"] == 80.0
    assert s["t_start_s"] == 0.0
    assert s["t_end_s"] >= 0.1       # at least the 100 ms we rewound


def test_warmup_flag_only_for_first_set():
    """With warmup_set=True, only set 0 is the warmup."""
    rec = make_recorder(warmup_set=True)
    simulate_recording_start(rec)
    rec._close_current_set(phase_s=5.0)
    assert rec._sets[0]["warmup"] is True
    # Move to set 1
    rec._current_set_idx = 1
    rec._set_t_start_s = 30.0
    rec._close_current_set(phase_s=10.0)
    assert rec._sets[1]["warmup"] is False


def test_warmup_disabled_first_set_not_warmup():
    """With warmup_set=False, even set 0 is treated as a working set."""
    rec = make_recorder(warmup_set=False)
    simulate_recording_start(rec)
    rec._close_current_set(phase_s=5.0)
    assert rec._sets[0]["warmup"] is False


def test_enter_inter_set_rest_initialises_timer():
    rec = make_recorder()
    simulate_recording_start(rec)
    now_ns = time.monotonic_ns()
    rec._enter_inter_set_rest(now_ns)
    assert rec._state.phase == "inter_set_rest"
    assert rec._state.rest_remaining_s == 30.0
    assert rec._state.rest_paused is False
    assert rec._rest_t0_ns is not None
    assert rec._rest_pause_accum_ns == 0


# ────────────────────────────────────────────────────────────────────────────
# Pause / resume
# ────────────────────────────────────────────────────────────────────────────
def test_pause_rest_freezes_remaining():
    rec = make_recorder(rest_s=10.0)
    simulate_recording_start(rec)
    rec._enter_inter_set_rest(time.monotonic_ns())
    # Let some time pass, then tick.
    time.sleep(0.05)
    rec._tick_inter_set_rest(time.monotonic_ns())
    elapsed_before = rec._state.elapsed_s
    rec.pause_rest()
    assert rec._state.rest_paused is True
    # Sleep more — the elapsed should NOT continue while paused.
    time.sleep(0.10)
    rec._tick_inter_set_rest(time.monotonic_ns())
    elapsed_during_pause = rec._state.elapsed_s
    assert abs(elapsed_during_pause - elapsed_before) < 0.02, \
        f"elapsed advanced while paused: {elapsed_before} → {elapsed_during_pause}"


def test_resume_rest_continues_countdown():
    rec = make_recorder(rest_s=10.0)
    simulate_recording_start(rec)
    rec._enter_inter_set_rest(time.monotonic_ns())
    time.sleep(0.02)
    rec._tick_inter_set_rest(time.monotonic_ns())
    rec.pause_rest()
    time.sleep(0.10)              # paused for 100ms
    rec.resume_rest()
    assert rec._state.rest_paused is False
    time.sleep(0.02)
    rec._tick_inter_set_rest(time.monotonic_ns())
    # Total wall ~140ms but elapsed should only count the unpaused
    # portion (~40ms). With 10s rest, remaining should still be ~9.96s.
    assert rec._state.rest_remaining_s > 9.5


def test_pause_outside_rest_phase_is_noop():
    """pause_rest must not change state when not in inter_set_rest."""
    rec = make_recorder()
    rec._state.phase = "recording"
    rec.pause_rest()
    assert rec._state.rest_paused is False
    assert rec._rest_paused_at_ns is None


# ────────────────────────────────────────────────────────────────────────────
# Skip-rest trigger
# ────────────────────────────────────────────────────────────────────────────
def test_skip_rest_advances_to_next_set():
    rec = make_recorder(rest_s=30.0)
    simulate_recording_start(rec)
    rec._enter_inter_set_rest(time.monotonic_ns())
    rec.skip_rest()
    rec._tick_inter_set_rest(time.monotonic_ns())
    assert rec._state.phase == "recording"
    assert rec._state.current_set_idx == 1


# ────────────────────────────────────────────────────────────────────────────
# Auto-transition when timer reaches 0
# ────────────────────────────────────────────────────────────────────────────
def test_rest_auto_completes_after_rest_s():
    """With a tiny rest_s (0.05s), one tick after sleep should
    auto-transition back to recording for the next set."""
    rec = make_recorder(rest_s=1.0)        # 1s rest (cfg validates ≥ 1)
    simulate_recording_start(rec)
    rec._enter_inter_set_rest(time.monotonic_ns())
    time.sleep(1.05)                        # exceed the rest window
    rec._tick_inter_set_rest(time.monotonic_ns())
    assert rec._state.phase == "recording"
    assert rec._state.current_set_idx == 1


# ────────────────────────────────────────────────────────────────────────────
# end_session — operator-driven finalize
# ────────────────────────────────────────────────────────────────────────────
def test_end_session_during_recording_closes_set_and_finalises():
    rec = make_recorder()
    simulate_recording_start(rec)
    rec.end_session()
    rec._handle_end_session(time.monotonic_ns())
    assert rec._state.phase == "done"
    # The active set must have been closed → boundary recorded.
    assert len(rec._sets) == 1
    assert rec._sets[0]["set_idx"] == 0


def test_end_session_during_rest_no_set_close():
    """If pressed during inter_set_rest (after a set already closed),
    we go to done without re-closing anything."""
    rec = make_recorder()
    simulate_recording_start(rec)
    rec._close_current_set(phase_s=5.0)             # close set 0 normally
    rec._enter_inter_set_rest(time.monotonic_ns())
    n_sets_before = len(rec._sets)
    rec.end_session()
    rec._handle_end_session(time.monotonic_ns())
    assert rec._state.phase == "done"
    assert len(rec._sets) == n_sets_before          # no extra append


# ────────────────────────────────────────────────────────────────────────────
# Metadata
# ────────────────────────────────────────────────────────────────────────────
def test_metadata_includes_strength_fields():
    rec = make_recorder()
    rec._rec_start_wall = 1234567890.0
    meta = rec._build_metadata(cancelled=False)
    assert meta["exercise"] == "bench_press"
    assert meta["n_sets"] == 3
    assert meta["target_reps"] == 12
    assert meta["rest_s"] == 30.0
    assert meta["warmup_set"] is True
    assert meta["sets"] == []           # no sets recorded yet


def test_metadata_strength_fields_none_for_other_tests():
    """A CMJ session shouldn't carry strength_3lift fields populated."""
    cfg = RecorderConfig(test="cmj", subject_kg=80.0)
    rec = SessionRecorder(cfg)
    rec._rec_start_wall = 1234567890.0
    meta = rec._build_metadata(cancelled=False)
    assert meta["exercise"] is None
    assert meta["n_sets"] is None
    assert meta["sets"] is None


def test_metadata_carries_completed_sets():
    rec = make_recorder()
    simulate_recording_start(rec)
    rec._close_current_set(phase_s=5.0)
    rec._current_set_idx = 1
    rec._set_t_start_s = 35.0
    rec._close_current_set(phase_s=10.0)
    meta = rec._build_metadata(cancelled=False)
    assert len(meta["sets"]) == 2
    assert meta["sets"][0]["warmup"] is True
    assert meta["sets"][1]["warmup"] is False


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
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"=== {failed}/{len(fns)} tests failed ===")
        sys.exit(1)
    print(f"=== All {len(fns)} tests passed ===")
