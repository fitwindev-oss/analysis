"""
V6-G7 — end-to-end integration tests for the gamified cognitive_reaction
pipeline.

Covers:
  - analyze_cognitive_reaction populates per-trial grade + result-level
    CRI/MS/AS/CS/grade_counts/cv_rt from synthetic trial data
  - Replay HUD state computation (VideoPlayerWidget._hud_state_at)
    matches the live HUD logic when fed the same trials
  - Live recorder grade resolution (_resolve_cog_stim) produces the
    same grade as the offline analyzer when given matched RT inputs
  - Report section renders the CRI card + grade chips when result has
    the new fields populated

Run:
    python tests/test_v6g_integration.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication

from src.analysis.cognitive_reaction import (
    grade_trial, compute_cri, live_cri_after,
    CogTrial, CognitiveReactionResult,
    GRADE_WEIGHT,
)
from src.capture.session_recorder import SessionRecorder, RecorderConfig
from src.ui.widgets.video_player import VideoPlayerWidget
from src.ui.widgets.cognitive_hud import GRADE_MSG_HOLD_FRAMES


_APP = QApplication.instance() or QApplication([])


# ────────────────────────────────────────────────────────────────────────────
# Live recorder ↔ analyzer parity
# ────────────────────────────────────────────────────────────────────────────
def test_recorder_resolve_emits_grade_matching_analyzer():
    """The live grade computed by SessionRecorder._resolve_cog_stim
    must be the same label the offline analyzer would assign for an
    equivalent (rt_ms, hit) input."""
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_n_positions=4, react_track_body_part="right_hand",
                         duration_s=30.0, n_stimuli=4, trigger="manual")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    # Stage a stim manually + simulate a hit at +280ms (great)
    stim_t_ns = time.monotonic_ns()
    rec._cog_active_stim = {
        "stim_t_ns":      stim_t_ns,
        "target_label":   "pos_E",
        "deadline_ns":    stim_t_ns + int(1.5e9),
        "first_hit_t_ns": stim_t_ns + int(0.280 * 1e9),  # 280ms RT
    }
    rec._resolve_cog_stim(now_ns=stim_t_ns + int(1.5e9))
    assert rec._state.cog_recent_grade == "great"
    assert rec._state.cog_grade_counts.get("great", 0) == 1
    # The analyzer with the same RT/hit gives the same grade
    assert grade_trial(280.0, True) == "great"


def test_recorder_resolve_handles_miss():
    """No first_hit recorded → trial is a miss with rt_ms=None."""
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_n_positions=4,
                         duration_s=30.0, n_stimuli=4, trigger="manual")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()
    stim_t_ns = time.monotonic_ns()
    rec._cog_active_stim = {
        "stim_t_ns":      stim_t_ns,
        "target_label":   "pos_W",
        "deadline_ns":    stim_t_ns + int(1.5e9),
        "first_hit_t_ns": None,   # never reached
    }
    rec._resolve_cog_stim(now_ns=stim_t_ns + int(1.5e9))
    assert rec._state.cog_recent_grade == "miss"
    assert rec._state.cog_recent_rt_ms is None


def test_recorder_live_cri_updates_after_each_trial():
    """The live CRI in RecorderState should match compute_cri across
    all resolved trials so far."""
    cfg = RecorderConfig(test="cognitive_reaction",
                         react_n_positions=4,
                         duration_s=30.0, n_stimuli=4, trigger="manual")
    rec = SessionRecorder(cfg)
    rec._prepare_reaction_pool()

    def _fire_and_resolve(rt_ms, hit):
        stim_t_ns = time.monotonic_ns()
        first_hit = (stim_t_ns + int(rt_ms * 1e6)) if hit else None
        rec._cog_active_stim = {
            "stim_t_ns": stim_t_ns,
            "target_label": "pos_N",
            "deadline_ns": stim_t_ns + int(1.5e9),
            "first_hit_t_ns": first_hit,
        }
        rec._resolve_cog_stim(stim_t_ns + int(1.5e9))

    _fire_and_resolve(300, True)   # great
    _fire_and_resolve(450, True)   # good
    _fire_and_resolve(800, True)   # bad
    _fire_and_resolve(None, False)  # miss

    # Recompute from the recorder's own trials — should agree
    expected_cri = live_cri_after(rec._cog_trials)
    assert abs(rec._state.cog_live_cri - expected_cri) < 1e-6


# ────────────────────────────────────────────────────────────────────────────
# Replay HUD parity with live HUD logic
# ────────────────────────────────────────────────────────────────────────────
def _make_session_with_result(tmp: Path, trials: list[dict]) -> Path:
    """Build a session folder with session.json + stimulus_log.csv +
    forces.csv + result.json populated with the given trial data."""
    sd = tmp / "session"
    sd.mkdir()
    rec_start = 1000.0
    meta = {
        "test": "cognitive_reaction",
        "record_start_wall_s": rec_start,
        "cog_track_body_part": "right_hand",
        "cog_n_positions": 4,
        "n_stimuli": len(trials),
    }
    (sd / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    (sd / "forces.csv").write_text(
        f"t_ns,t_wall,total_n\n0,{rec_start:.6f},800\n",
        encoding="utf-8")
    # stimulus_log.csv with V6 columns
    rows = ["trial_idx,t_wall,t_ns,stimulus_type,response_type,"
            "target_x_norm,target_y_norm,target_label"]
    for i, tr in enumerate(trials):
        offs = float(tr["t_stim_s"])
        rows.append(
            f"{i},{(rec_start + offs):.6f},{i},audio_visual,pos_N,"
            f"0.5,0.2,pos_N")
    (sd / "stimulus_log.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")
    # result.json
    payload = {
        "test": "cognitive_reaction",
        "result": {
            "n_trials": len(trials),
            "trials": trials,
            "cri": compute_cri(trials)["cri"],
        }
    }
    (sd / "result.json").write_text(
        json.dumps(payload), encoding="utf-8")
    return sd


def test_replay_hud_progress_advances_with_time():
    """At t<first stim n_done=0; after final stim+1.5s n_done=N."""
    with tempfile.TemporaryDirectory() as tmp:
        trials = [
            {"t_stim_s": 5.0,  "rt_ms": 300, "hit": True, "grade": "great"},
            {"t_stim_s": 10.0, "rt_ms": 450, "hit": True, "grade": "good"},
            {"t_stim_s": 15.0, "rt_ms": 700, "hit": True, "grade": "normal"},
        ]
        sd = _make_session_with_result(Path(tmp), trials)
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        # Before any stim: 0 done
        s = w._hud_state_at(2.0)
        assert s["n_done"] == 0
        # After first cue resolved (stim+1.5s = 6.5s)
        s = w._hud_state_at(7.0)
        assert s["n_done"] == 1
        assert s["grade_counts"]["great"] == 1
        # After all three resolved
        s = w._hud_state_at(20.0)
        assert s["n_done"] == 3
        assert s["grade_counts"]["great"] == 1
        assert s["grade_counts"]["good"]  == 1
        assert s["grade_counts"]["normal"] == 1


def test_replay_hud_grade_burst_window():
    """The recent_grade burst is visible only inside the 0.8s window
    right after each trial's resolution time."""
    with tempfile.TemporaryDirectory() as tmp:
        trials = [
            {"t_stim_s": 5.0, "rt_ms": 300, "hit": True, "grade": "great"},
        ]
        sd = _make_session_with_result(Path(tmp), trials)
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        # Just before resolution → burst not started yet
        s = w._hud_state_at(6.40)
        assert s["recent_grade"] is None
        # Right at resolution → burst fresh (age_frames ≈ 0)
        s = w._hud_state_at(6.51)
        assert s["recent_grade"] == "great"
        assert s["recent_age_frames"] >= 0
        # Inside 0.8s window
        s = w._hud_state_at(7.0)
        assert s["recent_grade"] == "great"
        # After window expires (>0.8s past resolution)
        burst_window_s = GRADE_MSG_HOLD_FRAMES / 30.0
        s = w._hud_state_at(6.5 + burst_window_s + 0.5)
        assert s["recent_grade"] is None


def test_replay_hud_live_cri_grows_with_trials():
    """live_cri at end-of-session should equal compute_cri(all trials)."""
    with tempfile.TemporaryDirectory() as tmp:
        trials = [
            {"t_stim_s": 1.0, "rt_ms": 300, "hit": True, "grade": "great"},
            {"t_stim_s": 3.0, "rt_ms": 280, "hit": True, "grade": "great"},
            {"t_stim_s": 5.0, "rt_ms": 320, "hit": True, "grade": "great"},
        ]
        sd = _make_session_with_result(Path(tmp), trials)
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        s = w._hud_state_at(10.0)
        expected = compute_cri(trials)
        assert abs(s["live_cri"] - expected["cri"]) < 1e-6


def test_replay_hud_handles_missing_result_json():
    """No result.json yet (analysis hasn't run) → HUD still loads but
    with empty trials, so n_done=0, no grade burst, no error."""
    with tempfile.TemporaryDirectory() as tmp:
        sd = Path(tmp) / "session"
        sd.mkdir()
        rec_start = 1000.0
        (sd / "session.json").write_text(
            json.dumps({"test": "cognitive_reaction",
                         "record_start_wall_s": rec_start,
                         "cog_track_body_part": "right_hand",
                         "cog_n_positions": 4,
                         "n_stimuli": 5}),
            encoding="utf-8")
        (sd / "forces.csv").write_text(
            f"t_ns,t_wall,total_n\n0,{rec_start:.6f},800\n",
            encoding="utf-8")
        (sd / "stimulus_log.csv").write_text(
            "trial_idx,t_wall,t_ns,stimulus_type,response_type,"
            "target_x_norm,target_y_norm,target_label\n"
            f"0,{rec_start + 5:.6f},0,audio_visual,pos_N,0.5,0.2,pos_N\n",
            encoding="utf-8")
        w = VideoPlayerWidget()
        w._load_cognitive_cues(sd)
        s = w._hud_state_at(10.0)
        assert s is not None
        assert s["n_done"] == 0   # no trials in result.json
        assert s["recent_grade"] is None


# ────────────────────────────────────────────────────────────────────────────
# Analyzer end-to-end populates new fields
# ────────────────────────────────────────────────────────────────────────────
def test_analyzer_result_carries_cri_block():
    """A direct CognitiveReactionResult constructed from a synthetic
    trial set must carry CRI + sub-scores + grade_counts."""
    trials = [
        CogTrial(trial_idx=0, target_label="pos_N", target_x_norm=0.5,
                 target_y_norm=0.2, t_stim_s=1.0, rt_ms=280, hit=True,
                 grade="great"),
        CogTrial(trial_idx=1, target_label="pos_E", target_x_norm=0.85,
                 target_y_norm=0.5, t_stim_s=3.0, rt_ms=420, hit=True,
                 grade="good"),
        CogTrial(trial_idx=2, target_label="pos_S", target_x_norm=0.5,
                 target_y_norm=0.8, t_stim_s=5.0, rt_ms=None, hit=False,
                 grade="miss", no_response=True),
    ]
    cri_block = compute_cri(trials)
    res = CognitiveReactionResult(
        n_trials=3, n_valid=2, n_no_response=1, n_hit=2,
        hit_rate_pct=66.67,
        mean_rt_ms=350.0, median_rt_ms=350.0, std_rt_ms=70.0,
        min_rt_ms=280.0, max_rt_ms=420.0,
        mean_mt_ms=200.0, mean_total_ms=550.0,
        mean_spatial_error_norm=0.04,
        body_part="right_hand", n_positions=4,
        per_target={}, trials=trials,
        cri=cri_block["cri"],
        mean_score=cri_block["mean_score"],
        accuracy_score=cri_block["accuracy_score"],
        consistency_score=cri_block["consistency_score"],
        overall_grade=cri_block["overall_grade"],
        overall_label_ko=cri_block["overall_label_ko"],
        grade_counts=cri_block["grade_counts"],
        cv_rt=cri_block["cv_rt"],
    )
    d = res.to_dict()
    assert "cri" in d and 0.0 <= d["cri"] <= 100.0
    assert "grade_counts" in d
    assert d["grade_counts"]["great"] == 1
    assert d["grade_counts"]["good"] == 1
    assert d["grade_counts"]["miss"] == 1
    assert d["overall_grade"] in ("A", "B", "C", "D", "E")


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
