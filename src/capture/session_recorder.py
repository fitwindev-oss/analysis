"""
Framework-agnostic session-recording orchestrator.

Extracted from scripts/record_session.py so the same state machine can drive:
  - the new Qt Measure-tab live view (via QThread worker), and
  - the legacy CLI script (unchanged semantics).

Design:
  - No Qt imports. No cv2 imports (display overlays are the UI's job).
  - Hardware IO lives in DaqReader + MultiCameraCapture as before.
  - run() is a blocking control loop; call it from a background thread.
  - Every meaningful event is reported via caller-supplied callbacks:
        on_camera_frame(cam_id, bgr_ndarray, frame_idx, t_ns)
        on_daq_frame(DaqFrame)
        on_state(RecorderState)      # phase transitions + per-tick updates
        on_log(str)
  - Callbacks are invoked on the recorder thread. Qt consumers should
    translate them into queued-connection signals.

State machine:
        idle → wait (optional) → countdown (if no smart-wait) → recording
                                                        → done
                                                        → cancelled
"""
from __future__ import annotations

import csv
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import config
from src.capture.camera_worker import MultiCameraCapture
from src.capture.daq_reader import DaqReader, DaqFrame
from src.capture.wait_for_stance import StabilityDetector, StabilityState
from src.capture.cop_state import (
    classify_on_plate, departure_threshold_n,
)
from src.capture.departure_events import DepartureEventTracker


TEST_PROMPTS: dict[str, str] = {
    "balance_eo":     "EYES OPEN - stand still",
    "balance_ec":     "EYES CLOSED - stand still (close your eyes NOW)",
    "cmj":            "CMJ - stand still, then JUMP (with counter-movement)",
    "sj":             "SJ - squat hold on box, then JUMP (no counter-movement)",
    "encoder":        "ENCODER - follow movement prompt",
    "reaction":       "REACTION - respond to stimulus",
    "cognitive_reaction": "COGNITIVE REACTION - reach hand to cued position",
    "squat":          "SQUAT - perform reps when ready",
    "overhead_squat": "OVERHEAD SQUAT - bar/dowel overhead",
    "proprio":        "PROPRIOCEPTION - follow target cue",
    "free_exercise":  "FREE EXERCISE - perform reps when ready",
    # Phase V1 — barbell strength assessment with multi-set protocol.
    # Single session = one exercise × N sets (3-5) of target_reps reps,
    # with a fixed inter-set rest. Set ends are operator-driven (manual
    # button), rest auto-transitions to next set after rest_s seconds.
    "strength_3lift": "STRENGTH (bench/squat/deadlift) - 3-5 sets",
}

STANCE_LABEL: dict[str, str] = {
    "two":   "BOTH feet on plates",
    "left":  "LEFT foot only (Board1)",
    "right": "RIGHT foot only (Board2)",
}

# response_type -> (label, BGR color for any legacy overlay)
REACTION_RESPONSES: dict[str, tuple[str, tuple[int, int, int]]] = {
    "left_shift":  ("LEFT SHIFT!",   (0, 200, 255)),
    "right_shift": ("RIGHT SHIFT!",  (0, 200, 255)),
    "jump":        ("JUMP!",         (0, 255, 0)),
    # Phase S1d — squat bottom-of-descent cue for VRT measurement.
    # Fires once per rep when the operator (or future auto-depth
    # detection) signals the subject should start the concentric push.
    "squat_ascent": ("⚡ UP!",       (170, 245, 0)),   # FITWIN green
    # Phase V6 — positional stimuli for the cognitive_reaction test.
    # Each label is the on-screen banner shown to cue the subject;
    # the actual target XY is logged separately in stimulus_log.csv.
    "pos_N":  ("↑ 위로",     (170, 245, 0)),
    "pos_NE": ("↗ 오른쪽 위", (170, 245, 0)),
    "pos_E":  ("→ 오른쪽",    (170, 245, 0)),
    "pos_SE": ("↘ 오른쪽 아래", (170, 245, 0)),
    "pos_S":  ("↓ 아래로",    (170, 245, 0)),
    "pos_SW": ("↙ 왼쪽 아래", (170, 245, 0)),
    "pos_W":  ("← 왼쪽",      (170, 245, 0)),
    "pos_NW": ("↖ 왼쪽 위",   (170, 245, 0)),
}

# V6 — positional stimulus geometry (normalised image coords, 0..1).
# Each entry: (label_key, target_x_norm, target_y_norm).
# (0.5, 0.5) is the camera frame center; lower y is image-top.
COGNITIVE_REACTION_POSITIONS_8: list[tuple[str, float, float]] = [
    ("pos_N",  0.50, 0.20),
    ("pos_NE", 0.78, 0.25),
    ("pos_E",  0.85, 0.50),
    ("pos_SE", 0.78, 0.75),
    ("pos_S",  0.50, 0.80),
    ("pos_SW", 0.22, 0.75),
    ("pos_W",  0.15, 0.50),
    ("pos_NW", 0.22, 0.25),
]
COGNITIVE_REACTION_POSITIONS_4: list[tuple[str, float, float]] = [
    ("pos_N", 0.50, 0.20),
    ("pos_E", 0.85, 0.50),
    ("pos_S", 0.50, 0.80),
    ("pos_W", 0.15, 0.50),
]


@dataclass
class RecorderConfig:
    test: str
    duration_s: float = 30.0
    countdown_s: float = 5.0
    # Subject
    subject_id:   Optional[str] = None
    subject_name: Optional[str] = None
    subject_kg:   float = 90.0
    # Phase V1-bugfix — sex / birthdate are now first-class subject
    # context fields. They flow from the Subjects DB through MeasureTab
    # into the recorder, then get written to session.json so the
    # downstream 1RM grade lookup has everything it needs without
    # touching the DB at analysis time. Optional for back-compat —
    # tests or older callers that don't pass them get None, and the
    # analyzer falls back to a DB lookup by subject_id.
    subject_sex:        Optional[str] = None     # 'M' / 'F' (any case)
    subject_birthdate:  Optional[str] = None     # 'YYYY-MM-DD'
    subject_height_cm:  Optional[float] = None   # informational; future use
    # Balance
    stance: str = "two"
    # Encoder
    encoder_prompt: Optional[str] = None
    # Reaction
    n_stimuli:    int   = 10
    stim_min_gap: float = 2.0
    stim_max_gap: float = 5.0
    trigger:      str   = "auto"      # auto / manual
    responses:    str   = "random"    # "random" or comma-separated subset
    # Smart-wait
    use_smart_wait: bool  = True
    wait_timeout_s: float = 60.0
    # Free exercise
    exercise_name:        Optional[str] = None   # e.g. "back squat", "bench press"
    load_kg:              float = 0.0            # external load on the encoder
    use_bodyweight_load:  bool  = False          # if True, load_kg += subject_kg
    # Encoder hardware usage (non-balance tests). When False, replay hides
    # encoder timeseries + bars so the user knows no bar/rod was attached.
    uses_encoder:         bool  = True
    # ── Cognitive reaction (Phase V6) ────────────────────────────────────
    # Active only when test == "cognitive_reaction". Uses the existing
    # n_stimuli / stim_min_gap / stim_max_gap timing fields. The
    # positional cues come from the COGNITIVE_REACTION_POSITIONS_{4,8}
    # tables; pose tracking after each stim measures RT + accuracy.
    react_track_body_part: str = "right_hand"      # right_hand / left_hand /
                                                    # right_foot / left_foot
    react_n_positions:     int = 4                  # 4 (cardinal) or 8
    # ── Strength 3-lift multi-set (Phase V1-D) ───────────────────────────
    # Active only when test == "strength_3lift". The recorder cycles
    # through `n_sets` recordings separated by `rest_s` of inter-set rest.
    # Set ends are operator-driven via end_set(); rest auto-transitions.
    exercise:        Optional[str] = None    # bench_press|back_squat|deadlift
    n_sets:          int           = 3       # 3, 4, or 5
    target_reps:     int           = 12      # informational target (reps to failure)
    rest_s:          float         = 30.0    # inter-set rest (forced timer)
    warmup_set:      bool          = True    # if True, set 0 is a warmup (excluded from 1RM)
    # Session folder
    sessions_dir:        Optional[Path] = None   # default: config.SESSIONS_DIR
    session_name_suffix: Optional[str]  = None   # appended after timestamp

    def __post_init__(self):
        # Resolve bodyweight addition at construction time so the saved
        # session.json meta records the effective load, not the flag.
        #
        # Two different policies live here on purpose:
        #
        #   strength_3lift  → DON'T modify load_kg. The analyzer applies
        #                     the exercise-specific BW factor (bench:0,
        #                     squat:0.85, deadlift:0.10) at analysis
        #                     time so per-set load_kg in session.json
        #                     stays as the raw external bar weight, and
        #                     the operator can later switch the flag
        #                     off-line if the wrong protocol was selected.
        #
        #   free_exercise   → ADD 100 % of bodyweight (legacy behavior:
        #                     used for push-ups + weighted vests where
        #                     the convention is "load_kg is total mass
        #                     being lifted").
        if self.use_bodyweight_load and self.subject_kg > 0:
            if self.test == "strength_3lift":
                # Phase V1.5: strength_3lift handles BW addition at
                # analysis time via EXERCISE_BW_FACTOR. Leave load_kg
                # as the raw bar weight here.
                pass
            else:
                self.load_kg = float(self.load_kg) + float(self.subject_kg)
        # Validate multi-set strength config when that test is selected.
        if self.test == "strength_3lift":
            if self.exercise not in ("bench_press", "back_squat", "deadlift"):
                raise ValueError(
                    f"strength_3lift requires exercise ∈ "
                    f"(bench_press, back_squat, deadlift), got {self.exercise!r}")
            if not (3 <= self.n_sets <= 5):
                raise ValueError(
                    f"strength_3lift n_sets must be 3-5, got {self.n_sets}")
            if self.rest_s < 1.0:
                raise ValueError(
                    f"strength_3lift rest_s must be ≥ 1s, got {self.rest_s}")


@dataclass
class RecorderState:
    # phase: idle / wait / countdown / recording / inter_set_rest / done / cancelled
    # ``inter_set_rest`` is only used by the strength_3lift multi-set test;
    # all other tests transition recording → done directly.
    phase: str = "idle"
    prompt: str = ""
    elapsed_s: float = 0.0       # within current phase
    total_s: float = 0.0         # recording duration target
    wait: Optional[StabilityState] = None
    zeroing: bool = False        # True only during DAQ zero-cal (first ~5.5s)
    zero_cal_remaining_s: float = 0.0  # countdown during zero-cal
    stim_banner: Optional[str] = None
    n_stim_fired: int = 0
    # ── Multi-set strength assessment (Phase V1-D) ─────────────────────
    # Populated only when test == "strength_3lift". The current set
    # being recorded (or just rested between) and the rest countdown.
    current_set_idx:     int   = 0       # 0-based; 0 = first set (or warmup)
    n_sets:              int   = 0       # total sets (= cfg.n_sets), 0 if not multi-set
    rest_remaining_s:    float = 0.0     # seconds left in current inter-set rest
    rest_paused:         bool  = False   # operator pressed pause during rest


# Callback typedefs (all optional)
CameraFrameCB = Callable[[str, np.ndarray, int, int], None]   # cam_id, bgr, frame_idx, t_ns
DaqFrameCB    = Callable[[DaqFrame], None]
StateCB       = Callable[[RecorderState], None]
LogCB         = Callable[[str], None]


class SessionRecorder:
    """Orchestrates one recording session. Thread-safe start/cancel."""

    def __init__(self, cfg: RecorderConfig):
        if cfg.test not in TEST_PROMPTS:
            raise ValueError(f"unknown test: {cfg.test!r}")
        if cfg.stance not in ("two", "left", "right"):
            raise ValueError(f"invalid stance: {cfg.stance!r}")
        self.cfg = cfg

        # Session folder (create lazily on run())
        self.session_dir: Optional[Path] = None
        self.session_name: Optional[str] = None

        # State / callbacks
        self._state = RecorderState(
            phase="idle",
            prompt=self._build_prompt(),
            total_s=cfg.duration_s,
        )
        self._on_cam:   Optional[CameraFrameCB] = None
        self._on_daq:   Optional[DaqFrameCB]    = None
        self._on_state: Optional[StateCB]       = None
        self._on_log:   Optional[LogCB]         = None

        # Hardware handles
        self._daq: Optional[DaqReader]           = None
        self._cam: Optional[MultiCameraCapture]  = None
        self._stability: Optional[StabilityDetector] = None
        self._daq_connected = False

        # Persisted DAQ frames during recording
        self._daq_frames: list[DaqFrame] = []
        # Per-sample on/off-plate classification, parallel to _daq_frames.
        # Written to forces.csv as the ``on_plate`` column. Kept as a
        # plain list so it grows in lockstep under the same _daq_lock
        # the frames list uses.
        self._cop_states: list[int] = []
        # Streaming tracker that turns the per-sample states into
        # departure events. Threshold is recomputed once we know
        # subject_kg (in _setup_hardware).
        self._dep_threshold_n: float = departure_threshold_n(
            self.cfg.subject_kg)
        self._dep_tracker: DepartureEventTracker = DepartureEventTracker(
            min_duration_s=0.05)
        # Aggregated stats from the tracker — populated in _finalize
        # so _build_metadata can include them in session.json.
        self._departures_summary: dict = {}
        # Single-slot latest frame — atomic reference assignment eliminates
        # the clear()+append() race the list-based version had.
        self._daq_latest_frame: Optional[DaqFrame] = None
        self._daq_lock = threading.Lock()
        self._record_ready = threading.Event()

        # Reaction stimuli
        self._response_pool: list[str] = []
        self._stim_times: list[tuple[float, str]] = []    # auto mode
        self._stim_events: list[dict] = []
        self._stim_rng = random.Random()
        self._pending_manual: list[str] = []              # from caller
        self._pending_lock = threading.Lock()

        # Timing
        self._t_phase_ns   = 0
        self._rec_start_ns: Optional[int]   = None
        self._rec_start_wall: Optional[float] = None
        # Snapshot of the wall instant we declared "recording finished"
        # — captured at the very top of _finalize() before any hardware
        # stop blocks, so it's a tight upper bound on both force and
        # camera streams. See _finalize() for the close-window protocol.
        self._rec_end_ns: Optional[int]     = None
        self._rec_end_wall: Optional[float] = None
        self._wait_duration_s = 0.0
        self._stim_banner_until = 0.0
        self._stim_banner_type: Optional[str] = None

        # Fall-off detection (balance tests): timestamp when total_n first
        # fell below the 30%-BW threshold. Reset whenever force recovers.
        self._fall_off_since_ns: Optional[int] = None
        self._fell_off: bool = False

        # Control flags
        self._cancel = threading.Event()
        self._running = False

        # ── Multi-set strength assessment (Phase V1-D) ─────────────────
        # Active only when cfg.test == "strength_3lift". The state machine
        # cycles recording → inter_set_rest → recording across n_sets,
        # driven by external triggers (end_set / pause_rest / resume_rest /
        # skip_rest / end_session) plus the rest countdown.
        self._sets: list[dict] = []                # boundary records, one per completed set
        self._current_set_idx: int = 0
        self._set_t_start_s: Optional[float] = None  # phase_s when current set started
        # Rest timing — wall-clock-anchored countdown (PlaybackController
        # pattern). _rest_t0_ns is the monotonic instant rest started;
        # _rest_paused_at_ns is non-None while paused.
        self._rest_t0_ns: Optional[int] = None
        self._rest_pause_accum_ns: int = 0
        self._rest_paused_at_ns: Optional[int] = None
        # External triggers — set by the operator-facing methods, consumed
        # by the next tick of the run loop. Single-flag booleans are
        # atomic in CPython so no lock is needed for these.
        self._trig_end_set: bool      = False
        self._trig_skip_rest: bool    = False
        self._trig_end_session: bool  = False

    # ── public API ──────────────────────────────────────────────────────────
    def set_callbacks(self, *, on_camera_frame: Optional[CameraFrameCB] = None,
                      on_daq_frame: Optional[DaqFrameCB] = None,
                      on_state: Optional[StateCB] = None,
                      on_log:   Optional[LogCB]   = None) -> None:
        self._on_cam, self._on_daq = on_camera_frame, on_daq_frame
        self._on_state, self._on_log = on_state, on_log

    def cancel(self) -> None:
        """Request early termination. Safe to call from any thread."""
        self._cancel.set()

    # ── Multi-set strength assessment controls (Phase V1-D) ────────────────
    # All four are safe to call from any thread; they just flip flags
    # the next ``_run_loop`` tick consumes. They have no effect for
    # tests other than ``strength_3lift``.

    def end_set(self) -> None:
        """Operator-driven "세트 종료" — close the current set and
        transition to inter-set rest (or to ``done`` if it was the
        last set). Called from the GUI button.
        """
        self._trig_end_set = True

    def pause_rest(self) -> None:
        """Pause the inter-set rest countdown. Subject can call this if
        they need extra time. Has no effect outside the rest phase."""
        if self._state.phase != "inter_set_rest":
            return
        if self._rest_paused_at_ns is None:
            self._rest_paused_at_ns = time.monotonic_ns()
            self._state.rest_paused = True

    def resume_rest(self) -> None:
        """Resume a paused rest countdown."""
        if self._state.phase != "inter_set_rest":
            return
        if self._rest_paused_at_ns is not None:
            self._rest_pause_accum_ns += (
                time.monotonic_ns() - self._rest_paused_at_ns)
            self._rest_paused_at_ns = None
            self._state.rest_paused = False

    def skip_rest(self) -> None:
        """Skip the remaining rest and start the next set immediately."""
        self._trig_skip_rest = True

    def end_session(self) -> None:
        """Operator-driven "세션 종료" — finalize the session, saving
        whatever sets have been completed so far. Distinct from
        ``cancel()`` (which discards the recording entirely)."""
        self._trig_end_session = True

    def manual_reaction(self, response_type: str) -> None:
        """Queue a reaction stimulus / VRT cue from the operator.

        Accepts any key in REACTION_RESPONSES including the Phase S1d
        ``"squat_ascent"`` cue used during squat/overhead_squat VRT
        measurement. The main loop filters which types are actually
        valid for the current ``cfg.test``."""
        if response_type not in REACTION_RESPONSES:
            return
        with self._pending_lock:
            self._pending_manual.append(response_type)

    def manual_random(self) -> None:
        """Queue a random reaction stimulus from the configured pool."""
        if not self._response_pool:
            return
        self.manual_reaction(self._stim_rng.choice(self._response_pool))

    @property
    def state(self) -> RecorderState:
        return self._state

    # ── main entry point ────────────────────────────────────────────────────
    def run(self) -> dict:
        """Blocking: runs the full session and returns a result summary dict.

        Safe to call from a worker thread. After return, the session folder
        contains forces.csv, videos, session.json, stimulus_log.csv (if any).
        """
        self._running = True
        try:
            self._prepare_session_dir()
            self._prepare_reaction_pool()
            self._start_hardware()
            self._loop()
            return self._finalize()
        finally:
            self._running = False

    # ── helpers: setup ──────────────────────────────────────────────────────
    def _prepare_session_dir(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        # Start: test key. Append an option tag so the folder name encodes
        # the stance/variant (balance_ec_left, encoder_squat5x60, ...).
        parts = [self.cfg.test]
        tag = self._option_tag()
        if tag:
            parts.append(tag)
        parts.append(ts)
        if self.cfg.subject_id:
            parts.append(self.cfg.subject_id)
        if self.cfg.session_name_suffix:
            parts.append(self.cfg.session_name_suffix)
        name = "_".join(parts)
        root = self.cfg.sessions_dir or config.SESSIONS_DIR
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        self.session_dir = d
        self.session_name = name
        self._log(f"session: {d}")

    def _option_tag(self) -> str:
        """Short, filename-safe tag summarising the test-specific options."""
        cfg = self.cfg
        if cfg.test in ("balance_eo", "balance_ec"):
            return cfg.stance       # two / left / right
        if cfg.test == "encoder" and cfg.encoder_prompt:
            # First whitespace-delimited word, sanitized
            first = cfg.encoder_prompt.split()[0] if cfg.encoder_prompt.split() else ""
            safe = "".join(c for c in first if c.isalnum())
            return safe[:12]
        if cfg.test == "reaction":
            return f"{cfg.n_stimuli}x{cfg.trigger}"
        if cfg.test == "cognitive_reaction":
            # e.g. "right_hand_4pos_10x"
            bp = cfg.react_track_body_part.replace("-", "_")
            return f"{bp}_{cfg.react_n_positions}pos_{cfg.n_stimuli}x"
        return ""

    def _prepare_reaction_pool(self) -> None:
        # Phase V6 — cognitive_reaction shares the same scheduling
        # machinery as classic ``reaction``; the difference is the
        # response-pool keys (positional pos_*) and that the analyzer
        # extracts the response from pose data automatically.
        if self.cfg.test == "cognitive_reaction":
            n_pos = self.cfg.react_n_positions
            pool_table = (COGNITIVE_REACTION_POSITIONS_8
                          if n_pos >= 8 else
                          COGNITIVE_REACTION_POSITIONS_4)
            self._response_pool = [k for (k, _, _) in pool_table]
            # Cache the position lookup for stim event metadata.
            self._cog_pos_lookup = {k: (x, y) for (k, x, y) in pool_table}
            self._log(
                f"cognitive_reaction pool: {self._response_pool} "
                f"(track={self.cfg.react_track_body_part})")
            if self.cfg.trigger == "auto":
                rng = random.Random()
                t_cur = 2.0
                for _ in range(self.cfg.n_stimuli):
                    t_cur += rng.uniform(self.cfg.stim_min_gap,
                                          self.cfg.stim_max_gap)
                    if t_cur >= self.cfg.duration_s - 1:
                        break
                    self._stim_times.append(
                        (t_cur, rng.choice(self._response_pool)))
                self._log(f"scheduled {len(self._stim_times)} auto stimuli")
            return

        if self.cfg.test != "reaction":
            return
        if self.cfg.responses == "random":
            self._response_pool = list(REACTION_RESPONSES.keys())
        else:
            self._response_pool = [
                r.strip() for r in self.cfg.responses.split(",")
                if r.strip() in REACTION_RESPONSES
            ]
        if not self._response_pool:
            raise RuntimeError(
                f"--responses produced empty pool: {self.cfg.responses}")
        self._log(f"reaction pool: {self._response_pool} "
                  f"(trigger={self.cfg.trigger})")

        # Pre-schedule auto-mode stimuli
        if self.cfg.trigger == "auto":
            rng = random.Random()
            t_cur = 2.0
            for _ in range(self.cfg.n_stimuli):
                t_cur += rng.uniform(self.cfg.stim_min_gap, self.cfg.stim_max_gap)
                if t_cur >= self.cfg.duration_s - 1:
                    break
                self._stim_times.append((t_cur, rng.choice(self._response_pool)))
            self._log(f"scheduled {len(self._stim_times)} auto stimuli")

    def _start_hardware(self) -> None:
        # DAQ
        self._daq = DaqReader()
        if self.cfg.use_smart_wait:
            stance_mode = self.cfg.stance if self.cfg.test in (
                "balance_eo", "balance_ec") else "two"
            # Let StabilityDetector pick stance-appropriate hold time:
            # two=3.0s, left/right=0.5s (easier for single-leg).
            self._stability = StabilityDetector(
                subject_kg=self.cfg.subject_kg,
                timeout_s=self.cfg.wait_timeout_s,
                stance_mode=stance_mode,
            )
        if self._daq.connect():
            def _on_daq(fr: DaqFrame) -> None:
                # Single reference assignment is atomic in CPython, so
                # no lock is needed for the UI-facing latest frame.
                self._daq_latest_frame = fr
                if self._record_ready.is_set():
                    # Classify on/off-plate at write-time using the same
                    # 20 N threshold the CMJ analyser uses for takeoff /
                    # landing detection — replay's departure bands and
                    # the CMJ flight shading then mark the same instants.
                    on_plate = classify_on_plate(
                        fr.total_n, self._dep_threshold_n)
                    with self._daq_lock:
                        self._daq_frames.append(fr)
                        self._cop_states.append(on_plate)
                    # Streaming event tracker — needs t relative to
                    # record_start so events line up with t_s in CSVs.
                    if self._rec_start_wall is not None:
                        self._dep_tracker.update(
                            on_plate=on_plate,
                            t_s=fr.t_wall - self._rec_start_wall,
                            t_wall=fr.t_wall,
                        )
                if self._on_daq is not None:
                    try:
                        self._on_daq(fr)
                    except Exception:
                        pass
            self._daq.set_callback(_on_daq)
            self._daq.start()
            self._daq_connected = True
            self._log("DAQ streaming (zero-cal ~5s; keep OFF the plate)")
        else:
            self._log("DAQ connect failed - video only")
            self._stability = None

        # Cameras
        self._cam = MultiCameraCapture(record_dir=self.session_dir)
        self._cam.start()

    # ── main loop ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        cfg = self.cfg
        # Initial phase
        if self._stability is not None:
            self._state.phase = "wait"
            # Give DAQ ~5 s to finish zero-cal
            stability_arm_ns = time.monotonic_ns() + int(5.5e9)
        else:
            self._state.phase = "countdown"
            stability_arm_ns = None
        self._t_phase_ns = time.monotonic_ns()
        self._emit_state()

        poll_s = 0.02   # ~50 Hz state tick (UI runs at own rate)

        while not self._cancel.is_set():
            now_ns = time.monotonic_ns()

            # Drain camera queue → forward frames via callback
            self._drain_cameras()

            # Operator-pressed end-session has highest priority — it can
            # land in any phase and immediately concludes the session
            # with whatever sets/data have been recorded so far.
            if self._trig_end_session and self._state.phase != "done":
                self._handle_end_session(now_ns)

            if self._state.phase == "wait":
                self._tick_wait(now_ns, stability_arm_ns)
            elif self._state.phase == "countdown":
                self._tick_countdown(now_ns)
            elif self._state.phase == "recording":
                self._tick_recording(now_ns)
            elif self._state.phase == "inter_set_rest":
                self._tick_inter_set_rest(now_ns)
            elif self._state.phase in ("done", "cancelled"):
                break

            self._emit_state()
            time.sleep(poll_s)

        if self._cancel.is_set() and self._state.phase not in ("done",):
            self._state.phase = "cancelled"
            self._emit_state()

    def _drain_cameras(self) -> None:
        if self._cam is None:
            return
        drained = 0
        while drained < 30:
            item = self._cam.get(timeout=0.003)
            if item is None:
                break
            if "error" in item:
                continue
            if self._on_cam is not None:
                try:
                    self._on_cam(item["cam_id"], item["bgr"],
                                 int(item.get("frame_idx", 0)),
                                 int(item.get("t_ns", 0)))
                except Exception:
                    pass
            drained += 1

    def _tick_wait(self, now_ns: int, stability_arm_ns: Optional[int]) -> None:
        phase_s = (now_ns - self._t_phase_ns) / 1e9
        self._state.elapsed_s = phase_s
        # Zero-cal window: show countdown, skip stability evaluation
        if stability_arm_ns is not None and now_ns < stability_arm_ns:
            self._state.zeroing = True
            self._state.zero_cal_remaining_s = max(
                0.0, (stability_arm_ns - now_ns) / 1e9)
            self._state.wait = None
            return
        # Zero-cal done — from here on, zeroing stays False. Even if the DAQ
        # momentarily has no fresh frame, we don't flip the banner back.
        self._state.zeroing = False
        self._state.zero_cal_remaining_s = 0.0
        frame = self._daq_latest_frame
        if frame is None:
            return
        stab = self._stability.update(frame)
        self._state.wait = stab
        if stab.status == "READY":
            self._wait_duration_s = phase_s
            self._transition_to_recording(now_ns)
        elif stab.status == "TIMEOUT":
            self._log(f"wait timed out after {self.cfg.wait_timeout_s:.0f} s")
            self._cancel.set()

    def _tick_countdown(self, now_ns: int) -> None:
        phase_s = (now_ns - self._t_phase_ns) / 1e9
        self._state.elapsed_s = phase_s
        if phase_s >= self.cfg.countdown_s:
            self._transition_to_recording(now_ns)

    def _transition_to_recording(self, now_ns: int) -> None:
        """Enter the recording phase. For single-set tests this is the
        one-and-only recording. For ``strength_3lift`` this is invoked
        again at the start of each set (set 0, 1, 2, ...); _record_ready,
        _daq_frames and rec_start are only initialised on the FIRST set
        so the entire session is one continuous DAQ stream + one mp4."""
        is_first_recording = (self._rec_start_wall is None)
        self._state.phase = "recording"
        self._t_phase_ns = now_ns

        if is_first_recording:
            self._rec_start_ns   = time.monotonic_ns()
            self._rec_start_wall = time.time()
            self._record_ready.set()
            with self._daq_lock:
                self._daq_frames.clear()
                self._cop_states.clear()
            # Re-arm the departure tracker so any wait-phase off-plate
            # state is forgotten now that the timed window starts. The
            # threshold itself is fixed (subject_kg-derived) and survives
            # across resets.
            self._dep_tracker = DepartureEventTracker(min_duration_s=0.05)
            self._log(
                f"RECORDING STARTED (waited {self._wait_duration_s:.1f} s)")

        # Multi-set: track when the current set started so we can emit
        # a per-set boundary record when end_set() fires. ``set_t_start_s``
        # is in session-relative seconds (0 = rec_start_ns).
        if self.cfg.test == "strength_3lift":
            self._set_t_start_s = (
                (time.monotonic_ns() - self._rec_start_ns) / 1e9)
            self._state.current_set_idx = self._current_set_idx
            self._state.n_sets = self.cfg.n_sets
            self._log(
                f"SET {self._current_set_idx + 1}/{self.cfg.n_sets} "
                f"started ({'warmup' if self._is_current_set_warmup() else 'working'})")

    def _tick_recording(self, now_ns: int) -> None:
        phase_s = (now_ns - self._t_phase_ns) / 1e9
        self._state.elapsed_s = phase_s

        # Auto-scheduled reaction stimuli
        while self._stim_times and phase_s >= self._stim_times[0][0]:
            _, resp = self._stim_times.pop(0)
            self._fire_stim(resp, now_ns)

        # Drain manual reaction / cue queue
        with self._pending_lock:
            pending = self._pending_manual[:]
            self._pending_manual.clear()
        for resp in pending:
            # Reaction tests are bounded by n_stimuli; squat/overhead_squat
            # VRT cues (``squat_ascent``) fire one-per-rep without a cap.
            if self.cfg.test == "reaction" \
                    and len(self._stim_events) < self.cfg.n_stimuli:
                self._fire_stim(resp, now_ns)
            elif self.cfg.test in ("squat", "overhead_squat") \
                    and resp == "squat_ascent":
                self._fire_stim(resp, now_ns)

        # Banner expiry
        if now_ns / 1e9 >= self._stim_banner_until:
            self._state.stim_banner = None
            self._stim_banner_type = None

        # Fall-off detection for balance tests: total_n below 30%BW for
        # >= 2 s contiguous → abort the recording. Rationale: a subject
        # who stepped off (or was carried off) generates near-zero force;
        # the remainder of the trial would be meaningless anyway.
        if self.cfg.test in ("balance_eo", "balance_ec") \
                and self._daq_latest_frame is not None:
            threshold_n = 0.30 * self.cfg.subject_kg * 9.80665
            total = self._daq_latest_frame.total_n
            if total < threshold_n:
                if self._fall_off_since_ns is None:
                    self._fall_off_since_ns = now_ns
                elif (now_ns - self._fall_off_since_ns) / 1e9 >= 2.0:
                    self._fell_off = True
                    self._log(
                        f"FELL_OFF — total {total:.0f} N < "
                        f"{threshold_n:.0f} N for 2 s; aborting")
                    self._cancel.set()
                    return
            else:
                self._fall_off_since_ns = None

        # ── Multi-set strength: operator-driven set end ────────────────
        # The "세트 종료" button sets _trig_end_set. We close the current
        # set and either go to inter_set_rest (more sets coming) or to
        # done (this was the last set).
        if self.cfg.test == "strength_3lift" and self._trig_end_set:
            self._trig_end_set = False
            self._close_current_set(phase_s)
            if self._current_set_idx + 1 >= self.cfg.n_sets:
                # Last set just finished — done.
                self._log("ALL SETS COMPLETE")
                self._state.phase = "done"
            else:
                # More sets remain — start inter-set rest.
                self._enter_inter_set_rest(now_ns)
            return

        # End conditions
        manual_done = (self.cfg.test == "reaction"
                       and self.cfg.trigger == "manual"
                       and len(self._stim_events) >= self.cfg.n_stimuli)
        if self.cfg.test == "strength_3lift":
            # Multi-set sessions have NO duration_s timeout. The operator
            # drives transitions via end_set() / end_session() exclusively.
            # This prevents the recording from auto-stopping mid-rep when
            # a slow set runs long.
            return
        if phase_s >= self.cfg.duration_s or manual_done:
            self._state.phase = "done"

    # ── Multi-set strength assessment helpers (Phase V1-D) ─────────────
    def _is_current_set_warmup(self) -> bool:
        """True when the current set should be flagged as warmup
        (excluded from the 1RM estimate). Per the plan, only set 0
        is the warmup, and only when ``cfg.warmup_set`` is True."""
        return bool(self.cfg.warmup_set and self._current_set_idx == 0)

    def _close_current_set(self, phase_s: float) -> None:
        """Append a boundary record for the set that just finished.

        ``phase_s`` is the current ``recording`` phase elapsed time —
        we convert to session-relative seconds (0 = rec_start_ns).
        """
        # Session-relative end time (consistent with how forces.csv's
        # t_wall - rec_start gets exposed in analysis).
        t_end_s = (time.monotonic_ns() - self._rec_start_ns) / 1e9
        if self._set_t_start_s is None:
            self._log(
                f"WARN: _close_current_set with no _set_t_start_s "
                f"— skipping boundary record")
            self._set_t_start_s = None
            return
        rec = {
            "set_idx":   self._current_set_idx,
            "t_start_s": round(self._set_t_start_s, 4),
            "t_end_s":   round(t_end_s, 4),
            "warmup":    self._is_current_set_warmup(),
            "load_kg":   float(self.cfg.load_kg),
            "exercise":  self.cfg.exercise,
        }
        self._sets.append(rec)
        self._set_t_start_s = None
        self._log(
            f"SET {rec['set_idx'] + 1}/{self.cfg.n_sets} closed: "
            f"{rec['t_end_s'] - rec['t_start_s']:.1f}s"
            f"{' (warmup)' if rec['warmup'] else ''}")

    def _enter_inter_set_rest(self, now_ns: int) -> None:
        """Transition recording → inter_set_rest. Sets up the wall-clock
        anchored countdown — start time + accumulated pause time."""
        self._state.phase = "inter_set_rest"
        self._t_phase_ns = now_ns
        self._rest_t0_ns = time.monotonic_ns()
        self._rest_pause_accum_ns = 0
        self._rest_paused_at_ns = None
        self._state.rest_remaining_s = float(self.cfg.rest_s)
        self._state.rest_paused = False
        self._log(f"REST started ({self.cfg.rest_s:.0f}s)")

    def _tick_inter_set_rest(self, now_ns: int) -> None:
        """Countdown the rest timer. Auto-transitions to next set when
        elapsed ≥ rest_s. Pause/resume/skip handled here too."""
        # Skip-rest trigger: jump straight to next set.
        if self._trig_skip_rest:
            self._trig_skip_rest = False
            self._log("REST skipped - starting next set")
            self._current_set_idx += 1
            self._transition_to_recording(now_ns)
            return
        # Compute elapsed rest time, excluding accumulated pause.
        if self._rest_paused_at_ns is not None:
            # Currently paused — elapsed freezes at the pause moment.
            effective_now = self._rest_paused_at_ns
        else:
            effective_now = time.monotonic_ns()
        elapsed_ns = (effective_now - self._rest_t0_ns
                      - self._rest_pause_accum_ns)
        elapsed_s = max(0.0, elapsed_ns / 1e9)
        remaining = max(0.0, float(self.cfg.rest_s) - elapsed_s)
        self._state.rest_remaining_s = remaining
        self._state.elapsed_s = elapsed_s
        # Auto-transition to next set when timer expires.
        if remaining <= 0.0 and self._rest_paused_at_ns is None:
            self._log(f"REST complete - starting next set")
            self._current_set_idx += 1
            self._transition_to_recording(now_ns)

    def _handle_end_session(self, now_ns: int) -> None:
        """Operator-pressed "세션 종료" — finalize from any phase.

        Distinct from cancel(): we keep all the data captured so far.
        If we're in the middle of a set, close it first so the
        boundary record is preserved.
        """
        self._trig_end_session = False
        if self._state.phase == "recording":
            phase_s = (now_ns - self._t_phase_ns) / 1e9
            self._close_current_set(phase_s)
        elif self._state.phase == "inter_set_rest":
            # No active set to close — just stop the timer.
            pass
        self._log("SESSION ended by operator")
        self._state.phase = "done"

    def _fire_stim(self, response_type: str, now_ns: int) -> None:
        ev = {
            "trial_idx":     len(self._stim_events),
            "t_wall":        time.time(),
            "t_ns":          time.monotonic_ns(),
            "stimulus_type": "audio_visual",
            "response_type": response_type,
        }
        # Phase V6 — attach normalised target XY for cognitive_reaction
        # so the offline analyzer can compute spatial accuracy without
        # consulting the position lookup table separately. The cue is
        # held on screen for ~1.5 s (longer than reaction's 0.5 s) so
        # the subject has time to reach.
        cog_lookup = getattr(self, "_cog_pos_lookup", None)
        if (self.cfg.test == "cognitive_reaction"
                and cog_lookup is not None
                and response_type in cog_lookup):
            tx, ty = cog_lookup[response_type]
            ev["target_x_norm"] = tx
            ev["target_y_norm"] = ty
            ev["target_label"]  = response_type
        self._stim_events.append(ev)
        self._state.n_stim_fired = len(self._stim_events)
        # Hold the visual cue longer for cognitive_reaction (subject
        # needs time to physically reach the position) than for the
        # classic reaction test where the response is a single jump.
        banner_hold_s = 1.5 if self.cfg.test == "cognitive_reaction" else 0.5
        self._stim_banner_until  = now_ns / 1e9 + banner_hold_s
        self._stim_banner_type   = response_type
        label = REACTION_RESPONSES.get(response_type, ("NOW!", (0, 0, 255)))[0]
        self._state.stim_banner = label
        self._beep(1500, 120)
        self._log(f"STIM #{len(self._stim_events)} response={response_type}")

    # ── finalisation ────────────────────────────────────────────────────────
    def _finalize(self) -> dict:
        cancelled = (self._state.phase == "cancelled") or self._cancel.is_set()

        # ── Atomic record-window close (Phase U3-2) ────────────────────
        # Three things have to happen back-to-back BEFORE we wait on any
        # blocking hardware stop, so neither stream leaks samples past
        # the declared end of recording:
        #
        #   1. Snapshot the wall instant of "recording ended". This
        #      becomes the canonical upper bound for both force and
        #      camera timestamps. Persisted to session.json so replay
        #      can use it instead of guessing from the longer of the
        #      two streams.
        #
        #   2. Clear ``_record_ready`` — the DAQ callback checks this
        #      flag before appending to ``_daq_frames``, so any sample
        #      still arriving from the DAQ thread after this point is
        #      dropped at the gate. (Single-flag check is atomic in
        #      CPython; no lock needed for clear()/is_set() pair.)
        #
        #   3. Signal the camera worker to stop. ``signal_stop()`` is
        #      non-blocking (event-set only); the actual join happens
        #      below after both streams have already been told to halt.
        #
        # Without this protocol, the previous ``cam.stop()`` call below
        # blocked for up to 2 s waiting for the camera process join,
        # during which the DAQ callback merrily kept appending ~200
        # extra samples — producing a forces.csv that ran ~2 s past
        # the last video frame and a replay slider tail where the
        # video froze on its last frame.
        self._rec_end_wall = time.time()
        self._rec_end_ns   = time.monotonic_ns()
        self._record_ready.clear()
        try:
            if self._cam is not None:
                self._cam.signal_stop()
        except Exception as e:
            self._log(f"cam signal error: {e}")

        # ── Hardware shutdown (writers flush) ──────────────────────────
        # DAQ first — it's stream-driven, stop() returns immediately
        # once the read thread joins. Camera join can take up to 2 s
        # while the worker process flushes its mp4 writer.
        try:
            if self._daq is not None and self._daq_connected:
                self._daq.stop()
        except Exception as e:
            self._log(f"daq stop error: {e}")
        try:
            if self._cam is not None:
                self._cam.wait_join(timeout=2.0)
        except Exception as e:
            self._log(f"cam join error: {e}")

        # ── Defense-in-depth: post-hoc trim DAQ tail ───────────────────
        # The clear() above is the primary line of defense, but a DAQ
        # callback that was already mid-flight (had passed the
        # ``_record_ready.is_set()`` check but not yet executed
        # ``_daq_frames.append``) could slip through. Drop any frame
        # whose wall time falls beyond our record_end snapshot.
        self._trim_daq_tail()

        # forces.csv
        if self._daq_connected and self._daq_frames and self.session_dir:
            self._write_forces_csv(self.session_dir / "forces.csv")

        # stimulus_log.csv
        if self._stim_events and self.session_dir:
            self._write_stim_log(self.session_dir / "stimulus_log.csv")

        # ── Departure events (Phase U3-3) ──────────────────────────────
        # Close any still-open off-plate interval and write events.csv +
        # cache the summary so _build_metadata can include it.
        dep_events = self._dep_tracker.finalize()
        self._departures_summary = self._dep_tracker.summary()
        # Always include the threshold used so consumers know how to
        # interpret the events (and could re-derive with a different
        # threshold if they want).
        self._departures_summary["threshold_n"] = round(
            self._dep_threshold_n, 2)
        if dep_events and self.session_dir:
            self._write_departure_events(
                self.session_dir / "events.csv", dep_events)

        # session.json
        meta = self._build_metadata(cancelled)
        if self.session_dir:
            (self.session_dir / "session.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8")

        # Final state
        self._state.phase = "cancelled" if cancelled else "done"
        self._emit_state()

        self._log(f"complete: {self.session_dir}")
        return {
            "session_dir": str(self.session_dir) if self.session_dir else None,
            "session_name": self.session_name,
            "cancelled": cancelled,
            "n_daq_samples": len(self._daq_frames),
            "n_stimuli":     len(self._stim_events),
            "wait_duration_s": self._wait_duration_s,
            "meta": meta,
        }

    def _trim_daq_tail(self) -> None:
        """Drop DAQ frames whose wall time exceeds ``_rec_end_wall``.

        Defense-in-depth complement to the ``_record_ready.clear()`` at
        the top of ``_finalize``. That gate stops the DAQ callback from
        appending new frames, but a callback already mid-flight could
        slip a frame through. Removing the tail here guarantees that
        forces.csv's last sample is no later than ``record_end_wall_s``.

        No-op when ``_rec_end_wall`` is None (e.g. session finalized
        before any recording occurred) or when no frames lie past the
        cutoff.
        """
        if self._rec_end_wall is None or not self._daq_frames:
            return
        cut = float(self._rec_end_wall)
        n_before = len(self._daq_frames)
        # Trim frames + parallel cop_states list together so indices stay
        # aligned. Done with a single comprehension over zipped pairs.
        if len(self._cop_states) == len(self._daq_frames):
            paired = [
                (fr, st) for fr, st in zip(self._daq_frames, self._cop_states)
                if fr.t_wall <= cut
            ]
            self._daq_frames = [p[0] for p in paired]
            self._cop_states = [p[1] for p in paired]
        else:
            # Defensive fallback if the parallel list got out of sync
            # for any reason — drop states entirely rather than risk
            # a misaligned column in forces.csv.
            self._daq_frames = [
                fr for fr in self._daq_frames if fr.t_wall <= cut
            ]
            self._cop_states = []
        n_dropped = n_before - len(self._daq_frames)
        if n_dropped > 0:
            self._log(
                f"trimmed {n_dropped} DAQ tail samples past record_end "
                f"(t_wall > {cut:.3f})")

    def _write_forces_csv(self, path: Path) -> None:
        # ``on_plate`` column added in Phase U3-3 — 0/1 per sample,
        # classified at write-time using the fixed 20 N threshold
        # (matches CMJ analyser's flight_threshold_n). Older sessions
        # don't have this column; load_force_session back-fills from
        # ``total_n >= 20`` when absent.
        states_aligned = (len(self._cop_states) == len(self._daq_frames))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_ns", "t_wall",
                "b1_tl_N", "b1_tr_N", "b1_bl_N", "b1_br_N",
                "b2_tl_N", "b2_tr_N", "b2_bl_N", "b2_br_N",
                "enc1_mm", "enc2_mm",
                "total_n", "cop_world_x_mm", "cop_world_y_mm",
                "on_plate",
            ])
            for i, fr in enumerate(self._daq_frames):
                cx, cy = fr.cop_world_mm()
                cx_s = "" if np.isnan(cx) else f"{cx:.2f}"
                cy_s = "" if np.isnan(cy) else f"{cy:.2f}"
                on_plate = (self._cop_states[i] if states_aligned
                            else int(fr.total_n >= self._dep_threshold_n))
                w.writerow([
                    fr.t_ns, f"{fr.t_wall:.6f}",
                    *[f"{v:.3f}" for v in fr.forces_n],
                    f"{fr.enc1_mm:.3f}", f"{fr.enc2_mm:.3f}",
                    f"{fr.total_n:.3f}", cx_s, cy_s,
                    on_plate,
                ])
        self._log(f"forces.csv: {len(self._daq_frames)} samples")

    def _write_departure_events(self, path: Path,
                                events: list[dict]) -> None:
        """Write events.csv listing each off-plate interval.

        Columns mirror the dict keys produced by DepartureEventTracker
        (see ``DepartureEvent.to_row``). Empty file (header only) is
        written when no events qualified — keeps the schema discoverable
        for downstream tools.
        """
        cols = ["trial_idx", "t_start_s", "t_end_s",
                "t_start_wall", "t_end_wall",
                "duration_s", "n_samples"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for ev in events:
                w.writerow([ev[c] for c in cols])
        self._log(f"events.csv: {len(events)} departure events")

    def _write_stim_log(self, path: Path) -> None:
        # Phase V6 — when the test is cognitive_reaction we add three
        # extra columns (target_x_norm, target_y_norm, target_label) so
        # the offline analyzer can map each stim to a target XY without
        # rebuilding the position lookup. For other tests the schema
        # stays identical to pre-V6 — keeps replay/legacy parsing intact.
        is_cog = (self.cfg.test == "cognitive_reaction")
        cols = ["trial_idx", "t_wall", "t_ns",
                "stimulus_type", "response_type"]
        if is_cog:
            cols += ["target_x_norm", "target_y_norm", "target_label"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for e in self._stim_events:
                row = [e["trial_idx"], f"{e['t_wall']:.6f}",
                       e["t_ns"], e["stimulus_type"],
                       e.get("response_type", "")]
                if is_cog:
                    tx = e.get("target_x_norm")
                    ty = e.get("target_y_norm")
                    row += [
                        "" if tx is None else f"{tx:.4f}",
                        "" if ty is None else f"{ty:.4f}",
                        e.get("target_label", ""),
                    ]
                w.writerow(row)
        self._log(f"stimulus_log.csv: {len(self._stim_events)} events")

    def _build_metadata(self, cancelled: bool) -> dict:
        cfg = self.cfg
        vision = None
        if cfg.test == "balance_eo": vision = "open"
        elif cfg.test == "balance_ec": vision = "closed"
        stance_mode = cfg.stance if cfg.test in (
            "balance_eo", "balance_ec") else "two"
        return {
            "name":              self.session_name,
            "test":              cfg.test,
            "duration_s":        cfg.duration_s,
            "cancelled":         cancelled,
            "fell_off_detected": self._fell_off,
            "cameras":           config.CAMERAS if self._cam is not None else [],
            "daq_connected":     self._daq_connected,
            "n_daq_samples":     len(self._daq_frames),
            "n_stimuli":         len(self._stim_events),
            "record_start_monotonic_ns": self._rec_start_ns,
            "record_start_wall_s":       self._rec_start_wall,
            # Snapshot of recording-end wall instant — captured at the
            # top of _finalize() before hardware shutdown blocks. Used by
            # ReplayPanel as the slider's right edge so force/camera
            # streams can be tail-trimmed to a common bound (Phase U3-2).
            "record_end_monotonic_ns":   self._rec_end_ns,
            "record_end_wall_s":         self._rec_end_wall,
            # Aggregated off-plate / departure statistics computed by
            # DepartureEventTracker (Phase U3-3). Empty dict when the
            # session was cancelled before any DAQ samples landed.
            "departures":                self._departures_summary or None,
            "wait_duration_s":   self._wait_duration_s,
            "subject_id":        cfg.subject_id,
            "subject_name":      cfg.subject_name,
            "subject_kg":        cfg.subject_kg,
            # Phase V1-bugfix — sex/birthdate must reach the analyzer
            # via session.json (not only via the live DB) so analysis
            # of older sessions stays reproducible if the DB rotates.
            "subject_sex":       (cfg.subject_sex.upper()
                                  if isinstance(cfg.subject_sex, str)
                                  and cfg.subject_sex else None),
            "subject_birthdate": cfg.subject_birthdate,
            "subject_height_cm": cfg.subject_height_cm,
            "smart_wait":        self._stability is not None,
            "vision":            vision,
            "stance":            stance_mode,
            "reaction_trigger":  cfg.trigger    if cfg.test == "reaction" else None,
            "reaction_responses": self._response_pool if cfg.test == "reaction" else None,
            # ── Cognitive reaction (Phase V6) ─────────────────────────
            # Persisted so the analyzer + replay know what body part was
            # tracked and which positions were in the rotation. Shape of
            # cog_positions: list of [label, x_norm, y_norm].
            "cog_track_body_part":
                cfg.react_track_body_part if cfg.test == "cognitive_reaction" else None,
            "cog_n_positions":
                cfg.react_n_positions     if cfg.test == "cognitive_reaction" else None,
            "cog_trigger":
                cfg.trigger               if cfg.test == "cognitive_reaction" else None,
            "cog_positions": (
                [[k, x, y] for k, (x, y) in (
                    getattr(self, "_cog_pos_lookup", {}) or {}).items()]
                if cfg.test == "cognitive_reaction" else None),
            "encoder_prompt":    cfg.encoder_prompt if cfg.test == "encoder" else None,
            # Free exercise — load_kg has already been adjusted by
            # use_bodyweight_load in RecorderConfig.__post_init__, so the
            # saved value is the effective load used during recording.
            "exercise_name":        cfg.exercise_name       if cfg.test == "free_exercise" else None,
            "load_kg":              cfg.load_kg             if cfg.test == "free_exercise" else None,
            "use_bodyweight_load":  cfg.use_bodyweight_load if cfg.test == "free_exercise" else None,
            # Encoder hardware usage — None for balance tests (flag doesn't
            # apply). Consumers should default to True when absent so old
            # sessions (pre-G3) keep working.
            "uses_encoder": (None if cfg.test in ("balance_eo", "balance_ec")
                              else cfg.uses_encoder),
            # ── Strength 3-lift multi-set (Phase V1-D) ────────────────
            "exercise":     cfg.exercise    if cfg.test == "strength_3lift" else None,
            "n_sets":       cfg.n_sets      if cfg.test == "strength_3lift" else None,
            "target_reps":  cfg.target_reps if cfg.test == "strength_3lift" else None,
            "rest_s":       cfg.rest_s      if cfg.test == "strength_3lift" else None,
            "warmup_set":   cfg.warmup_set  if cfg.test == "strength_3lift" else None,
            # Phase V1.5 — when True, the analyzer adds an
            # exercise-specific fraction of bodyweight to load_kg as
            # the effective 1RM input (see strength_norms.EXERCISE_BW_FACTOR).
            "strength_use_bw_load": (cfg.use_bodyweight_load
                                       if cfg.test == "strength_3lift"
                                       else None),
            # Per-set boundary records — list of dicts with set_idx,
            # t_start_s (relative to record_start), t_end_s, warmup,
            # load_kg, exercise. Empty list when not a multi-set test
            # or when the session was cancelled before any set finished.
            "sets":         self._sets if cfg.test == "strength_3lift" else None,
        }

    # ── misc helpers ────────────────────────────────────────────────────────
    def _build_prompt(self) -> str:
        base = TEST_PROMPTS.get(self.cfg.test, self.cfg.test)
        if self.cfg.test in ("balance_eo", "balance_ec"):
            suffix = STANCE_LABEL.get(self.cfg.stance, "")
            return f"{base}  -  {suffix}" if suffix else base
        if self.cfg.test == "encoder" and self.cfg.encoder_prompt:
            return f"ENCODER - {self.cfg.encoder_prompt}"
        if self.cfg.test == "free_exercise":
            name = (self.cfg.exercise_name or "free exercise").strip()
            load = self.cfg.load_kg
            if load > 0:
                return f"{name.upper()} - load {load:.1f} kg"
            return name.upper()
        return base

    def _emit_state(self) -> None:
        self._state.prompt = self._build_prompt()
        if self._on_state is not None:
            try:
                self._on_state(self._state)
            except Exception:
                pass

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(msg)
                return
            except Exception:
                pass
        print(f"[rec] {msg}", flush=True)

    @staticmethod
    def _beep(freq: int = 1000, ms: int = 180) -> None:
        def _b():
            try:
                import winsound
                winsound.Beep(freq, ms)
            except Exception:
                pass
        threading.Thread(target=_b, daemon=True).start()
