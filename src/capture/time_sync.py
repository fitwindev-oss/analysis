"""
Time-based alignment of multi-camera pose sequences using per-frame
monotonic timestamps (from the capture worker).

Problem:
    Each USB webcam drops/duplicates frames independently. Even though all
    three cameras are set to 30 fps, after 60 seconds you typically get
    different frame counts per camera:  e.g.  C0: 1769, C1: 1624, C2: 1849.
    So "frame 100 in C0" and "frame 100 in C1" are NOT the same moment in
    time; naive index alignment gives ~0.3-1s drift by the end of the clip.

Fix:
    Pick one camera (the slowest - smallest frame count - is usually safest)
    as the time reference. For each reference timestamp, find the nearest
    timestamp in every other camera within MAX_DT_NS.  If any camera has no
    match within tolerance, drop that reference frame.

    Returns aligned keypoint/score arrays plus the mapping from aligned row
    -> each camera's original frame index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MAX_DT_NS = 20_000_000   # 20 ms  (~0.6 of a 30 fps frame interval)


def load_timestamps(session_dir: Path, cam_ids: list[str]) -> dict[str, np.ndarray]:
    """Read per-camera timestamps.csv. Returns cam_id -> (N,) int64 ns."""
    out: dict[str, np.ndarray] = {}
    for cid in cam_ids:
        path = session_dir / f"{cid}.timestamps.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing: {path}")
        df = pd.read_csv(path)
        out[cid] = df["t_monotonic_ns"].to_numpy(dtype=np.int64)
    return out


def align_pose_sequences(
    pose_seqs: dict,                    # cam_id -> PoseSequence
    timestamps: dict[str, np.ndarray],  # cam_id -> t_ns
    reference_cam: str | None = None,
    max_dt_ns: int = DEFAULT_MAX_DT_NS,
) -> dict:
    """
    Align all per-camera pose sequences to a common time axis.

    Returns a dict with:
        kpts[cam]:   (M, 17, 2)
        scores[cam]: (M, 17)
        frame_map[cam]: (M,) original per-camera frame index
        t_ref_ns:    (M,) monotonic ns of reference camera frames
        reference_cam: str
        stats: dict (diagnostic counts)
    """
    cam_ids = list(pose_seqs.keys())
    # Pick reference = shortest recording (slowest) to maximize matches.
    if reference_cam is None:
        reference_cam = min(cam_ids,
                            key=lambda c: len(pose_seqs[c].keypoints))

    # Timestamps may be slightly longer than kpts (race at shutdown). Trim.
    for cid in cam_ids:
        n_kp = len(pose_seqs[cid].keypoints)
        if len(timestamps[cid]) < n_kp:
            raise RuntimeError(
                f"{cid}: {len(timestamps[cid])} timestamps < {n_kp} frames"
            )
        timestamps[cid] = timestamps[cid][:n_kp]

    ref_ts = timestamps[reference_cam]
    # All timestamps come from time.monotonic_ns() in the SAME process, so
    # they share the same epoch and are directly comparable.

    # For each ref frame, pick the nearest frame in each other cam.
    aligned_kpts:  dict[str, list] = {c: [] for c in cam_ids}
    aligned_scores: dict[str, list] = {c: [] for c in cam_ids}
    frame_map: dict[str, list] = {c: [] for c in cam_ids}
    matched_t_ref: list[int] = []

    other_ids = [c for c in cam_ids if c != reference_cam]
    max_dt = int(max_dt_ns)

    stats = {
        "n_ref": int(len(ref_ts)),
        "dropped_no_match": 0,
        "dropped_dt_too_large": 0,
    }

    for ri, t_ref in enumerate(ref_ts):
        picks = {reference_cam: ri}
        ok = True
        for cid in other_ids:
            ts = timestamps[cid]
            j = int(np.searchsorted(ts, t_ref))
            # test j and j-1 for closest
            cand = []
            if j < len(ts):
                cand.append((j, int(abs(ts[j] - t_ref))))
            if j > 0:
                cand.append((j - 1, int(abs(ts[j - 1] - t_ref))))
            if not cand:
                ok = False
                stats["dropped_no_match"] += 1
                break
            best_j, best_dt = min(cand, key=lambda x: x[1])
            if best_dt > max_dt:
                ok = False
                stats["dropped_dt_too_large"] += 1
                break
            picks[cid] = best_j
        if not ok:
            continue
        matched_t_ref.append(int(t_ref))
        for cid in cam_ids:
            fi = picks[cid]
            aligned_kpts[cid].append(pose_seqs[cid].keypoints[fi])
            aligned_scores[cid].append(pose_seqs[cid].scores[fi])
            frame_map[cid].append(fi)

    stats["matched"] = len(matched_t_ref)

    out = {
        "kpts":   {c: np.stack(aligned_kpts[c]).astype(np.float32)
                   for c in cam_ids},
        "scores": {c: np.stack(aligned_scores[c]).astype(np.float32)
                   for c in cam_ids},
        "frame_map": {c: np.asarray(frame_map[c], dtype=np.int32)
                      for c in cam_ids},
        "t_ref_ns": np.asarray(matched_t_ref, dtype=np.int64),
        "reference_cam": reference_cam,
        "stats": stats,
    }
    return out
