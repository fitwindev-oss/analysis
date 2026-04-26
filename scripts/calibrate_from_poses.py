"""
Full skeleton-based calibration of 3 cameras from a recorded session.

Prerequisites:
    - A session recorded with scripts/record_calibration_session.py
    - rtmlib installed (see requirements.txt)

Workflow:
    1. Load the 3 videos from data/calibration/session_*/
    2. Run 2D pose detection (RTMPose via rtmlib) on each video. Results
       are cached to `poses_C*.npz` for fast re-runs.
    3. Pick frames where all 3 cameras have confident detections.
    4. Initialize cam0-cam1 via Essential matrix; triangulate 3D points;
       solve PnP for cam2.
    5. Optional bundle adjustment for joint refinement.
    6. Save intrinsics + extrinsics (UNSCALED - arbitrary global scale).

Next step:
    scripts/align_world_from_force.py  will use the force plate CoP to
    resolve the scale and align axes to world coordinates.

Usage:
    python scripts/calibrate_from_poses.py --session session_20260421_150000
    python scripts/calibrate_from_poses.py --session ... --conf 0.40
    python scripts/calibrate_from_poses.py --session ... --no-ba  (skip BA)
    python scripts/calibrate_from_poses.py --session ... --redetect

Output (into data/calibration/):
    intrinsics_C0.npz  intrinsics_C1.npz  intrinsics_C2.npz
    extrinsics_unscaled.npz
    poses3d_session_*.npz         (triangulated 3D joints per frame)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.pose.detector import detect_session, PoseSequence
from src.capture.time_sync import load_timestamps, align_pose_sequences
from src.calibration.skeleton_calib import (
    calibrate_from_three_view_poses, CameraParams,
)


def run_all_frames_triangulation(
    cams: list[CameraParams],
    kpts: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    conf_thresh: float,
) -> dict:
    """
    Triangulate every joint in every frame that has >=2 confident views.
    Returns a dict with keys:
        kpts3d: (N_frames, 17, 3)  float32, NaN where untriangulable
        triangulated_per_frame: (N_frames,) int
    """
    import cv2
    n_frames = kpts[cams[0].cam_id].shape[0]
    P = {c.cam_id: c.P for c in cams}
    out = np.full((n_frames, 17, 3), np.nan, dtype=np.float32)
    tri_counts = np.zeros(n_frames, dtype=np.int32)

    for fi in range(n_frames):
        for j in range(17):
            # collect confident views
            views = []
            for c in cams:
                if scores[c.cam_id][fi, j] >= conf_thresh:
                    views.append((c.cam_id, kpts[c.cam_id][fi, j]))
            if len(views) < 2:
                continue
            # Stack Ax=0 from N views, solve with SVD (linear DLT)
            A = []
            for cid, pt in views:
                Pm = P[cid]
                u, v = pt
                A.append(u * Pm[2] - Pm[0])
                A.append(v * Pm[2] - Pm[1])
            A = np.stack(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X[:3] / X[3]
            out[fi, j] = X
            tri_counts[fi] += 1
    return {"kpts3d": out, "triangulated_per_frame": tri_counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="session folder name (e.g. session_20260421_150000)")
    ap.add_argument("--conf", type=float, default=0.45,
                    help="joint confidence threshold")
    ap.add_argument("--min-joints", type=int, default=6,
                    help="minimum joints per frame visible in ALL 3 cams")
    ap.add_argument("--no-ba", action="store_true",
                    help="skip bundle adjustment")
    ap.add_argument("--no-focal", action="store_true",
                    help="do NOT optimize camera focal lengths in BA")
    ap.add_argument("--refine-iters", type=int, default=2,
                    help="iterations of BA with outlier rejection (default 2)")
    ap.add_argument("--outlier-px", type=float, default=8.0,
                    help="outlier rejection threshold between BA passes")
    ap.add_argument("--redetect", action="store_true",
                    help="re-run 2D detection even if cached")
    args = ap.parse_args()

    session_dir = config.CALIB_DIR / args.session
    if not session_dir.exists():
        alt = config.CALIB_DIR / f"session_{args.session}"
        if alt.exists():
            session_dir = alt
        else:
            raise FileNotFoundError(f"no such session: {session_dir}")

    print(f"\n[calib] Session: {session_dir}", flush=True)

    # 1. Detect or load cached 2D poses
    pose_seqs = detect_session(session_dir, overwrite=args.redetect)
    if len(pose_seqs) != 3:
        raise RuntimeError(
            f"expected 3 cameras, found {len(pose_seqs)}: {list(pose_seqs)}"
        )

    cam_ids = [c["id"] for c in config.CAMERAS]
    for cid in cam_ids:
        if cid not in pose_seqs:
            raise RuntimeError(f"missing pose sequence for {cid}")

    # 2. TIME-align all 3 cameras using their per-frame monotonic timestamps.
    #    USB webcams drift/drop frames independently, so frame-index alignment
    #    would pair SCENES TAKEN AT DIFFERENT MOMENTS.  See src/capture/time_sync.py.
    print(f"[calib] Time-aligning cameras via timestamps...", flush=True)
    timestamps = load_timestamps(session_dir, cam_ids)
    for cid in cam_ids:
        print(f"    {cid}: {len(pose_seqs[cid].keypoints):5d} frames, "
              f"{len(timestamps[cid]):5d} timestamps", flush=True)

    aligned = align_pose_sequences(pose_seqs, timestamps)
    kpts   = aligned["kpts"]
    scores = aligned["scores"]
    ref_cam = aligned["reference_cam"]
    frame_map = aligned["frame_map"]
    t_ref_ns = aligned["t_ref_ns"]
    image_sizes = {c: pose_seqs[c].image_size   for c in cam_ids}

    stats = aligned["stats"]
    n_aligned = stats["matched"]
    print(f"    reference cam: {ref_cam}  (shortest sequence)", flush=True)
    print(f"    matched {n_aligned}/{stats['n_ref']} ref frames  "
          f"(dropped: no_match={stats['dropped_no_match']}, "
          f"dt_too_large={stats['dropped_dt_too_large']})", flush=True)
    if n_aligned < 100:
        raise RuntimeError(
            f"only {n_aligned} time-aligned frames. Likely cameras are not "
            f"sharing a timebase. Re-record session.")
    n_min = n_aligned

    # 3. Calibrate
    print(f"\n[calib] Running skeleton-based calibration "
          f"(conf_thresh={args.conf})...", flush=True)
    cams, pts_3d_init, info = calibrate_from_three_view_poses(
        cam_ids, image_sizes, kpts, scores,
        conf_thresh=args.conf,
        min_joints_per_frame=args.min_joints,
        do_bundle_adjust=not args.no_ba,
        optimize_focal=not args.no_focal,
        n_refine_iters=args.refine_iters,
        outlier_px=args.outlier_px,
    )

    if info:
        print(f"\n[calib] Bundle-adjust RMS reprojection error: "
              f"{info.get('rms_pixel', float('nan')):.3f} px", flush=True)

    # 4. Save intrinsics + extrinsics (UNSCALED)
    for c in cams:
        out_path = config.CALIB_DIR / f"intrinsics_{c.cam_id}.npz"
        cam_cfg = next(x for x in config.CAMERAS if x["id"] == c.cam_id)
        np.savez(
            out_path,
            K=c.K, dist=c.dist,
            image_size=np.array(c.image_size),
            cam_id=c.cam_id, cam_index=cam_cfg["index"],
            source="skeleton_calibration",
        )
        print(f"  saved: {out_path}", flush=True)

    extr_path = config.CALIB_DIR / "extrinsics_unscaled.npz"
    np.savez(
        extr_path,
        cams=np.array([c.cam_id for c in cams]),
        R0=cams[0].R, t0=cams[0].t,
        R1=cams[1].R, t1=cams[1].t,
        R2=cams[2].R, t2=cams[2].t,
        K0=cams[0].K, K1=cams[1].K, K2=cams[2].K,
        source="skeleton_calibration",
        ba_rms_px=info.get("rms_pixel", -1.0) if info else -1.0,
        conf_thresh=args.conf,
        n_frames=n_min,
    )
    print(f"  saved: {extr_path}", flush=True)

    # 5. Triangulate all TIME-ALIGNED frames (save as intermediate)
    print("\n[calib] Triangulating full sequence for each aligned frame...",
          flush=True)
    tri = run_all_frames_triangulation(cams, kpts, scores, args.conf)
    poses_path = config.CALIB_DIR / f"poses3d_{session_dir.name}.npz"
    np.savez(
        poses_path,
        kpts3d=tri["kpts3d"],
        triangulated_per_frame=tri["triangulated_per_frame"],
        fps=pose_seqs[cam_ids[0]].fps,
        session=session_dir.name,
        joint_names=np.array(config.KPT_NAMES),
        # Time alignment artifacts -- align_world_from_force.py uses these.
        t_ref_ns=t_ref_ns,
        reference_cam=ref_cam,
        frame_map_ref=frame_map[ref_cam],
    )
    print(f"  saved: {poses_path}", flush=True)
    usable = int(np.sum(tri["triangulated_per_frame"] >= 10))
    print(f"  frames with >=10 joints triangulated: {usable}/{n_min}",
          flush=True)

    print(
        "\n[calib] Done. NOTE: extrinsics are still UNSCALED "
        "(arbitrary global scale, cam0-cam1 baseline = 1).\n"
        "Next step:\n"
        f"  python scripts/align_world_from_force.py --session "
        f"{session_dir.name}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
