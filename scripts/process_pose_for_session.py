"""
Run MediaPipe BlazePose on every camera video in a session folder.

Saves one `poses_<cam_id>.npz` per camera in the session, containing:
    kpts_mp33, visibility_mp33, world_mp33, angles, angle_names,
    fps, image_size, cam_id, backend, model_complexity

NO 3D triangulation or cross-camera fusion. Each camera is treated
independently; joint angles are computed in each camera's image plane.

Usage:
    python scripts/process_pose_for_session.py --session balance_eo_20260422_010718
    python scripts/process_pose_for_session.py --session squat_20260422_012329 --complexity 2 --overwrite
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.pose.mediapipe_backend import MPPoseDetector
from src.analysis.pose2d import compute_angles_timeseries, ANGLE_NAMES


def _resolve_session(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for base in (config.SESSIONS_DIR, config.CALIB_DIR):
        c = base / name
        if c.exists():
            return c
    raise FileNotFoundError(f"session not found: {name}")


def _process_one_video(detector: MPPoseDetector,
                       video_path: Path, out_path: Path,
                       cam_id: str, complexity: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or config.CAMERA_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt_ms = max(1, int(round(1000.0 / float(fps))))

    kpts_all, vis_all, world_all = [], [], []
    i = 0
    ts_ms = 0
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, desc=f"[pose] {cam_id}", ncols=80)
    except Exception:
        bar = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            r = detector.detect(bgr, timestamp_ms=ts_ms)
            kpts_all.append(r.kpts33)
            vis_all.append(r.vis33)
            world_all.append(r.world33)
            i += 1
            ts_ms += dt_ms
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        cap.release()

    if not kpts_all:
        raise RuntimeError("no frames produced")

    kpts  = np.stack(kpts_all,  axis=0).astype(np.float32)
    vis   = np.stack(vis_all,   axis=0).astype(np.float32)
    world = np.stack(world_all, axis=0).astype(np.float32)
    angles = compute_angles_timeseries(kpts, vis, conf_thresh=0.3)

    np.savez(
        out_path,
        cam_id=cam_id,
        kpts_mp33=kpts,
        visibility_mp33=vis,
        world_mp33=world,
        angles=angles,
        angle_names=np.array(ANGLE_NAMES),
        fps=float(fps),
        image_size=np.array([w, h], dtype=np.int32),
        backend="mediapipe",
        model_complexity=int(complexity),
    )
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--complexity", type=int, default=None,
                    choices=[0, 1, 2],
                    help="MediaPipe model complexity "
                         "(0=Lite, 1=Full, 2=Heavy). "
                         "Defaults to config.POSE_POSTRECORD_COMPLEXITY.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    session_dir = _resolve_session(args.session)
    complexity = (config.POSE_POSTRECORD_COMPLEXITY
                  if args.complexity is None else args.complexity)
    print(f"[pose] session: {session_dir}")
    print(f"[pose] complexity: {complexity}")

    def _dl_log(m: str) -> None:
        print(f"[pose] {m}", flush=True)
    processed: list[tuple[str, int]] = []
    detector = None
    try:
        for cam in config.CAMERAS:
            cid = cam["id"]
            video_path = session_dir / f"{cid}.mp4"
            out_path   = session_dir / f"poses_{cid}.npz"

            if not video_path.exists():
                print(f"  [{cid}] skip (no video)")
                continue
            if out_path.exists() and not args.overwrite:
                print(f"  [{cid}] cached")
                processed.append((cid, -1))
                continue

            # One detector per camera (see PoseWorker comment re:
            # monotonic-timestamp requirement of VIDEO running mode).
            if detector is not None:
                detector.close(); detector = None
            detector = MPPoseDetector(
                complexity=complexity, running_mode="video",
                lr_swap=bool(getattr(config, "CAMERA_MIRROR", False)),
                progress_cb=_dl_log)

            t0 = time.time()
            try:
                n = _process_one_video(
                    detector, video_path, out_path, cid, complexity)
                print(f"  [{cid}] saved {out_path.name}  "
                      f"({n} frames, {time.time()-t0:.1f}s)")
                processed.append((cid, n))
            except Exception as e:
                print(f"  [{cid}] ERROR: {e}")
    finally:
        if detector is not None:
            detector.close()

    if processed:
        print(f"\n[pose] done. Processed cameras: "
              f"{', '.join(c for c,_ in processed)}")
    else:
        print("\n[pose] nothing to do (no videos found).")


if __name__ == "__main__":
    main()
