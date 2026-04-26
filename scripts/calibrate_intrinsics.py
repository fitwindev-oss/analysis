"""
Per-camera intrinsic calibration using a ChArUco board.

Usage:
    python scripts/calibrate_intrinsics.py --cam 0
    python scripts/calibrate_intrinsics.py --cam 1
    python scripts/calibrate_intrinsics.py --cam 2

Controls (in the live window):
    c  = capture current frame (only if >=10 corners detected)
    u  = undo last capture
    s  = solve & save (need >=INTRINSIC_MIN_IMAGES captures)
    q  = quit without saving

Output:
    data/calibration/intrinsics_C{cam_id}.npz
        containing: K (3x3), dist (5,), rms (float), image_size (w, h),
                    squares_x, squares_y, square_len_mm, marker_len_mm

Capture strategy:
    20-40 captures, varying:
      - position (4 corners of FOV + center)
      - distance (near ~0.5 m, mid ~1.5 m, far ~3 m)
      - tilt (0, +/-15, +/-30, +/-45 degrees)
    The board should NOT be moving during capture (avoid motion blur).
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
from src.calibration.charuco import (
    build_detector, detect, draw_overlay, image_object_correspondences,
)


def open_camera(cam_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.CAMERA_AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE,      config.CAMERA_EXPOSURE)
    return cap


def solve_calibration(all_obj_pts, all_img_pts, image_size):
    """Run cv2.calibrateCamera with the collected correspondences."""
    # cv2.calibrateCamera expects (N, 3) obj_pts and (N, 2) img_pts per view
    obj_pts_list = [op.reshape(-1, 3).astype(np.float32) for op in all_obj_pts]
    img_pts_list = [ip.reshape(-1, 2).astype(np.float32) for ip in all_img_pts]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None,
        flags=0,
    )
    return rms, K, dist, rvecs, tvecs


def per_view_reprojection_errors(obj_pts_list, img_pts_list, K, dist, rvecs, tvecs):
    errors = []
    for op, ip, rv, tv in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        op = np.asarray(op).reshape(-1, 3).astype(np.float32)
        ip = np.asarray(ip).reshape(-1, 2).astype(np.float32)
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - ip, axis=1).mean()
        errors.append(float(err))
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, required=True,
                   help="camera index (0, 1, or 2)")
    p.add_argument("--min", type=int, default=config.INTRINSIC_MIN_IMAGES)
    args = p.parse_args()

    cam_cfg = next((c for c in config.CAMERAS if c["index"] == args.cam), None)
    cam_id  = cam_cfg["id"] if cam_cfg else f"idx{args.cam}"
    cam_label = cam_cfg["label"] if cam_cfg else ""

    print(f"[calib] Intrinsics for camera {args.cam} ({cam_id} / {cam_label})")
    print(f"        Target: {args.min}+ captures.  Press 'c' to capture, 's' to solve.\n")

    cap = open_camera(args.cam)
    detector, board = build_detector()

    captures: list[dict] = []   # {"obj_pts", "img_pts", "preview"}
    image_size = None

    win = f"Intrinsics - {cam_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_flash_until = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[calib] frame read failed, retrying...", flush=True)
            # Pump the GUI event loop even if the camera stalled
            cv2.waitKey(30)
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect(detector, gray)

        vis = draw_overlay(frame, det)
        n = 0 if det.ch_ids is None else len(det.ch_ids)
        color = (0, 255, 0) if n >= 10 else (0, 0, 255)
        cv2.putText(vis, f"corners: {n}   captures: {len(captures)}/{args.min}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, "c=capture  u=undo  s=solve  q=quit",
                    (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        # White flash border right after a capture, as visual confirmation
        if time.monotonic() < last_flash_until:
            cv2.rectangle(vis, (0, 0), (vis.shape[1]-1, vis.shape[0]-1),
                          (255, 255, 255), 6)
        cv2.imshow(win, vis)
        # 20ms wait gives Windows enough time to pump the GUI event loop,
        # which prevents the "Not Responding" stall seen with waitKey(1).
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[calib] quit without saving", flush=True)
            break
        elif key == ord('c'):
            if not det.ok(min_corners=10):
                print("  [skip] need at least 10 charuco corners", flush=True)
                continue
            obj_pts, img_pts = image_object_correspondences(
                board, det.ch_corners, det.ch_ids
            )
            if obj_pts is None or img_pts is None or len(obj_pts) < 10:
                print("  [skip] matchImagePoints failed", flush=True)
                continue
            captures.append({"obj_pts": obj_pts, "img_pts": img_pts})
            last_flash_until = time.monotonic() + 0.15
            print(f"  captured #{len(captures)}  ({len(obj_pts)} corners)",
                  flush=True)
        elif key == ord('u'):
            if captures:
                captures.pop()
                print(f"  undo, now {len(captures)} captures", flush=True)
        elif key == ord('s'):
            if len(captures) < args.min:
                print(f"  [need {args.min - len(captures)} more captures]",
                      flush=True)
                continue
            print(f"\n[calib] Solving with {len(captures)} views...", flush=True)
            try:
                rms, K, dist, rvecs, tvecs = solve_calibration(
                    [c["obj_pts"] for c in captures],
                    [c["img_pts"] for c in captures],
                    image_size,
                )
            except Exception as e:
                print(f"  [error] calibrateCamera failed: {e}")
                continue
            per_view = per_view_reprojection_errors(
                [c["obj_pts"] for c in captures],
                [c["img_pts"] for c in captures],
                K, dist, rvecs, tvecs,
            )
            print(f"  RMS reprojection error: {rms:.4f} px")
            print(f"  per-view mean error: min={min(per_view):.3f}, "
                  f"max={max(per_view):.3f}, mean={np.mean(per_view):.3f}")
            print(f"  K =\n{K}")
            print(f"  dist = {dist.ravel()}")

            if rms > config.INTRINSIC_TARGET_RMS_PX:
                print(f"  [warn] RMS {rms:.3f} exceeds target "
                      f"{config.INTRINSIC_TARGET_RMS_PX}. Consider adding more "
                      f"varied captures or check board flatness.")

            out_path = config.CALIB_DIR / f"intrinsics_{cam_id}.npz"
            np.savez(
                out_path,
                K=K, dist=dist, rms=rms,
                image_size=np.array(image_size),
                squares_x=config.CHARUCO_SQUARES_X,
                squares_y=config.CHARUCO_SQUARES_Y,
                square_len_mm=config.CHARUCO_SQUARE_LEN_MM,
                marker_len_mm=config.CHARUCO_MARKER_LEN_MM,
                cam_id=cam_id,
                cam_index=args.cam,
                n_views=len(captures),
            )
            print(f"\n[calib] saved: {out_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
