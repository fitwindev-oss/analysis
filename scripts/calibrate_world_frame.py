"""
World frame alignment - rotates/translates the extrinsics so the world origin
is at the force plate's left-bottom corner, +X points to the right (Board1
toward Board2), +Y points forward (away from user), +Z points up.

Prerequisite:
    data/calibration/extrinsics.npz  (from calibrate_extrinsics.py)

Workflow:
    1) PLACE the ChArUco board FLAT on the force plates so that:
         - board's LEFT-BOTTOM corner == plate's LEFT-BOTTOM (0, 0)
         - board's long axis pointing along plate's +X (rightward)
         - board face UP, text readable when looking from front
       Use a ruler/tape to snap the corner. Board only covers a fraction of
       the plate - that's fine, we only need it to define axes.

    2) Run this script. It reads live frames from all 3 cameras, detects the
       board in each, solves PnP using the calibrated intrinsics, and finds
       the rigid transform that re-aligns the extrinsics so:
         - origin = anchor-world position of the board's (0,0) corner
         - axes  = board's axes (+X right along board, +Y forward along board,
                   +Z up, normal to the board plane)
       Then applies an ADDITIONAL offset so world origin = plate left-bottom,
       which equals the board's (0,0) corner since you placed them coincident.

    3) Press 's' to save. Output: data/calibration/world_frame.npz.

Controls:
    c  = (optional) capture/freeze current best detection (not required; the
         script keeps using latest until 's' pressed)
    s  = save and exit
    q  = quit without saving
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


def open_cameras():
    caps = []
    for cam in config.CAMERAS:
        cap = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.CAMERA_AUTO_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE,      config.CAMERA_EXPOSURE)
        caps.append((cam, cap))
    return caps


def load_extrinsics():
    path = config.CALIB_DIR / "extrinsics.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing: {path}\nRun: python scripts/calibrate_extrinsics.py first"
        )
    z = np.load(path, allow_pickle=True)
    cams = list(map(str, z["cams"].tolist()))
    intr = {
        cams[0]: {"K": z["K0"], "dist": z["dist0"]},
        cams[1]: {"K": z["K1"], "dist": z["dist1"]},
        cams[2]: {"K": z["K2"], "dist": z["dist2"]},
    }
    extr = {
        cams[0]: {"R": z["R0"], "t": z["t0"]},
        cams[1]: {"R": z["R1"], "t": z["t1"]},
        cams[2]: {"R": z["R2"], "t": z["t2"]},
    }
    return cams, intr, extr


def solve_pnp_for_board(K, dist, obj_pts, img_pts):
    if obj_pts is None or len(obj_pts) < 6:
        return None
    op = np.asarray(obj_pts).reshape(-1, 3).astype(np.float32)
    ip = np.asarray(img_pts).reshape(-1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def board_pose_in_temp_world(R_cb, t_cb, R_cw, t_cw):
    """
    Given:
      board pose in cam:          X_cam = R_cb * X_board + t_cb
      cam pose in temp-world:     X_cam = R_cw * X_world + t_cw
    Return board pose in temp-world:
      X_world = R_wb * X_board + t_wb
    """
    # From X_cam = R_cb X_board + t_cb and X_cam = R_cw X_world + t_cw:
    #   X_world = R_cw^T (R_cb X_board + t_cb - t_cw)
    #           = (R_cw^T R_cb) X_board + R_cw^T (t_cb - t_cw)
    R_wb = R_cw.T @ R_cb
    t_wb = R_cw.T @ (t_cb - t_cw)
    return R_wb, t_wb


def combine_grid(frames, labels):
    h = max(f.shape[0] for f in frames)
    out = []
    for f, lab in zip(frames, labels):
        r = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
        cv2.putText(r, lab, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        out.append(r)
    return np.hstack(out)


def main():
    args = argparse.ArgumentParser().parse_args()

    print("[world] Loading extrinsics + intrinsics...")
    cam_ids, intr, extr = load_extrinsics()
    print(f"  cams: {cam_ids}")

    print("[world] Opening cameras...")
    caps = open_cameras()
    detector, board = build_detector()

    # Cache latest board pose in temp-world, per camera
    latest_board_world: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    win = "World Frame Alignment"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(
        "\nPlace the ChArUco board FLAT on the force plate such that its\n"
        "(0,0) corner coincides with plate left-bottom (0,0) and its +X axis\n"
        "points toward Board2 (rightward). Press 's' when ready.\n"
    )

    while True:
        frames, dets = [], []
        for cam, cap in caps:
            ok, f = cap.read()
            if not ok:
                f = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), np.uint8)
            frames.append(f)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            dets.append(detect(detector, g))

        # Update board-in-world pose for each camera that sees it.
        for (cam, _), d in zip(caps, dets):
            cid = cam["id"]
            if not d.ok(min_corners=8):
                continue
            obj_pts, img_pts = image_object_correspondences(
                board, d.ch_corners, d.ch_ids
            )
            pose = solve_pnp_for_board(
                intr[cid]["K"], intr[cid]["dist"], obj_pts, img_pts,
            )
            if pose is None:
                continue
            R_cb, t_cb = pose
            R_wb, t_wb = board_pose_in_temp_world(
                R_cb, t_cb, extr[cid]["R"], extr[cid]["t"],
            )
            latest_board_world[cid] = (R_wb, t_wb)

        # Visualization
        vis_frames = []
        for f, d, cam in zip(frames, dets, [c for c, _ in caps]):
            o = draw_overlay(f, d)
            n = 0 if d.ch_ids is None else len(d.ch_ids)
            col = (0, 255, 0) if n >= 8 else (0, 0, 255)
            cv2.putText(o, f"{cam['id']} corners:{n}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            vis_frames.append(o)

        grid = combine_grid(vis_frames, [c["id"] for c, _ in caps])
        status = f"board visible in {len(latest_board_world)}/3 cams"
        cv2.putText(grid, status, (10, grid.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(grid, "Position board at plate (0,0), axes aligned. "
                          "s=save  q=quit",
                    (10, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, grid)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[world] quit without saving.")
            break
        elif key == ord('s'):
            if not latest_board_world:
                print("  [error] board not detected in any camera")
                continue

            # Average the board pose across cameras (quaternion avg for R)
            Rs = np.stack([R for R, t in latest_board_world.values()])
            ts = np.stack([t for R, t in latest_board_world.values()])
            # Simple average; for <=3 cams with close views this is adequate
            R_avg = np.mean(Rs, axis=0)
            # Orthonormalize via SVD
            U, _, Vt = np.linalg.svd(R_avg)
            R_avg = U @ Vt
            if np.linalg.det(R_avg) < 0:
                R_avg[:, -1] *= -1
            t_avg = np.mean(ts, axis=0)

            # Spread check - if cameras disagree strongly, warn
            origins_disagree_mm = float(np.linalg.norm(ts - t_avg, axis=1).max() * 1000)
            print(f"  board (0,0) in temp-world: t = {t_avg*1000} mm")
            print(f"  inter-camera origin disagreement: {origins_disagree_mm:.1f} mm")

            # Transform extrinsics so that new_world coincides with board pose.
            # Currently (old) cam pose in TEMP world: X_cam = R_cw X_temp + t_cw
            # Board pose in TEMP world: X_temp = R_avg X_board + t_avg
            # We declare NEW WORLD == BOARD frame (in meters) first, then apply
            # an offset (0, 0, 0) because we placed board's (0,0) at plate (0,0).
            #
            # So X_temp = R_avg X_newworld + t_avg
            #    X_cam  = R_cw (R_avg X_newworld + t_avg) + t_cw
            #           = (R_cw R_avg) X_newworld + (R_cw t_avg + t_cw)
            R_avg = R_avg.astype(np.float64)
            t_avg = t_avg.astype(np.float64)
            new_extr = {}
            for cid in cam_ids:
                R_cw = extr[cid]["R"].astype(np.float64)
                t_cw = extr[cid]["t"].astype(np.float64)
                R_new = R_cw @ R_avg
                t_new = R_cw @ t_avg + t_cw
                new_extr[cid] = {"R": R_new, "t": t_new}

            # Plate sanity: origin in world is (0,0,0). Second corner (plate
            # width along X) is at (0.558, 0, 0) m if board +X is along plate +X.
            # (This is only a sanity print; actual plate covers Z=0 surface.)

            # NOTE about board orientation: user placed board with +X toward
            # Board2 and face up. In the board's native coordinate system
            # (OpenCV), X is along the short edge (6 squares), Y is along the
            # long edge (8 squares), Z is normal to the face (pointing up when
            # face is up). This matches our world frame convention (X right,
            # Y forward, Z up) IF the board is oriented with its short side
            # (6 squares, 180 mm) along plate X and long side (8 squares,
            # 240 mm) along plate Y.

            out_path = config.CALIB_DIR / "world_frame.npz"
            np.savez(
                out_path,
                cams=np.array(cam_ids),
                R0=new_extr[cam_ids[0]]["R"], t0=new_extr[cam_ids[0]]["t"],
                R1=new_extr[cam_ids[1]]["R"], t1=new_extr[cam_ids[1]]["t"],
                R2=new_extr[cam_ids[2]]["R"], t2=new_extr[cam_ids[2]]["t"],
                K0=intr[cam_ids[0]]["K"], dist0=intr[cam_ids[0]]["dist"],
                K1=intr[cam_ids[1]]["K"], dist1=intr[cam_ids[1]]["dist"],
                K2=intr[cam_ids[2]]["K"], dist2=intr[cam_ids[2]]["dist"],
                board_in_temp_world_R=R_avg,
                board_in_temp_world_t=t_avg,
                origin_disagreement_mm=origins_disagree_mm,
            )
            print(f"\n[world] saved: {out_path}")
            if origins_disagree_mm > 10.0:
                print(
                    f"  [warn] camera origins disagree by {origins_disagree_mm:.1f} "
                    f"mm - consider redoing extrinsic calibration with more "
                    f"captures."
                )
            print(
                "\nWorld frame is now: origin at force plate (0, 0, 0) mm,\n"
                "  +X toward Board2 (right), +Y forward, +Z up.\n"
                "Verify with: python scripts/verify_calibration.py (to be added)."
            )
            break

    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
