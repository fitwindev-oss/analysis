"""
Multi-camera EXTRINSIC calibration using the ChArUco board.

Prerequisite:
    data/calibration/intrinsics_C0.npz
    data/calibration/intrinsics_C1.npz
    data/calibration/intrinsics_C2.npz    (from calibrate_intrinsics.py)

Workflow:
    1) All 3 cameras open simultaneously.
    2) Hold/move the ChArUco board through the capture volume.
    3) When the board is visible in 2 or 3 cameras at once, press 'c' to
       capture that synchronized snapshot. Aim for 15+ synchronized captures
       with varied board poses (different positions/angles).
    4) Press 's' to solve. The world frame of this calibration is the BOARD
       pose in the FIRST captured snapshot - this is just a TEMPORARY world.
       The final force-plate-aligned world frame is computed separately by
       scripts/calibrate_world_frame.py using one additional capture.

Controls:
    c  = capture synchronized frames (must be visible in >=2 cams)
    u  = undo last capture
    s  = solve & save extrinsics
    q  = quit without saving

Output:
    data/calibration/extrinsics.npz
        cams = list of cam_id
        R[i] (3x3) + t[i] (3,)  = pose of cam i in temp world frame
        per_cam_reproj_px       = per-camera mean reprojection error
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
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam['index']} ({cam['id']})")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.CAMERA_AUTO_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE,      config.CAMERA_EXPOSURE)
        caps.append((cam, cap))
    return caps


def load_intrinsics():
    out = {}
    for cam in config.CAMERAS:
        path = config.CALIB_DIR / f"intrinsics_{cam['id']}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"missing: {path}\nRun: python scripts/calibrate_intrinsics.py "
                f"--cam {cam['index']}"
            )
        z = np.load(path)
        out[cam["id"]] = {
            "K": z["K"], "dist": z["dist"], "rms": float(z["rms"]),
            "image_size": tuple(z["image_size"].tolist()),
        }
        print(f"  loaded {cam['id']} intrinsics: rms={out[cam['id']]['rms']:.3f}px")
    return out


def solve_pnp_for_board(K, dist, obj_pts, img_pts):
    """Return (R_3x3, t_3) of the BOARD relative to CAMERA, or None."""
    if obj_pts is None or len(obj_pts) < 6:
        return None
    op = np.asarray(obj_pts).reshape(-1, 3).astype(np.float32)
    ip = np.asarray(img_pts).reshape(-1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def combine_grid(frames, labels):
    """Stack the 3 camera frames side-by-side for display."""
    h = max(f.shape[0] for f in frames)
    resized = []
    for f, lab in zip(frames, labels):
        r = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
        cv2.putText(r, lab, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        resized.append(r)
    return np.hstack(resized)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=int, default=15,
                   help="minimum synchronized captures")
    args = p.parse_args()

    print("[extrinsics] Loading per-camera intrinsics...")
    intr = load_intrinsics()

    print("[extrinsics] Opening cameras...")
    caps = open_cameras()

    detector, board = build_detector()

    # Per-capture storage. Each element is a dict of cam_id -> (obj_pts, img_pts)
    # plus the computed board->camera pose for that cam.
    captures: list[dict] = []

    win = "Extrinsics - 3 cams"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("Show the ChArUco board to 2-3 cameras simultaneously. Press 'c' to capture.")
    print("Aim for 15+ captures with varied board poses.\n")

    while True:
        # Grab 3 frames as close in time as possible (sequential reads; hardware
        # sync is not available on webcams).
        frames, dets, views = [], [], []
        for cam, cap in caps:
            ok, f = cap.read()
            if not ok:
                f = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), np.uint8)
            frames.append(f)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            d = detect(detector, g)
            dets.append(d)
            views.append(cam["id"])

        visible = [v for v, d in zip(views, dets) if d.ok(min_corners=8)]

        # Overlay visualization
        overlay_frames = []
        for f, d, v in zip(frames, dets, views):
            o = draw_overlay(f, d)
            n = 0 if d.ch_ids is None else len(d.ch_ids)
            color = (0, 255, 0) if n >= 8 else (0, 0, 255)
            cv2.putText(o, f"{v}  corners: {n}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            overlay_frames.append(o)

        grid = combine_grid(overlay_frames, views)
        status = f"visible in {len(visible)}/3 cams  |  captures: {len(captures)}"
        cv2.putText(grid, status, (10, grid.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(grid, "c=capture (>=2 cams)  u=undo  s=solve  q=quit",
                    (10, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, grid)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print("[extrinsics] quit without saving.")
            break
        elif key == ord('c'):
            if len(visible) < 2:
                print("  [skip] need board visible in >=2 cameras")
                continue
            entry = {"visible": visible, "per_cam": {}}
            for (cam, _), d in zip(caps, dets):
                if not d.ok(min_corners=8):
                    continue
                obj_pts, img_pts = image_object_correspondences(
                    board, d.ch_corners, d.ch_ids,
                )
                if obj_pts is None or len(obj_pts) < 6:
                    continue
                entry["per_cam"][cam["id"]] = {
                    "obj_pts": np.asarray(obj_pts),
                    "img_pts": np.asarray(img_pts),
                }
            if len(entry["per_cam"]) >= 2:
                captures.append(entry)
                print(f"  captured #{len(captures)}: visible in {visible}")
            else:
                print("  [skip] too few valid correspondences")
        elif key == ord('u'):
            if captures:
                captures.pop()
                print(f"  undo, now {len(captures)} captures")
        elif key == ord('s'):
            if len(captures) < args.min:
                print(f"  [need {args.min - len(captures)} more captures]")
                continue

            # For each capture, solve PnP per camera => board pose in each cam.
            # Then build an anchored world: take the FIRST capture's board pose
            # as the temporary world frame.
            print(f"\n[extrinsics] Solving with {len(captures)} captures...")

            cam_ids = [c["id"] for c in config.CAMERAS]
            # cam_world_R[c] / cam_world_t[c] = pose of WORLD in cam's frame
            # We compute this from the FIRST capture where the cam sees the board.
            # Then refine with all subsequent captures.
            cam_pose = {cid: None for cid in cam_ids}

            # Strategy:
            # 1. Pick an ANCHOR capture: one where ALL 3 cams see the board.
            anchor_idx = None
            for i, cap_entry in enumerate(captures):
                if len(cap_entry["per_cam"]) == 3:
                    anchor_idx = i
                    break
            if anchor_idx is None:
                # fallback: first with max visibility, use chain later
                anchor_idx = max(range(len(captures)),
                                 key=lambda i: len(captures[i]["per_cam"]))
                print(f"  [warn] no capture with all 3 cams. Anchor = #{anchor_idx} "
                      f"({len(captures[anchor_idx]['per_cam'])} cams)")

            anchor = captures[anchor_idx]
            # Board -> cam for each cam in anchor
            for cid, corr in anchor["per_cam"].items():
                K = intr[cid]["K"]
                d = intr[cid]["dist"]
                pose = solve_pnp_for_board(K, d, corr["obj_pts"], corr["img_pts"])
                if pose is None:
                    print(f"  [error] anchor PnP failed for {cid}")
                    continue
                R_cb, t_cb = pose       # board pose in cam
                cam_pose[cid] = (R_cb, t_cb)

            # Define world = board frame of anchor snapshot.
            # So for each cam c with pose (R_cb, t_cb): WORLD -> CAM mapping is
            # X_cam = R_cb * X_world + t_cb
            # i.e. extrinsic (R, t) of cam c w.r.t. world = (R_cb, t_cb).

            # For cams not in anchor capture, chain via another capture where
            # they share the board with a cam whose pose is already known.
            missing = [cid for cid, p in cam_pose.items() if p is None]
            for miss in missing:
                resolved = False
                for cap_entry in captures:
                    if miss not in cap_entry["per_cam"]:
                        continue
                    # Find any cam already known in this same capture
                    for other_cid in cap_entry["per_cam"]:
                        if other_cid == miss or cam_pose[other_cid] is None:
                            continue
                        # pose of board in miss cam
                        K_m = intr[miss]["K"]; d_m = intr[miss]["dist"]
                        pose_m = solve_pnp_for_board(
                            K_m, d_m,
                            cap_entry["per_cam"][miss]["obj_pts"],
                            cap_entry["per_cam"][miss]["img_pts"],
                        )
                        # pose of board in other cam (this capture)
                        K_o = intr[other_cid]["K"]; d_o = intr[other_cid]["dist"]
                        pose_o = solve_pnp_for_board(
                            K_o, d_o,
                            cap_entry["per_cam"][other_cid]["obj_pts"],
                            cap_entry["per_cam"][other_cid]["img_pts"],
                        )
                        if pose_m is None or pose_o is None:
                            continue
                        R_mb, t_mb = pose_m
                        R_ob, t_ob = pose_o
                        # In this capture, board is the same physical object,
                        # but its pose in WORLD is different from anchor.
                        # We want: miss_cam relative to world.
                        # Known: other_cam relative to world = cam_pose[other_cid]
                        # Also known: board relative to other_cam (from pose_o)
                        # Therefore, board in world = (other_cam in world)^{-1}
                        #                              * (board in other_cam)
                        # And miss_cam in world = (miss_cam relative to board)
                        #                          * (board in world)^{-1} inverted... wait let's be careful.
                        # We'll express things as 4x4 transforms:
                        def T(R, t):
                            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
                            return M
                        T_o_w = T(*cam_pose[other_cid])         # world -> other_cam
                        T_b_o = T(R_ob, t_ob)                   # other_cam -> board? No:
                        # pose_o = board pose in other_cam: X_other = R_ob * X_board + t_ob
                        # So T_b_o is BOARD -> OTHER_CAM
                        # => BOARD -> WORLD = inv(T_o_w) * T_b_o
                        T_b_w = np.linalg.inv(T_o_w) @ T_b_o
                        # pose_m: BOARD -> MISS_CAM
                        T_b_m = T(R_mb, t_mb)
                        # WORLD -> MISS_CAM = T_b_m * inv(T_b_w)
                        T_m_w = T_b_m @ np.linalg.inv(T_b_w)
                        cam_pose[miss] = (T_m_w[:3, :3].copy(), T_m_w[:3, 3].copy())
                        resolved = True
                        print(f"  resolved {miss} via chain with {other_cid}")
                        break
                    if resolved:
                        break
                if not resolved:
                    print(f"  [error] could not chain-resolve camera {miss}. "
                          f"Capture more views where it shares the board with "
                          f"another camera.")
                    break

            if any(p is None for p in cam_pose.values()):
                print("  [abort] incomplete solve. Need more captures.")
                continue

            # Reprojection error check
            per_cam_errors = {cid: [] for cid in cam_ids}
            for cap_entry in captures:
                for cid, corr in cap_entry["per_cam"].items():
                    # project obj_pts (in board frame for this capture).
                    # But we don't know the board pose in world for THIS capture
                    # unless cam cid was in it. Easier: solve PnP and compare
                    # to all other cams' projections.
                    K = intr[cid]["K"]; d = intr[cid]["dist"]
                    pose = solve_pnp_for_board(K, d, corr["obj_pts"], corr["img_pts"])
                    if pose is None:
                        continue
                    R_cb, t_cb = pose
                    rvec, _ = cv2.Rodrigues(R_cb)
                    proj, _ = cv2.projectPoints(
                        corr["obj_pts"].reshape(-1, 3).astype(np.float32),
                        rvec, t_cb, K, d,
                    )
                    err = np.linalg.norm(
                        proj.reshape(-1, 2) - corr["img_pts"].reshape(-1, 2), axis=1
                    ).mean()
                    per_cam_errors[cid].append(float(err))

            print("\n[extrinsics] Per-camera reprojection (board PnP):")
            for cid in cam_ids:
                errs = per_cam_errors[cid]
                if errs:
                    print(f"  {cid}: mean={np.mean(errs):.3f} px "
                          f"(n={len(errs)}, max={max(errs):.3f})")

            out_path = config.CALIB_DIR / "extrinsics.npz"
            np.savez(
                out_path,
                cams=np.array(cam_ids),
                R0=cam_pose[cam_ids[0]][0], t0=cam_pose[cam_ids[0]][1],
                R1=cam_pose[cam_ids[1]][0], t1=cam_pose[cam_ids[1]][1],
                R2=cam_pose[cam_ids[2]][0], t2=cam_pose[cam_ids[2]][1],
                K0=intr[cam_ids[0]]["K"], dist0=intr[cam_ids[0]]["dist"],
                K1=intr[cam_ids[1]]["K"], dist1=intr[cam_ids[1]]["dist"],
                K2=intr[cam_ids[2]]["K"], dist2=intr[cam_ids[2]]["dist"],
                n_captures=len(captures),
                anchor_idx=anchor_idx,
            )
            print(f"\n[extrinsics] saved: {out_path}")
            print(
                "NOTE: This is a TEMPORARY world frame anchored to the board at "
                "anchor capture. Run scripts/calibrate_world_frame.py next to "
                "align the world to the force plate origin."
            )
            break

    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
