"""
Multi-camera INTRINSIC calibration with AUTO capture.

All 3 cameras open simultaneously. The script automatically captures when:
  (a) the board is DETECTED with >=10 corners in a given camera,
  (b) the board is STILL (low motion over last few frames),
  (c) the current view is sufficiently DIFFERENT from every previous capture
      in that camera's pool (novelty in pixel space),
  (d) enough time has passed since the last capture for that camera.

Each camera maintains its OWN capture pool - one frame may feed 1, 2, or 3
camera pools depending on which cams see the board well enough.

Just press SPACE once to start, then move the board slowly through each
camera's FOV. Stop in each position for about 1 second to let the auto
capture fire.

Usage:
    python scripts/calibrate_intrinsics_multi.py
    python scripts/calibrate_intrinsics_multi.py --min 25

Controls:
    SPACE   = toggle auto-capture (starts OFF; press once to begin)
    c       = manual capture override (any cam with >=10 corners)
    u       = undo last frame (rolls back every contributing cam)
    r       = reset ALL pools (start over)
    s       = solve & save when every pool has reached --min
    q       = quit without saving
"""
from __future__ import annotations
import argparse
import collections
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


# ── Auto-capture thresholds ───────────────────────────────────────────────────
STILLNESS_WINDOW    = 5           # frames used for motion estimate
STILLNESS_PX        = 2.0         # centroid must move < this over window
NOVELTY_POS_FRAC    = 0.08        # >= 8% image width/height from prev center
NOVELTY_SCALE_FRAC  = 0.12        # >= 12% bbox scale change
NOVELTY_ANGLE_DEG   = 10.0        # >= 10 deg rotation change
MIN_CAPTURE_GAP_S   = 0.4         # min seconds between auto captures per cam


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


def view_descriptor(corners_2d: np.ndarray) -> dict:
    """Cheap pose proxy from 2D corner pixels (no intrinsics required)."""
    pts = corners_2d.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    # Orientation via PCA of zero-mean points
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    # principal axis = eigenvector of largest eigenvalue
    v = evecs[:, -1]
    angle = float(np.degrees(np.arctan2(v[1], v[0])))
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}


def is_novel(desc, prev_descs, img_w, img_h) -> bool:
    if not prev_descs:
        return True
    for p in prev_descs:
        pos_norm = max(abs(desc["cx"] - p["cx"]) / img_w,
                       abs(desc["cy"] - p["cy"]) / img_h)
        scale = max(abs(desc["w"] - p["w"]) / max(p["w"], 1.0),
                    abs(desc["h"] - p["h"]) / max(p["h"], 1.0))
        # wrap angle difference into [0, 90]
        da = abs(desc["angle"] - p["angle"])
        da = min(da, 180.0 - da)
        if (pos_norm < NOVELTY_POS_FRAC
                and scale < NOVELTY_SCALE_FRAC
                and da < NOVELTY_ANGLE_DEG):
            return False
    return True


def solve_calibration(obj_pts_list, img_pts_list, image_size):
    obj = [op.reshape(-1, 3).astype(np.float32) for op in obj_pts_list]
    img = [ip.reshape(-1, 2).astype(np.float32) for ip in img_pts_list]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, None, None, flags=0,
    )
    return rms, K, dist, rvecs, tvecs


def per_view_errors(obj_pts_list, img_pts_list, K, dist, rvecs, tvecs):
    errs = []
    for op, ip, rv, tv in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        o = np.asarray(op).reshape(-1, 3).astype(np.float32)
        i = np.asarray(ip).reshape(-1, 2).astype(np.float32)
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        errs.append(float(np.linalg.norm(proj.reshape(-1, 2) - i, axis=1).mean()))
    return errs


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
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=int, default=config.INTRINSIC_MIN_IMAGES)
    args = p.parse_args()

    print("[multi-intrinsics] Opening cameras...", flush=True)
    caps = open_cameras()
    detector, board = build_detector()

    cam_ids = [cam["id"] for cam, _ in caps]
    pool: dict[str, list[dict]] = {cid: [] for cid in cam_ids}
    pool_descs: dict[str, list[dict]] = {cid: [] for cid in cam_ids}
    capture_history: list[list[str]] = []   # for undo
    image_sizes: dict[str, tuple[int, int]] = {}

    # Stillness tracking: recent centroids per camera
    recent_centroids: dict[str, collections.deque] = {
        cid: collections.deque(maxlen=STILLNESS_WINDOW) for cid in cam_ids
    }
    last_capture_time: dict[str, float] = {cid: 0.0 for cid in cam_ids}

    auto_mode = False
    flash_cam_until: dict[str, float] = {cid: 0.0 for cid in cam_ids}

    win = "Multi-Intrinsics - auto capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(
        "\nPress SPACE to start auto-capture. Move the board slowly through\n"
        f"each camera's FOV. Target: {args.min}+ captures per camera.\n"
        "Press 's' when all cameras reach the target to solve & save.\n",
        flush=True,
    )

    while True:
        frames, dets = [], []
        for cam, cap in caps:
            ok, f = cap.read()
            if not ok:
                f = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), np.uint8)
            frames.append(f)
            if cam["id"] not in image_sizes:
                image_sizes[cam["id"]] = (f.shape[1], f.shape[0])
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            dets.append(detect(detector, g))

        now = time.monotonic()
        contributed_this_frame: list[str] = []

        # ── Per-camera status + auto capture logic ──────────────────────────
        status_per_cam: dict[str, tuple[str, tuple[int, int, int]]] = {}
        for (cam, _), d in zip(caps, dets):
            cid = cam["id"]
            img_w, img_h = image_sizes[cid]
            # Determine status
            n = 0 if d.ch_ids is None else len(d.ch_ids)

            if n < 10:
                status_per_cam[cid] = ("NO BOARD", (0, 0, 255))
                recent_centroids[cid].clear()
                continue

            corners_2d = np.asarray(d.ch_corners).reshape(-1, 2)
            desc = view_descriptor(corners_2d)
            recent_centroids[cid].append((desc["cx"], desc["cy"]))

            # Stillness
            still = False
            if len(recent_centroids[cid]) == STILLNESS_WINDOW:
                arr = np.array(recent_centroids[cid])
                motion = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
                still = motion < STILLNESS_PX

            novel = is_novel(desc, pool_descs[cid], img_w, img_h)

            in_gap = (now - last_capture_time[cid]) < MIN_CAPTURE_GAP_S

            if not still:
                status_per_cam[cid] = ("MOVING", (0, 165, 255))  # orange
            elif not novel:
                status_per_cam[cid] = ("SIMILAR", (180, 180, 180))  # gray
            elif in_gap:
                status_per_cam[cid] = ("WAIT", (100, 255, 255))
            else:
                status_per_cam[cid] = ("READY", (0, 255, 0))

                # Auto capture if armed
                if auto_mode:
                    obj_pts, img_pts = image_object_correspondences(
                        board, d.ch_corners, d.ch_ids,
                    )
                    if obj_pts is not None and img_pts is not None \
                            and len(obj_pts) >= 10:
                        pool[cid].append({"obj_pts": obj_pts, "img_pts": img_pts})
                        pool_descs[cid].append(desc)
                        last_capture_time[cid] = now
                        flash_cam_until[cid] = now + 0.12
                        contributed_this_frame.append(cid)

        if contributed_this_frame:
            capture_history.append(contributed_this_frame)
            pool_sizes = ",".join(f"{c}:{len(pool[c])}" for c in cam_ids)
            print(f"  auto frame #{len(capture_history)} -> "
                  f"{contributed_this_frame}  [{pool_sizes}]", flush=True)

        # ── Visualization ────────────────────────────────────────────────────
        vis_frames = []
        for (cam, _), f, d in zip(caps, frames, dets):
            cid = cam["id"]
            o = draw_overlay(f, d)
            n = 0 if d.ch_ids is None else len(d.ch_ids)
            cnt = len(pool[cid])
            cv2.putText(o, f"{cid}  corners:{n:3d}  pool:{cnt}/{args.min}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            lab, col = status_per_cam.get(cid, ("...", (150, 150, 150)))
            cv2.putText(o, lab, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            # Flash border when this cam just captured
            if now < flash_cam_until[cid]:
                cv2.rectangle(o, (2, 2), (o.shape[1]-3, o.shape[0]-3),
                              (0, 255, 0), 8)
            vis_frames.append(o)

        grid = combine_grid(vis_frames, [cam["id"] for cam, _ in caps])
        mode_txt = "AUTO: ON " if auto_mode else "AUTO: OFF"
        mode_col = (0, 255, 0) if auto_mode else (0, 0, 255)
        total_ok = sum(len(pool[c]) >= args.min for c in cam_ids)
        cv2.putText(grid, f"{mode_txt}   cams met target: {total_ok}/3   "
                          f"total frames: {len(capture_history)}",
                    (10, grid.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_col, 2)
        cv2.putText(grid,
                    "SPACE=toggle auto  c=manual  u=undo  r=reset  s=solve  q=quit",
                    (10, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, grid)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[multi-intrinsics] quit without saving.", flush=True)
            break

        elif key == ord(' '):
            auto_mode = not auto_mode
            print(f"  auto-capture {'ON' if auto_mode else 'OFF'}", flush=True)

        elif key == ord('c'):
            # manual override
            contributed: list[str] = []
            for (cam, _), d in zip(caps, dets):
                if not d.ok(min_corners=10):
                    continue
                obj_pts, img_pts = image_object_correspondences(
                    board, d.ch_corners, d.ch_ids,
                )
                if obj_pts is None or img_pts is None or len(obj_pts) < 10:
                    continue
                corners_2d = np.asarray(d.ch_corners).reshape(-1, 2)
                desc = view_descriptor(corners_2d)
                pool[cam["id"]].append({"obj_pts": obj_pts, "img_pts": img_pts})
                pool_descs[cam["id"]].append(desc)
                last_capture_time[cam["id"]] = time.monotonic()
                flash_cam_until[cam["id"]] = time.monotonic() + 0.12
                contributed.append(cam["id"])
            if contributed:
                capture_history.append(contributed)
                pool_sizes = ",".join(f"{c}:{len(pool[c])}" for c in cam_ids)
                print(f"  manual frame #{len(capture_history)} -> "
                      f"{contributed}  [{pool_sizes}]", flush=True)

        elif key == ord('u'):
            if not capture_history:
                print("  nothing to undo", flush=True)
                continue
            last = capture_history.pop()
            for cid in last:
                if pool[cid]:
                    pool[cid].pop()
                if pool_descs[cid]:
                    pool_descs[cid].pop()
            print(f"  undo frame: rolled back {last}", flush=True)

        elif key == ord('r'):
            for cid in cam_ids:
                pool[cid].clear()
                pool_descs[cid].clear()
                recent_centroids[cid].clear()
                last_capture_time[cid] = 0.0
            capture_history.clear()
            print("  ALL pools reset", flush=True)

        elif key == ord('s'):
            missing = {c: args.min - len(pool[c]) for c in cam_ids
                       if len(pool[c]) < args.min}
            if missing:
                print(f"  [need more] {missing}", flush=True)
                continue

            print(f"\n[multi-intrinsics] Solving each camera...", flush=True)
            all_ok = True
            for cid in cam_ids:
                caps_pool = pool[cid]
                img_size = image_sizes[cid]
                print(f"\n  {cid}: {len(caps_pool)} views, size={img_size}",
                      flush=True)
                try:
                    rms, K, dist, rvecs, tvecs = solve_calibration(
                        [c["obj_pts"] for c in caps_pool],
                        [c["img_pts"] for c in caps_pool],
                        img_size,
                    )
                except Exception as e:
                    print(f"    [error] {e}", flush=True)
                    all_ok = False
                    continue
                errs = per_view_errors(
                    [c["obj_pts"] for c in caps_pool],
                    [c["img_pts"] for c in caps_pool],
                    K, dist, rvecs, tvecs,
                )
                print(f"    RMS = {rms:.4f} px  "
                      f"(per-view min={min(errs):.3f}, max={max(errs):.3f}, "
                      f"mean={np.mean(errs):.3f})", flush=True)
                if rms > config.INTRINSIC_TARGET_RMS_PX:
                    print(f"    [warn] RMS > target "
                          f"{config.INTRINSIC_TARGET_RMS_PX:.2f} px", flush=True)

                out_path = config.CALIB_DIR / f"intrinsics_{cid}.npz"
                cam_cfg = next(c for c in config.CAMERAS if c["id"] == cid)
                np.savez(
                    out_path,
                    K=K, dist=dist, rms=rms,
                    image_size=np.array(img_size),
                    squares_x=config.CHARUCO_SQUARES_X,
                    squares_y=config.CHARUCO_SQUARES_Y,
                    square_len_mm=config.CHARUCO_SQUARE_LEN_MM,
                    marker_len_mm=config.CHARUCO_MARKER_LEN_MM,
                    cam_id=cid,
                    cam_index=cam_cfg["index"],
                    n_views=len(caps_pool),
                )
                print(f"    saved: {out_path}", flush=True)

            if all_ok:
                print("\n[multi-intrinsics] Done.", flush=True)
                break

    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
