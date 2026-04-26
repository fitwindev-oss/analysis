"""
Skeleton-based multi-view calibration primitives.

Given per-camera 2D keypoints (COCO-17) across synchronized frames, estimate:
  - camera intrinsics (lightweight: fx=fy=f, cx=W/2, cy=H/2, no distortion)
  - camera extrinsics (R, t) up to a single global scale
  - 3D joint positions per frame

The single global scale is RESOLVED LATER by align_world_from_force.py using
the force plate CoP as a known-mm anchor. Here we only need internal
self-consistency.

Algorithm:
  1) Fix cam0 at identity. Assume initial f = image_width for all cams.
  2) Pair cam0 <-> cam1: build 2D correspondences from high-confidence joints,
     estimate Essential matrix + recoverPose -> (R1, t1), |t1| = 1.
  3) Triangulate 3D points from (cam0, cam1). These live in cam0's frame,
     with arbitrary scale = |t1| = 1.
  4) Use cam2's 2D observations of those same 3D points -> solvePnP ->
     (R2, t2) in the SAME scale as step 3.
  5) (Optional) Bundle adjustment via scipy.least_squares, refining
     (R_i, t_i) for i in {1, 2} and 3D joint positions. cam0 stays fixed.

Joint confidence threshold is applied per-observation; missing/low-confidence
observations are dropped entirely, and frames with too few valid 3-view
correspondences are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import cv2

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class CameraParams:
    """Lightweight pinhole (no distortion) for initial skeleton calibration."""
    cam_id: str
    image_size: tuple[int, int]    # (W, H)
    K: np.ndarray                  # (3, 3)
    dist: np.ndarray               # (5,) - kept zero in this module
    R: np.ndarray                  # (3, 3)  world -> cam
    t: np.ndarray                  # (3,)

    @property
    def P(self) -> np.ndarray:
        """Projection matrix K @ [R | t]."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])


def initial_intrinsics(image_size: tuple[int, int],
                       hfov_deg: float = 60.0) -> np.ndarray:
    """Rough initial K assuming a symmetric pinhole.

    focal = (W/2) / tan(hfov/2)

    Typical webcam horizontal FOVs:
        Microsoft LifeCam Studio: ~60 deg
        Samsung PC-CAM USB 2.0:   ~55 deg
        Logitech StreamCam:       ~78 deg  (wide!)
    BA optimizes focal length later, so a rough guess is enough - but a
    very wrong initial guess (e.g. 640 for a 78-deg wide-angle lens) traps
    the optimization in a bad basin.
    """
    W, H = image_size
    hfov_rad = np.deg2rad(hfov_deg)
    f = (W / 2.0) / np.tan(hfov_rad / 2.0)
    return np.array([
        [f, 0, W / 2.0],
        [0, f, H / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)


# Per-camera FOV estimates (horizontal degrees). Order matches config.CAMERAS.
DEFAULT_HFOV_DEG = {
    "C0": 60.0,   # Microsoft LifeCam Studio
    "C1": 55.0,   # Samsung PC-CAM
    "C2": 78.0,   # Logitech StreamCam (wide)
}


def select_correspondences(kpts: dict[str, np.ndarray],
                           scores: dict[str, np.ndarray],
                           cam_ids: list[str],
                           conf_thresh: float = 0.4,
                           min_joints_per_frame: int = 6,
                           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build per-frame 2D correspondences visible in ALL cameras in cam_ids.

    Inputs:
        kpts[cam]:   (N_frames, 17, 2) pixel coords
        scores[cam]: (N_frames, 17)   confidences
        cam_ids: the cameras whose frames must ALL have the joint visible

    Returns:
        corr_2d[cam]: (M, 2) flattened valid 2D points for that cam
        meta with per-element (frame_idx, joint_idx) for debugging
    """
    n_frames = kpts[cam_ids[0]].shape[0]
    cam_kp = [kpts[c] for c in cam_ids]
    cam_sc = [scores[c] for c in cam_ids]

    valid_mask = np.ones((n_frames, 17), dtype=bool)
    for s in cam_sc:
        valid_mask &= (s >= conf_thresh)

    # Flatten (frame, joint) -> row index
    rows: list[tuple[int, int]] = []
    out: dict[str, list] = {c: [] for c in cam_ids}
    for fi in range(n_frames):
        visible_joints = np.where(valid_mask[fi])[0]
        if len(visible_joints) < min_joints_per_frame:
            continue
        for j in visible_joints:
            rows.append((fi, int(j)))
            for c, arr in zip(cam_ids, cam_kp):
                out[c].append(arr[fi, j])

    result_2d = {c: np.asarray(out[c], dtype=np.float64) for c in cam_ids}
    meta = {
        "rows": np.asarray(rows, dtype=np.int32),
        "n_total_observations": len(rows),
    }
    return result_2d, None, meta


def estimate_pair_pose(pts1: np.ndarray, pts2: np.ndarray,
                       K1: np.ndarray, K2: np.ndarray,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given 2D points in two cameras, estimate pose of cam2 relative to cam1
    via Essential matrix + recoverPose.  |t| = 1.

    Returns (R, t, inlier_mask).
    """
    # Normalize both sets with their own intrinsics, then use a unit K for
    # findEssentialMat (the standard workaround when K1 != K2).
    def normalize(pts, K):
        Kinv = np.linalg.inv(K)
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        n = (Kinv @ homog.T).T
        return n[:, :2]

    p1n = normalize(pts1, K1).astype(np.float64)
    p2n = normalize(pts2, K2).astype(np.float64)
    I = np.eye(3)
    # threshold ~ 3 px in 640-wide image (scaled to normalized coords)
    # 2D keypoint noise from RTMPose is 3-5 px so this must be loose enough.
    thresh_norm = 3.0 / max(K1[0, 0], K2[0, 0])
    E, mask = cv2.findEssentialMat(
        p1n, p2n, I, method=cv2.RANSAC, prob=0.999, threshold=thresh_norm,
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed - check point variety")
    _, R, t, pose_mask = cv2.recoverPose(E, p1n, p2n, I, mask=mask)
    # recoverPose returns R, t such that X_cam2 = R * X_cam1 + t, with |t|=1.
    return R, t.reshape(3), pose_mask.ravel().astype(bool)


def triangulate_two_view(P1: np.ndarray, P2: np.ndarray,
                         pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Linear DLT triangulation; returns (N, 3)."""
    X4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X3 = (X4[:3] / X4[3]).T
    return X3


def solve_pnp(obj_pts: np.ndarray, img_pts: np.ndarray,
              K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust PnP with RANSAC. Returns (R, t)."""
    obj = obj_pts.astype(np.float32).reshape(-1, 1, 3)
    img = img_pts.astype(np.float32).reshape(-1, 1, 2)
    # Use UPNP or ITERATIVE depending on count
    if len(obj) >= 4:
        dist = np.zeros(5)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dist, reprojectionError=6.0,
            iterationsCount=500, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnPRansac failed")
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3)
    raise RuntimeError(f"not enough points for PnP: got {len(obj)}")


# ── Bundle adjustment ────────────────────────────────────────────────────────

def _params_to_state(params: np.ndarray, n_cams_total: int, n_pts: int,
                     optimize_focal: bool
                     ) -> tuple[np.ndarray,
                                list[tuple[np.ndarray, np.ndarray]],
                                np.ndarray]:
    """
    Unpack flat parameter vector.

    Layout (optimize_focal=True):
        [ f0, f1, f2, rvec1, t1, rvec2, t2, X0..Xn ]
        (cam0 extrinsics fixed at identity; focals are always free)

    Layout (optimize_focal=False):
        [ rvec1, t1, rvec2, t2, X0..Xn ]
        (focals passed separately)
    """
    offset = 0
    if optimize_focal:
        focals = params[:n_cams_total]
        offset = n_cams_total
    else:
        focals = None
    cam_poses: list[tuple[np.ndarray, np.ndarray]] = [
        (np.eye(3), np.zeros(3))  # cam0 fixed
    ]
    for _ in range(n_cams_total - 1):
        rvec = params[offset:offset + 3]
        tvec = params[offset + 3:offset + 6]
        R, _ = cv2.Rodrigues(rvec)
        cam_poses.append((R, tvec))
        offset += 6
    pts = params[offset:offset + n_pts * 3].reshape(n_pts, 3)
    return focals, cam_poses, pts


def _pack_state(focals: np.ndarray | None,
                cams_R_t_free: list[tuple[np.ndarray, np.ndarray]],
                pts_3d: np.ndarray) -> np.ndarray:
    buf = []
    if focals is not None:
        buf.append(np.asarray(focals, dtype=np.float64).ravel())
    for R, t in cams_R_t_free:
        rvec, _ = cv2.Rodrigues(R)
        buf.append(rvec.ravel())
        buf.append(t.ravel())
    buf.append(pts_3d.ravel())
    return np.concatenate(buf)


def _build_residuals(params: np.ndarray,
                     n_cams_total: int, n_pts: int,
                     Ks_fixed: list[np.ndarray] | None,
                     cxcy: np.ndarray,               # (n_cams, 2)
                     observations_arr: np.ndarray,   # (n_obs, 4): ci, pi, u, v
                     optimize_focal: bool):
    focals, cam_poses, pts = _params_to_state(
        params, n_cams_total, n_pts, optimize_focal
    )
    ci = observations_arr[:, 0].astype(np.int32)
    pi = observations_arr[:, 1].astype(np.int32)
    obs_uv = observations_arr[:, 2:4]

    # Vectorized projection
    Rs = np.stack([cp[0] for cp in cam_poses])    # (n_cams, 3, 3)
    ts = np.stack([cp[1] for cp in cam_poses])    # (n_cams, 3)
    X  = pts[pi]                                  # (n_obs, 3)
    R_sel = Rs[ci]                                # (n_obs, 3, 3)
    t_sel = ts[ci]                                # (n_obs, 3)
    Xc = np.einsum('nij,nj->ni', R_sel, X) + t_sel   # (n_obs, 3)
    z = Xc[:, 2]
    # Prevent division-by-zero / points behind camera
    bad = z <= 1e-6
    z_safe = np.where(bad, 1.0, z)
    u_norm = Xc[:, 0] / z_safe
    v_norm = Xc[:, 1] / z_safe

    if optimize_focal:
        f_sel = focals[ci]
    else:
        f_sel = np.array([Ks_fixed[c][0, 0] for c in range(n_cams_total)])[ci]

    cx_sel = cxcy[ci, 0]
    cy_sel = cxcy[ci, 1]
    u = f_sel * u_norm + cx_sel
    v = f_sel * v_norm + cy_sel
    ru = u - obs_uv[:, 0]
    rv = v - obs_uv[:, 1]
    # For bad (behind-camera) rows, push huge residual
    ru = np.where(bad, 1e4, ru)
    rv = np.where(bad, 1e4, rv)
    return np.stack([ru, rv], axis=1).ravel()


def _ba_jacobian_sparsity(n_cams_total: int, n_pts: int,
                          observations_arr: np.ndarray,
                          optimize_focal: bool):
    n_obs = observations_arr.shape[0]
    n_cams_free_extr = n_cams_total - 1
    focal_cols = n_cams_total if optimize_focal else 0
    n_params = focal_cols + 6 * n_cams_free_extr + 3 * n_pts
    jac = lil_matrix((2 * n_obs, n_params), dtype=np.uint8)

    for row in range(n_obs):
        ci = int(observations_arr[row, 0])
        pi = int(observations_arr[row, 1])
        if optimize_focal:
            # focal of cam ci depends on column ci
            jac[2 * row,     ci] = 1
            jac[2 * row + 1, ci] = 1
        if ci >= 1:   # non-fixed extrinsics
            cam_free_idx = ci - 1
            base = focal_cols + cam_free_idx * 6
            jac[2 * row,     base:base + 6] = 1
            jac[2 * row + 1, base:base + 6] = 1
        pt_base = focal_cols + 6 * n_cams_free_extr + pi * 3
        jac[2 * row,     pt_base:pt_base + 3] = 1
        jac[2 * row + 1, pt_base:pt_base + 3] = 1
    return jac.tocsr()


def bundle_adjust(cams: list[CameraParams],
                  pts_3d_init: np.ndarray,
                  obs_by_cam: dict[str, tuple[np.ndarray, np.ndarray]],
                  max_nfev: int = 300,
                  verbose: bool = True,
                  loss: str = "huber",
                  f_scale: float = 3.0,
                  optimize_focal: bool = True,
                  ) -> tuple[list[CameraParams], np.ndarray, dict]:
    """
    Jointly refine cam[1..] extrinsics, all cam focal lengths (optional),
    and 3D points. cam[0] stays at identity extrinsics for gauge fixing.

    Args:
        cams: CameraParams in order. Cam 0's extrinsics are fixed.
        pts_3d_init: (M, 3) initial 3D points.
        obs_by_cam: cam_id -> (point indices (K,), 2D pixels (K, 2))
        optimize_focal: if True, per-camera focal length (fx=fy) is a free
                        parameter with cx=W/2, cy=H/2 fixed.
    """
    cam_ids = [c.cam_id for c in cams]
    cam_id_to_idx = {c.cam_id: i for i, c in enumerate(cams)}
    n_cams_total = len(cams)
    n_pts = pts_3d_init.shape[0]

    # Stack observations as ndarray of (ci, pi, u, v) for vectorized residuals
    rows = []
    for cid, (pt_idxs, pxs) in obs_by_cam.items():
        ci = cam_id_to_idx[cid]
        for k, pi in enumerate(pt_idxs):
            rows.append((ci, int(pi), float(pxs[k, 0]), float(pxs[k, 1])))
    obs_arr = np.asarray(rows, dtype=np.float64)

    cxcy = np.array([[c.K[0, 2], c.K[1, 2]] for c in cams], dtype=np.float64)
    focals_init = np.array([c.K[0, 0] for c in cams], dtype=np.float64)
    Ks_fixed = [c.K for c in cams] if not optimize_focal else None

    cams_free_Rt = [(c.R.copy(), c.t.copy()) for c in cams[1:]]
    x0 = _pack_state(
        focals_init if optimize_focal else None,
        cams_free_Rt,
        pts_3d_init,
    )

    jac_sp = _ba_jacobian_sparsity(n_cams_total, n_pts, obs_arr, optimize_focal)
    if verbose:
        mode = "+focal" if optimize_focal else ""
        print(f"  [ba{mode}] n_cams={n_cams_total}  n_pts={n_pts}  "
              f"n_obs={len(rows)}  params={len(x0)}", flush=True)
        if optimize_focal:
            print(f"  [ba] initial focals: "
                  f"{[f'{f:.1f}' for f in focals_init]}", flush=True)

    result = least_squares(
        _build_residuals, x0,
        jac_sparsity=jac_sp,
        x_scale="jac",
        ftol=1e-6, xtol=1e-8, gtol=1e-10,
        method="trf",
        loss=loss, f_scale=f_scale,
        args=(n_cams_total, n_pts, Ks_fixed, cxcy, obs_arr, optimize_focal),
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
    )

    focals_new, cam_poses, pts = _params_to_state(
        result.x, n_cams_total, n_pts, optimize_focal
    )
    if focals_new is None:
        focals_new = focals_init

    refined = []
    for i, c in enumerate(cams):
        R, t = cam_poses[i]
        K_new = c.K.copy()
        if optimize_focal:
            K_new[0, 0] = float(focals_new[i])
            K_new[1, 1] = float(focals_new[i])
        refined.append(CameraParams(
            cam_id=c.cam_id, image_size=c.image_size,
            K=K_new, dist=c.dist.copy(),
            R=R.copy(), t=t.copy(),
        ))

    # Compute RAW reprojection RMS (without Huber damping) for reporting
    raw_res = _build_residuals(result.x, n_cams_total, n_pts,
                               Ks_fixed, cxcy, obs_arr, optimize_focal)
    rms_px = float(np.sqrt(np.mean(raw_res ** 2)))

    info = {
        "rms_pixel": rms_px,
        "cost": float(result.cost),
        "status": int(result.status),
        "n_observations": len(rows),
        "focals_final": focals_new.copy(),
    }
    if verbose:
        print(f"  [ba] done. RMS (raw) = {rms_px:.3f} px", flush=True)
        if optimize_focal:
            print(f"  [ba] final focals: "
                  f"{[f'{f:.1f}' for f in focals_new]}", flush=True)
    return refined, pts, info


def iterative_bundle_adjust(cams: list[CameraParams],
                            pts_3d_init: np.ndarray,
                            obs_by_cam: dict[str, tuple[np.ndarray, np.ndarray]],
                            n_iters: int = 2,
                            outlier_px: float = 8.0,
                            **kwargs):
    """Run BA, discard observations with residual > outlier_px, re-run.

    This cleans up wildly-wrong 2D detections that Huber loss damps but
    doesn't remove. Typically yields 2-3x better RMS on noisy data.
    """
    current_obs = {cid: (p.copy(), u.copy()) for cid, (p, u) in obs_by_cam.items()}
    last_info: dict = {}
    pts = pts_3d_init
    for it in range(n_iters):
        tag = "initial" if it == 0 else f"refine pass {it}"
        print(f"\n  [ba] === {tag} ===", flush=True)
        cams, pts, info = bundle_adjust(cams, pts, current_obs, **kwargs)
        last_info = info

        if it == n_iters - 1:
            break

        # Reject outliers per observation
        cam_id_to_idx = {c.cam_id: i for i, c in enumerate(cams)}
        kept_any = False
        new_obs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        dropped_total = 0
        for cid, (pt_idxs, pxs) in current_obs.items():
            ci = cam_id_to_idx[cid]
            R = cams[ci].R; t = cams[ci].t; K = cams[ci].K
            X = pts[pt_idxs]                       # (N, 3)
            Xc = (R @ X.T).T + t                   # (N, 3)
            z = Xc[:, 2].copy()
            valid = z > 1e-6
            u = np.where(valid, K[0, 0] * Xc[:, 0] / np.where(valid, z, 1.0)
                                  + K[0, 2], 1e6)
            v = np.where(valid, K[1, 1] * Xc[:, 1] / np.where(valid, z, 1.0)
                                  + K[1, 2], 1e6)
            err = np.sqrt((u - pxs[:, 0]) ** 2 + (v - pxs[:, 1]) ** 2)
            keep = (err <= outlier_px) & valid
            dropped_total += int(np.sum(~keep))
            if np.sum(keep) >= 4:
                new_obs[cid] = (pt_idxs[keep], pxs[keep])
                kept_any = True
        print(f"  [ba] dropped {dropped_total} obs with residual > "
              f"{outlier_px} px", flush=True)
        if not kept_any or dropped_total == 0:
            break
        current_obs = new_obs
    return cams, pts, last_info


# ── End-to-end helper ────────────────────────────────────────────────────────

def calibrate_from_three_view_poses(
    cam_ids: list[str],
    image_sizes: dict[str, tuple[int, int]],
    kpts: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    conf_thresh: float = 0.45,
    min_joints_per_frame: int = 6,
    do_bundle_adjust: bool = True,
    optimize_focal: bool = True,
    n_refine_iters: int = 2,
    outlier_px: float = 8.0,
    fov_overrides: dict[str, float] | None = None,
) -> tuple[list[CameraParams], np.ndarray, dict]:
    """
    Run the full 3-view initialization + iterative BA with focal optimization
    and outlier rejection.

    Returns (cams, pts_3d, info).
    pts_3d is in arbitrary global scale (|t_cam1| ~ 1 initially).
    """
    assert len(cam_ids) == 3, "this helper is hardcoded for 3 cameras"

    # 1. Correspondences visible in ALL 3 cams
    corr_2d, _, meta = select_correspondences(
        kpts, scores, cam_ids, conf_thresh=conf_thresh,
        min_joints_per_frame=min_joints_per_frame,
    )
    n_obs = meta["n_total_observations"]
    if n_obs < 60:
        raise RuntimeError(
            f"only {n_obs} 3-view observations (need >=60). Increase "
            f"recording length or lower conf_thresh."
        )
    print(f"  [init] {n_obs} 3-view correspondences", flush=True)

    # 2. Per-camera initial intrinsics using known FOV priors. The initial
    #    focal matters A LOT for the subsequent essential-matrix step.
    fov_map = dict(DEFAULT_HFOV_DEG)
    if fov_overrides:
        fov_map.update(fov_overrides)
    Ks = {}
    for c in cam_ids:
        hfov = fov_map.get(c, 60.0)
        Ks[c] = initial_intrinsics(image_sizes[c], hfov_deg=hfov)
        f = Ks[c][0, 0]
        print(f"  [init] {c}: hfov={hfov:.1f}deg  init focal={f:.1f} px",
              flush=True)

    # 3. Pair pose cam0-cam1
    c0, c1, c2 = cam_ids
    R01, t01, mask01 = estimate_pair_pose(
        corr_2d[c0], corr_2d[c1], Ks[c0], Ks[c1]
    )
    inl = mask01
    print(f"  [init] cam0-cam1 essential inliers: {int(inl.sum())}/{len(inl)}",
          flush=True)
    pts1 = corr_2d[c0][inl]
    pts2 = corr_2d[c1][inl]
    pts3 = corr_2d[c2][inl]

    P0 = Ks[c0] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = Ks[c1] @ np.hstack([R01,       t01.reshape(3, 1)])

    # 4. Triangulate
    X3d = triangulate_two_view(P0, P1, pts1, pts2)
    cam1_z = (R01 @ X3d.T).T[:, 2] + t01[2]
    keep = (X3d[:, 2] > 0) & (cam1_z > 0)
    X3d = X3d[keep]
    pts1 = pts1[keep]; pts2 = pts2[keep]; pts3 = pts3[keep]
    if len(X3d) < 20:
        raise RuntimeError(f"only {len(X3d)} valid 3D points after cheirality "
                           f"filtering.")

    # 5. PnP for cam2
    R02, t02 = solve_pnp(X3d, pts3, Ks[c2])

    cams = [
        CameraParams(c0, image_sizes[c0], Ks[c0], np.zeros(5),
                     np.eye(3), np.zeros(3)),
        CameraParams(c1, image_sizes[c1], Ks[c1], np.zeros(5),
                     R01, t01),
        CameraParams(c2, image_sizes[c2], Ks[c2], np.zeros(5),
                     R02, t02),
    ]

    # 6. Iterative bundle adjust with outlier rejection + focal optimization
    info = {}
    if do_bundle_adjust:
        obs_by_cam = {
            c0: (np.arange(len(X3d)), pts1),
            c1: (np.arange(len(X3d)), pts2),
            c2: (np.arange(len(X3d)), pts3),
        }
        cams, X3d, info = iterative_bundle_adjust(
            cams, X3d, obs_by_cam,
            n_iters=n_refine_iters,
            outlier_px=outlier_px,
            optimize_focal=optimize_focal,
            verbose=True,
        )
    return cams, X3d, info
