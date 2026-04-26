"""
Shared ChArUco helpers - single source of truth for board geometry + detector.

Compatible with OpenCV 4.7+ (aruco.CharucoDetector API).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import cv2.aruco as aruco
import numpy as np

import config


DICT_MAP = {
    "DICT_4X4_50":  aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_5X5_50":  aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_6X6_50":  aruco.DICT_6X6_50,
    "DICT_6X6_250": aruco.DICT_6X6_250,
}


@dataclass
class CharucoDetection:
    ch_corners: np.ndarray | None     # (N, 1, 2) subpixel image coords
    ch_ids: np.ndarray | None         # (N, 1) charuco corner ids
    marker_corners: list | None
    marker_ids: np.ndarray | None

    def ok(self, min_corners: int = 10) -> bool:
        return (
            self.ch_corners is not None
            and self.ch_ids is not None
            and len(self.ch_ids) >= min_corners
        )


def build_board():
    """Build the ChArUco board matching project config (meters)."""
    aruco_dict = aruco.getPredefinedDictionary(DICT_MAP[config.CHARUCO_DICT])
    board = aruco.CharucoBoard(
        (config.CHARUCO_SQUARES_X, config.CHARUCO_SQUARES_Y),
        squareLength=config.CHARUCO_SQUARE_LEN_MM / 1000.0,
        markerLength=config.CHARUCO_MARKER_LEN_MM / 1000.0,
        dictionary=aruco_dict,
    )
    return board, aruco_dict


def build_detector():
    """Build a CharucoDetector matching project config."""
    board, _ = build_board()
    detector = aruco.CharucoDetector(board)
    return detector, board


def detect(detector, gray: np.ndarray) -> CharucoDetection:
    """Run ChArUco detection on a grayscale image. Returns structured result."""
    ch_corners, ch_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    return CharucoDetection(ch_corners, ch_ids, marker_corners, marker_ids)


def draw_overlay(bgr: np.ndarray, det: CharucoDetection) -> np.ndarray:
    """Return a copy of bgr with detected markers + charuco corners drawn."""
    out = bgr.copy()
    if det.marker_corners and det.marker_ids is not None:
        aruco.drawDetectedMarkers(out, det.marker_corners, det.marker_ids)
    if det.ok(min_corners=1):
        aruco.drawDetectedCornersCharuco(out, det.ch_corners, det.ch_ids,
                                         cornerColor=(0, 255, 0))
    return out


def board_object_points(board) -> np.ndarray:
    """
    Return the 3D coordinates (meters) of all charuco inner corners
    in the board's own frame. Shape: (N, 3) where N = (sq_x-1)*(sq_y-1).
    """
    # OpenCV exposes these via board.getChessboardCorners()
    return board.getChessboardCorners().astype(np.float64)


def image_object_correspondences(board, ch_corners, ch_ids):
    """
    Use OpenCV 4.7+ matchImagePoints to get paired (imgPts, objPts) for
    solvePnP / calibrateCamera. Works per-frame.
    """
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    return obj_pts, img_pts
