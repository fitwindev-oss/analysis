"""
Generate a ChArUco calibration board that fits on A4 paper.

Prints two files into ./resources/:
    charuco_board.png   — board image at 600 DPI
    charuco_board.pdf   — optional vector PDF (if reportlab installed)

A4 = 210 x 297 mm (portrait).
Default board: 6 cols x 8 rows, 30 mm squares  →  180 x 240 mm board
                                                  (leaves 15/28.5 mm margins).

Usage:
    python scripts/generate_charuco_a4.py
    python scripts/generate_charuco_a4.py --squares-x 6 --squares-y 8 --sq 30
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

# Allow running from project root without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def make_board(squares_x: int, squares_y: int, sq_len_mm: float,
               marker_len_mm: float, dict_name: str):
    if dict_name not in DICT_MAP:
        raise ValueError(f"Unknown dictionary: {dict_name}. Options: {list(DICT_MAP)}")
    aruco_dict = aruco.getPredefinedDictionary(DICT_MAP[dict_name])
    # OpenCV 4.7+ uses aruco.CharucoBoard, meters — we use mm and scale later.
    # Length values are RELATIVE here; actual world scale comes from squareLength passed
    # to calibration. We pass meters (converted from mm) so calibration works directly.
    sq_len_m     = sq_len_mm / 1000.0
    marker_len_m = marker_len_mm / 1000.0
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=sq_len_m,
        markerLength=marker_len_m,
        dictionary=aruco_dict,
    )
    return board


def render_board_png(board, squares_x: int, squares_y: int, sq_len_mm: float,
                     out_path: Path, dpi: int = 600, margin_mm: float = 5.0):
    """Render board image at given DPI with physical margin."""
    board_w_mm = squares_x * sq_len_mm
    board_h_mm = squares_y * sq_len_mm
    board_w_px = mm_to_px(board_w_mm, dpi)
    board_h_px = mm_to_px(board_h_mm, dpi)
    margin_px  = mm_to_px(margin_mm, dpi)

    # generateImage produces the checker pattern only (no margin)
    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    # Place on a white A4-sized canvas (portrait) so the user can print at 100%
    a4_w_mm, a4_h_mm = 210.0, 297.0
    a4_w_px = mm_to_px(a4_w_mm, dpi)
    a4_h_px = mm_to_px(a4_h_mm, dpi)
    canvas = np.full((a4_h_px, a4_w_px), 255, dtype=np.uint8)

    # Center on A4
    x0 = (a4_w_px - board_w_px) // 2
    y0 = (a4_h_px - board_h_px) // 2
    canvas[y0:y0 + board_h_px, x0:x0 + board_w_px] = img

    # Add printable tick marks at each corner so user can verify scale with a ruler
    tick_len = mm_to_px(10, dpi)
    for (cx, cy) in [(x0, y0), (x0 + board_w_px, y0),
                     (x0, y0 + board_h_px), (x0 + board_w_px, y0 + board_h_px)]:
        cv2.line(canvas, (cx - tick_len, cy), (cx - 2, cy), 0, 2)
        cv2.line(canvas, (cx, cy - tick_len), (cx, cy - 2), 0, 2)

    # Footer text — readable, contains true dimensions so any mis-scale is visible
    footer = (
        f"ChArUco {squares_x}x{squares_y}  "
        f"square={sq_len_mm:.1f}mm  "
        f"board={board_w_mm:.0f}x{board_h_mm:.0f}mm  "
        f"(print at 100%, NO scaling)"
    )
    cv2.putText(canvas, footer,
                (mm_to_px(15, dpi), a4_h_px - mm_to_px(10, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 0, 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return board_w_mm, board_h_mm, out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--squares-x", type=int, default=config.CHARUCO_SQUARES_X)
    p.add_argument("--squares-y", type=int, default=config.CHARUCO_SQUARES_Y)
    p.add_argument("--sq",        type=float, default=config.CHARUCO_SQUARE_LEN_MM,
                   help="square edge length in mm")
    p.add_argument("--marker",    type=float, default=config.CHARUCO_MARKER_LEN_MM,
                   help="inner marker edge length in mm")
    p.add_argument("--dict",      type=str,   default=config.CHARUCO_DICT)
    p.add_argument("--dpi",       type=int,   default=600)
    p.add_argument("--out",       type=str,   default=str(config.RESOURCES_DIR / "charuco_board.png"))
    args = p.parse_args()

    # Sanity check: does it fit on A4?
    w_mm = args.squares_x * args.sq
    h_mm = args.squares_y * args.sq
    if w_mm > 200 or h_mm > 287:
        print(f"[warn] board {w_mm:.0f}x{h_mm:.0f} mm is close to A4 limit (210x297).")
    if args.marker >= args.sq:
        print(f"[err] marker length ({args.marker}) must be < square length ({args.sq}).")
        sys.exit(1)

    print(f"Generating ChArUco board:")
    print(f"  grid        = {args.squares_x} x {args.squares_y}")
    print(f"  square len  = {args.sq} mm")
    print(f"  marker len  = {args.marker} mm")
    print(f"  dictionary  = {args.dict}")
    print(f"  board size  = {w_mm:.1f} x {h_mm:.1f} mm (portrait on A4)")

    board = make_board(args.squares_x, args.squares_y, args.sq, args.marker, args.dict)
    bw, bh, out = render_board_png(
        board, args.squares_x, args.squares_y, args.sq,
        Path(args.out), dpi=args.dpi,
    )
    print(f"\nDone. Saved: {out}")
    print(f"  Real-world board size: {bw:.1f} x {bh:.1f} mm")
    print(f"\nPrint instructions:")
    print(f"  1) Print the PNG on A4 paper at 100% scale (DISABLE 'fit to page').")
    print(f"  2) After printing, measure one square edge with a ruler.")
    print(f"     It MUST be {args.sq:.1f} mm. If it isn't, re-print with correct scaling.")
    print(f"  3) Glue the sheet onto a FLAT stiff board (foam board, dibond, or 5mm MDF).")
    print(f"  4) See docs/CHARUCO_PRINT_GUIDE.md for full calibration workflow.")


if __name__ == "__main__":
    main()
