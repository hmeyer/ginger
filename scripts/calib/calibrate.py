#!/usr/bin/env python3
"""Run the OpenCV ChArUco calibration on a folder of captured PNGs and
emit a verified slam.toml.

Pairs with capture.py — same board specs (must match the calib.io
PNG on screen). Uses the OpenCV 4.x modern path:
CharucoDetector.detectBoard() -> board.matchImagePoints() ->
cv2.calibrateCamera() with the default 5-term Brown-Conrady model
(k1, k2, p1, p2, k3).

Run via the project venv:
    .venv/bin/python scripts/calib/calibrate.py
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import cv2
import numpy as np

# Must match capture.py.
SQUARES_X = 9
SQUARES_Y = 6
DICT = cv2.aruco.DICT_4X4_100
SQUARE_LEN = 30.0
MARKER_LEN = 22.0

# Acceptance threshold for the global RMS. A Pi camera with this board
# should land well under 0.5 px; anything above usually means motion
# blur, screen moire, or wrong board specs.
RMS_WARN_PX = 0.5
MIN_FRAMES = 8


def per_image_error(obj_pts, img_pts, rvec, tvec, K, D):
    """Mean reprojection error (pixels) for one frame."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    err = np.linalg.norm(proj - img_pts, axis=2)
    return float(err.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--frames", default="calib-frames")
    ap.add_argument("--out", default="slam.toml")
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="print per-frame stats"
    )
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    pngs = sorted(frames_dir.glob("*.png"))
    if len(pngs) < MIN_FRAMES:
        print(
            f"calibrate: only {len(pngs)} frames in {frames_dir}/; "
            f"need at least {MIN_FRAMES}, want 20+",
            file=sys.stderr,
        )
        sys.exit(2)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary
    )
    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        cv2.aruco.DetectorParameters(),
    )

    all_obj: list[np.ndarray] = []
    all_img: list[np.ndarray] = []
    used_names: list[str] = []
    img_size: tuple[int, int] | None = None
    skipped = 0

    for png in pngs:
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skip {png.name}: imread failed")
            skipped += 1
            continue
        size = (gray.shape[1], gray.shape[0])
        if img_size is None:
            img_size = size
        elif size != img_size:
            print(
                f"  skip {png.name}: size {size[0]}x{size[1]} "
                f"!= {img_size[0]}x{img_size[1]}"
            )
            skipped += 1
            continue

        cc, ci, _, _ = detector.detectBoard(gray)
        if cc is None or len(cc) < 6:
            print(f"  skip {png.name}: only {0 if cc is None else len(cc)} corners")
            skipped += 1
            continue
        obj_pts, img_pts = board.matchImagePoints(cc, ci)
        if obj_pts is None or len(obj_pts) < 6:
            print(f"  skip {png.name}: matchImagePoints returned <6")
            skipped += 1
            continue
        all_obj.append(obj_pts)
        all_img.append(img_pts)
        used_names.append(png.name)
        if args.verbose:
            print(f"  {png.name}: {len(obj_pts)} matched points")

    if len(all_obj) < MIN_FRAMES:
        print(
            f"calibrate: only {len(all_obj)} usable frames; need {MIN_FRAMES}+",
            file=sys.stderr,
        )
        sys.exit(2)

    assert img_size is not None
    print(
        f"calibrate: {len(all_obj)} usable frames "
        f"({skipped} skipped) at {img_size[0]}x{img_size[1]}"
    )

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, img_size, None, None
    )

    # Per-frame reprojection error helps spot a single bad capture.
    per_frame = [
        per_image_error(all_obj[i], all_img[i], rvecs[i], tvecs[i], K, D)
        for i in range(len(all_obj))
    ]
    if args.verbose:
        print("\nper-frame mean reprojection error (px):")
        for name, e in sorted(zip(used_names, per_frame), key=lambda kv: kv[1]):
            print(f"  {name}: {e:.3f}")
    worst = max(zip(used_names, per_frame), key=lambda kv: kv[1])
    best = min(zip(used_names, per_frame), key=lambda kv: kv[1])

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    Df = D.flatten()
    k1, k2, p1, p2, k3 = (float(Df[i]) for i in range(5))

    hfov = 2.0 * np.degrees(np.arctan(img_size[0] * 0.5 / fx))
    vfov = 2.0 * np.degrees(np.arctan(img_size[1] * 0.5 / fy))

    print()
    print(f"  RMS reprojection error: {rms:.3f} px "
          f"({'OK' if rms < RMS_WARN_PX else 'HIGH — recapture suggested'})")
    print(f"  best frame: {best[0]} @ {best[1]:.3f} px")
    print(f"  worst frame: {worst[0]} @ {worst[1]:.3f} px")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  k1={k1:+.5f}  k2={k2:+.5f}  p1={p1:+.5f}  p2={p2:+.5f}  k3={k3:+.5f}")
    print(f"  implied HFOV={hfov:.1f} deg  VFOV={vfov:.1f} deg")

    today = dt.date.today().isoformat()
    model = (
        f"opencv-charuco {SQUARES_X}x{SQUARES_Y} DICT_4X4_100, "
        f"calib {today}, n={len(all_obj)}, rms={rms:.3f}px"
    )

    out = Path(args.out)
    out.write_text(
        f"fx = {fx:.6f}\n"
        f"fy = {fy:.6f}\n"
        f"cx = {cx:.6f}\n"
        f"cy = {cy:.6f}\n"
        f"k1 = {k1:.8f}\n"
        f"k2 = {k2:.8f}\n"
        f"p1 = {p1:.8f}\n"
        f"p2 = {p2:.8f}\n"
        f"k3 = {k3:.8f}\n"
        f"width = {img_size[0]}\n"
        f"height = {img_size[1]}\n"
        f'model = "{model}"\n'
        f"verified = true\n"
    )
    print(f"\nwrote {out}  (restart ginger.service for SLAM to pick it up)")


if __name__ == "__main__":
    main()
