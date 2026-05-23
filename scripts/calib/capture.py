#!/usr/bin/env python3
"""Capture ChArUco calibration frames via rpicam-still and run ChArUco
detection so the operator gets immediate "is this pose usable?"
feedback.

Requires ginger.service to be stopped (libcamera lets only one process
claim the camera at a time):

    systemctl --user stop ginger.service

Pair with the calib.io-generated ChArUco board (9x6 squares,
DICT_4X4_100). Capture ~20 varied poses (distance, yaw, pitch, roll).
Then run calibrate.py.

Run via the project venv:
    .venv/bin/python scripts/calib/capture.py
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Board specs — must match the calib.io-generated PNG/PDF on screen.
SQUARES_X = 9
SQUARES_Y = 6
DICT = cv2.aruco.DICT_4X4_100
SQUARE_LEN = 30.0
MARKER_LEN = 22.0

# Reject captures with fewer than this many ChArUco corners — the pose
# is too oblique / too far / too occluded to contribute much.
MIN_CORNERS = 20

# Match the SLAM runtime resolution exactly (libcamera ViewFinder mode
# at 800x600 on the OV5647 uses the 1296x972 binned sensor mode and
# downscales — which is the same path rpicam-still takes at the same
# output resolution, so FOV matches).
OUT_W = 800
OUT_H = 600

# Per-shot timeout for rpicam-still in milliseconds — enough for AE/AWB
# to converge but quick enough for an interactive workflow.
SHOT_TIMEOUT_MS = 1500


def rpicam_still(out_path: Path) -> None:
    """Run rpicam-still synchronously, writing a PNG to out_path."""
    cmd = [
        "rpicam-still",
        "--width", str(OUT_W),
        "--height", str(OUT_H),
        "--encoding", "png",
        "--timeout", str(SHOT_TIMEOUT_MS),
        "--awb", "auto",
        "--nopreview",
        "-o", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(
            "rpicam-still failed (is ginger.service still holding the "
            "camera? `systemctl --user stop ginger.service`):\n  "
            + "\n  ".join(msg)
        )
    if not out_path.exists():
        raise RuntimeError(f"rpicam-still: no output file at {out_path}")


def build_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary
    )
    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        cv2.aruco.DetectorParameters(),
    )
    return board, detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--out", default="calib-frames", help="output directory for accepted PNGs"
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.png"))
    next_idx = len(existing)
    if existing:
        print(f"capture: resuming, {next_idx} frames already in {out_dir}/")

    _, detector = build_detector()
    target = (SQUARES_X - 1) * (SQUARES_Y - 1)
    print(
        f"capture: {SQUARES_X}x{SQUARES_Y} ChArUco, DICT_4X4_100, "
        f"target {target} interior corners, output {OUT_W}x{OUT_H}"
    )
    print("capture: each shot takes ~2s (libcamera AE convergence)")
    print(f"capture: writing to {out_dir}/")
    print("capture: press Enter to grab a frame, Ctrl-D (or q+Enter) to finish\n")

    accepted = 0
    rejected = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="ginger-calib-"))
    pending = tmp_dir / "pending.png"

    try:
        while True:
            try:
                line = input(
                    f"[ok:{accepted} bad:{rejected}] Enter to capture: "
                )
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.strip().lower() == "q":
                break

            try:
                rpicam_still(pending)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            gray = cv2.imread(str(pending), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  ERROR: cv2.imread failed on {pending}")
                continue

            cc, _, _, _ = detector.detectBoard(gray)
            n = 0 if cc is None else len(cc)
            cov = 100.0 * n / target
            if n < MIN_CORNERS:
                print(
                    f"  REJECT: only {n}/{target} corners ({cov:.0f}%) — "
                    "reposition (closer / more centered / less oblique / less reflection)"
                )
                rejected += 1
                continue

            dst = out_dir / f"{next_idx:04d}.png"
            # Save grayscale for storage efficiency and to match what
            # calibrate.py loads anyway.
            cv2.imwrite(str(dst), gray)
            print(
                f"  saved {dst.name}: {n}/{target} corners ({cov:.0f}%), "
                f"{gray.shape[1]}x{gray.shape[0]}"
            )
            accepted += 1
            next_idx += 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\ndone: {accepted} accepted, {rejected} rejected -> {out_dir}/")
    if accepted < 20:
        print(
            f"note: 20+ accepted frames recommended for a good calibration "
            f"(have {accepted}); run again to add more"
        )


if __name__ == "__main__":
    main()
