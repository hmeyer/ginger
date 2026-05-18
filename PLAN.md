# ORB-SLAM Roadmap

Plan for evolving the visual SLAM frontend into a full monocular
ORB-SLAM pipeline on the Raspberry Pi 4. The inline roadmap stub lives in
`src/slam/mod.rs`; this document is the detailed version.

## Where the code is today

What exists is a **feature-tracking frontend, not yet SLAM** — there is
no pose, no map, no geometry.

- **M0** ✅ FAST-9 over an 8-level pyramid + NMS + grid-spread cap (NEON).
- **M1** ✅ Oriented BRIEF + brute-force mutual-NN matching with the Lowe
  ratio test, frame-to-frame.
- Live WebUI overlay, per-stage timing HUD, `slam_bench` A/B harness,
  decoupled frontend thread.

## Critical gaps before any geometry

1. **No camera intrinsics.** Nothing in `src/camera/` or `src/slam/`
   carries `K` / focal length / distortion. Every step below needs a
   calibrated pinhole model. This is the true blocker.
2. **No linear-algebra / optimization backbone.** `Cargo.toml` has no
   `nalgebra`. Two-view init, motion-only BA, and local BA all need
   SE3/SO3 Lie ops and a sparse least-squares solver. One-time
   architectural decision.
3. **No offline replay harness.** SLAM correctness cannot be iterated
   from the live camera. `slam_bench` is timing-only on synthetic noise.
   A deterministic image-sequence replay (recorded clip or a
   TUM/EuRoC-style set) is required or every regression is invisible.

## Milestones

### M2 — Calibration + pinhole model + replay harness

Foundation only; no SLAM yet. **Detailed plan:
[`m2-plan.md`](m2-plan.md)** (camera prior, hardware-accel /
NEON strategy, math backbone, work breakdown, exit criteria).

Decided: proceed on the Pi Camera "rev 1.3" standard-lens prior
(FOV-derived, flagged unverified); pure-Rust math (`nalgebra` + a small
`lie`/solver, no BLAS); reuse the `crates/fast` scalar→NEON→parity-test
discipline for all geometry kernels.

Open follow-up (deferred, needs a physical target): **proper camera
calibration** — offline OpenCV ChArUco tool emitting a verified
`slam.toml`. Not kalibr.

- Calibrate the OV5647 (checkerboard) or seed known intrinsics plus a
  calibration utility; store `K` + distortion.
- `src/slam/camera_model.rs`: project / unproject / keypoint
  undistortion.
- Offline replay: feed a recorded YUYV/PNG sequence through
  `detect_features` → matching, deterministically. Extend the
  `slam_bench` pattern.
- Decide the math backbone (recommended: `nalgebra` + hand-rolled SE3 +
  a small Gauss-Newton / Levenberg-Marquardt; full BA libraries are
  heavy for a Pi 4).

### M3 — Two-view monocular initialization

- Parallel Fundamental (8-point) and Homography (4-point) estimation,
  each RANSAC-scored; ORB-SLAM's H-vs-F model selection heuristic.
- Recover `(R, t)` via E or H decomposition + cheirality check;
  triangulate the initial map; two seed keyframes; arbitrary scale.
- **Visible deliverable:** first point cloud + two poses, drawn as a
  top-down 2D plot in the WebUI (extend `SlamSnapshot` or add
  `/api/slam/map`).

### M4 — Tracking thread (live 6-DoF pose)

- Constant-velocity motion model → predicted pose.
- Guided projection matching (search window) replacing brute-force.
- Motion-only BA (SE3, robust Huber); reference-keyframe fallback when
  the motion model fails.
- **Deliverable:** live camera trajectory in the WebUI.

### M5 — Local Mapping thread

- Keyframe insertion policy; covisibility graph + spanning tree.
- Epipolar-guided triangulation of new map points.
- **Local BA** over a keyframe window — the Pi 4 performance crux; runs
  on a slow background thread, as the current decoupled design
  anticipates.
- Map-point and keyframe culling.
- **Visible deliverable:** growing map point cloud + keyframes on the
  top-down canvas.

### M6 — Relocalization + loop closing

- Bag-of-Words vocabulary over ORB for place recognition.
- PnP-RANSAC relocalization on track loss.
- Loop detection → Sim3 (monocular scale drift) → Essential-graph pose
  optimization → global BA.
- **Visible deliverable:** "loop detected" event + trajectory snapping
  straighter on the canvas.

**WebUI surface:** all milestones draw into the single top-down canvas +
`#slam-hud` stubbed in M2 (see [`m2-plan.md`](m2-plan.md) §
WebUI-visible outputs); M2 itself only surfaces the calibration-status /
sensor-mode diagnostics.

## Sequencing & risks

- **Strict dependency chain:** M2 → M3 → M4 → M5; M6 last. Do not start
  M3 until M2's replay harness exists, or correctness bugs become
  untraceable.
- **The Pi 4 is the binding constraint.** Plan from the start to drop
  resolution / feature count for the geometry path and to run local BA
  and loop closing on background threads at a lower rate. The existing
  NEON and decoupled-thread work already sets this up well.
- **Biggest single risk:** local BA (M5) performance on ARM. De-risk
  early by prototyping the solver during M2.
