# ORB-SLAM Roadmap

Plan for evolving the visual SLAM frontend into a full monocular
ORB-SLAM pipeline on the Raspberry Pi 4. The inline roadmap stub lives in
`src/slam/mod.rs`; this document is the detailed version.

## Where the code is today

A monocular **two-view bootstrap** runs end-to-end: the frontend
detects/matches features, accumulates parallax against an anchor frame,
and recovers a relative pose + an initial triangulated point cloud,
surfaced live in the WebUI. No tracking/keyframes/BA yet (M4+).

- **M0** ✅ FAST-9 over an 8-level pyramid + NMS + grid-spread cap (NEON).
- **M1** ✅ Oriented BRIEF + brute-force mutual-NN matching (Lowe ratio).
- **M2** ✅ Calibration prior + pinhole model + math backbone + mock
  camera + headless replay/CI ([`m2-plan.md`](m2-plan.md)).
- **M3** ✅ Two-view initialization (essential + homography) wired into
  the live pipeline with a top-down map in the WebUI.
- Live WebUI overlay + HUD, `slam_bench` / `slam_replay` harnesses,
  decoupled frontend thread.

## Critical gaps — resolved in M2

All three original blockers are closed:

1. ~~No camera intrinsics~~ → `CameraModel` + the rev 1.3 FOV prior
   (flagged `UNVERIFIED`; real ChArUco calibration is the deferred
   follow-up).
2. ~~No linear-algebra / optimization backbone~~ → pure-Rust
   `ginger-slam-core` (`nalgebra`, hand-rolled SO3/SE3 `lie`, dense
   LM + Huber), no BLAS.
3. ~~No offline replay harness~~ → mock camera + deterministic
   `slam_replay`, headless and CI-gated.

## Milestones

### M2 — Calibration + pinhole model + replay harness ✅ (foundation done)

Foundation only; no SLAM behaviour. **Detailed plan + status:
[`m2-plan.md`](m2-plan.md)**.

Done: `ginger-slam-core` (hand-rolled SO3/SE3 `lie`, pinhole +
Brown–Conrady `CameraModel`, dense LM + Huber, PGM `dataset`); rev 1.3
FOV prior wired into `slam` with an `UNVERIFIED` HUD badge over
`/api/slam/stream`; `/api/slam/map` + top-down canvas stub; **mock
camera** behind an opt-out `libcamera` Cargo feature so the whole crate
builds/tests headless; deterministic `slam_replay` runner; `slam_bench`
geometry stages; `headless` + `aarch64-check` CI jobs. Pure-Rust math
(`nalgebra`, no BLAS); `crates/fast` scalar→NEON→parity discipline kept
for future kernels.

Open follow-ups (deferred, need the physical robot + target in one
session):
- **Proper camera calibration** — offline OpenCV ChArUco tool emitting a
  verified `slam.toml`. Not kalibr.
- **Frame recorder** — dump live libcamera frames to `*.pgm` to feed the
  (already-built) replay harness with real robot scenes.

### M3 — Two-view monocular initialization ✅

- Done in `ginger_slam_core::twoview`: normalized 8-point essential +
  4-point homography, seeded RANSAC, ORB-SLAM `R_H` model selection,
  essential SVD + Faugeras homography decomposition, cheirality, DLT
  triangulation (29 deterministic synthetic-scene tests).
- Wired into `slam::run`: anchor frame + median-parallax gate →
  `CameraModel` undistort/normalize → `twoview::initialize`; one-shot,
  re-anchors if the anchor goes stale. Arbitrary monocular scale.
- **Visible deliverable ✅:** `MapSnapshot` over `GET /api/slam/map`;
  the WebUI `map (M3)` mode renders the top-down point cloud + the two
  camera centres.
- Deferred to M5: promoting the two views to proper **keyframes** +
  covisibility (M3 yields poses + points, not a keyframe graph yet).

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

- **Strict dependency chain:** M2 ✅ → M3 ✅ → **M4 (next)** → M5;
  M6 last.
- **The Pi 4 is the binding constraint.** Plan from the start to drop
  resolution / feature count for the geometry path and to run local BA
  and loop closing on background threads at a lower rate. The existing
  NEON and decoupled-thread work already sets this up well.
- **Biggest single risk:** local BA (M5) performance on ARM. The dense
  LM hardened in M2 is the foundation; M5 adds only the block-sparse
  Schur structure, not new numerics.
- **M3 caveat:** verified only against synthetic scenes and the
  degenerate mock-camera scene; real-scene init quality is unproven
  until the deferred frame recorder / a public dataset is run on-Pi.
