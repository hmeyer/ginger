# ORB-SLAM Roadmap

Plan for evolving the visual SLAM frontend into a full monocular
ORB-SLAM pipeline on the Raspberry Pi 4. The inline roadmap stub lives in
`src/slam/mod.rs`; this document is the detailed version.

## Where the code is today

Monocular SLAM runs end-to-end: detect → two-view bootstrap → **live
6-DoF tracking** (constant-velocity prediction + motion-only BA) →
**local mapping** (keyframes + decoupled triangulation + block-sparse
Schur local BA on a background thread), surfaced as a growing,
locally-bundle-adjusted point cloud + keyframes + camera trajectory in
the WebUI. No loop closing / relocalization yet (M6).

- **M0** ✅ FAST-9 over an 8-level pyramid + NMS + grid-spread cap (NEON).
- **M1** ✅ Oriented BRIEF + brute-force mutual-NN matching (Lowe ratio).
- **M2** ✅ Calibration prior + pinhole model + math backbone + mock
  camera + headless replay/CI.
- **M3** ✅ Two-view initialization (essential + homography) wired into
  the live pipeline with a top-down map in the WebUI.
- **M4** ✅ Tracking thread — constant-velocity model + motion-only BA
  (Huber) re-finding map points each frame; live trajectory.
- **M5** ✅ Local mapping — M5-0 `Stage` enum; M5-1a
  map/covisibility; M5-1b block-sparse Schur local BA; M5-1c gated
  triangulation; **M5-2 frontend wiring** (keyframe insertion +
  decoupled `LocalMapper` thread: triangulation + local BA over the
  covisibility window).
- Testing stabilized: the pipeline state machine is a testable
  `Frontend::on_frame` seam covered by headless `pipeline_tests`;
  `slam_bench` / `slam_replay` harnesses; decoupled frontend thread.

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

Foundation only; no SLAM behaviour.

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

### M4 — Tracking thread (live 6-DoF pose) ✅

- Done in `ginger_slam_core::tracking`: constant-velocity prediction +
  motion-only BA (SE3 via `lie`, Huber on the hardened LM); behind-
  camera points get a clamped-depth penalty.
- Wired into `Frontend`: descriptor-match the map into each frame →
  `Observation`s → predict → `track_pose` → append to the trajectory;
  weak/failed solves report "tracking lost" without corrupting the map
  (relocalization is M6).
- **Deliverable ✅:** live camera-centre trajectory on the WebUI canvas.
- Caveat: guided/windowed projection matching is still brute-force
  descriptor matching against the whole map (refine in M5).

### M5 — Local Mapping thread ✅

- **M5-0 ✅** explicit `Stage` enum (`Bootstrapping` / `Tracking`),
  replacing the implicit `Option` trio.
- **M5-1a ✅** `map`: keyframes / map points / covisibility graph +
  spanning tree / culling / `needs_keyframe` insertion policy /
  `local_window` selector (+ `add_point_observed`: link a
  late-discovered point into its keyframe once per side, so local-BA
  observations and covisibility weights stay exact).
- **M5-1b ✅** `local_ba`: block-sparse Schur local bundle adjustment
  over a covisibility window (the Pi 4 performance crux).
- **M5-1c ✅** `triangulation`: gated two-view new-point triangulation
  (cheirality / parallax / symmetric reprojection).
- **M5-2 ✅** wiring (`slam/mapper.rs`): bootstrap promotes the two
  views to keyframes; tracking inserts keyframes via `needs_keyframe`
  (tracked inliers as observations) and hands them + their raw features
  to a **decoupled `LocalMapper` thread** that triangulates new points
  against covisible keyframes and runs `local_bundle_adjust` over
  `local_window` at a short per-keyframe iteration budget (heavy BA off
  the tracking core, lower cadence). The work unit
  (`LocalMapper::process_pending`) is the headless-tested seam — `run`
  drives it from a thread, `pipeline_tests` synchronously — so the
  tested path equals production. Coarse single map mutex; a finer
  tracking/mapping handoff is an M6 refinement.
- **Visible deliverable ✅:** growing, locally-bundle-adjusted point
  cloud + keyframe markers on the top-down canvas.

### M6 — Relocalization + loop closing 🔄

slam-core retrieval primitive started; the rest is built on it.

- **M6-1a ✅** `bow`: binary DBoW2-style visual vocabulary
  (hierarchical Hamming k-means tree, trained offline + deterministic),
  TF-IDF L1-normalized image vectors, and an inverted-index `Database`
  for `query`-by-place. Camera-free, Pi-cheap (bitwise, no BLAS).
  Deferred to M6-2: direct index (word→features for guided matching)
  and on-disk vocabulary (de)serialization.
- **M6-1b ✅** `pnp`: Grunert P3P (quartic via in-code polynomial
  elimination + companion-matrix roots) + RANSAC, polished by the
  tested motion-only BA; reuses `tracking::Observation`. The recovery
  solver for relocalization / loop verification.
- **M6-1c ⏭** Sim3 `lie` exp/log + Sim3 alignment + Essential-graph
  pose-graph optimization (monocular scale drift); optional global BA.
- **M6-2 ⏭** frontend wiring: relocalize on track loss (BoW candidates
  → guided match → PnP-RANSAC); per-keyframe BoW added on insertion;
  loop detection (query DB minus covisible/recent) → Sim3 verify →
  pose-graph correction on the local-mapper thread; extend
  `pipeline_tests`.
- **Visible deliverable:** "relocalized" / "loop detected" event +
  trajectory snapping straighter on the canvas.

**WebUI surface:** all milestones draw into the single top-down canvas
(`/api/slam/map`, `map` overlay mode) + the `#slam-hud` line; M3 filled
it with the bootstrap cloud + two cameras, M4 extended it with the live
trajectory, M5 grows the locally-bundle-adjusted map + draws keyframe
markers (orange squares).

## Sequencing & risks

- **Strict dependency chain:** M2 ✅ → M3 ✅ → M4 ✅ → M5 ✅ →
  **M6 (in progress: M6-1a BoW ✅, M6-1b PnP ✅)**.
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

## Performance strategy (RPi 4 / Cortex-A72)

Durable engineering guidance for M4–M6 kernels (the Pi 4 is a quad-core
**ARMv8.0-A** Cortex-A72):

- **NEON is the only SIMD** (128-bit; 4×f32 / 2×f64). A72 is ARMv8.0 —
  no SDOT/i8mm/FP16/SVE; don't design for instructions it lacks. Hamming
  stays `vcntq_u8` + `vaddvq`. The VideoCore GPU is a dead end for
  sparse SLAM math (excluded).
- **Multicore is the biggest lever**, not micro-SIMD: ORB-SLAM's
  tracking / local-mapping / loop-closing split maps onto the 4 cores —
  tracking holds frame rate while heavy BA runs off-core at lower
  cadence. `rayon` is the in-tree data-parallel tool.
- **Reuse the `crates/fast` discipline:** portable scalar reference →
  `#[cfg(target_arch="aarch64")]` NEON kernel → differential parity
  test → `slam_bench` A/B. Let the compiler vectorize first
  (`target-cpu=cortex-a72` is set); hand-write NEON only at a measured
  hotspot.
- **f32 per-observation residual/Jacobian, f64 accumulators.**
- **Block-sparse with small fixed dense blocks** (6×6 / 3×3 / 6×3),
  not a generic sparse lib: the Schur inner kernels are L1-resident and
  auto-vectorize well. nalgebra for the blocks; the block-sparse Schur
  (M5) is hand-written — it's a layout change, not new numerics.

## Verification

The gating correctness signal is **fast, deterministic, headless
`cargo test`** (CI `Headless` + `aarch64-check`), with three tiers:

- **slam-core unit tests** — geometry/optimization math on synthetic
  scenes with ground truth (`lie`, `camera`, `optimize`, `twoview`,
  `tracking`).
- **Pipeline integration tests** (`src/slam`, `pipeline_tests`) — drive
  the full `Frontend` state machine (anchor → two-view init → tracking →
  trajectory) via synthetic features projected from a known 3D scene +
  camera path, **bypassing image detection** (separately tested). The
  `slam::run` loop is a thin camera/server wrapper around the tested
  `Frontend::on_frame` seam.
- **`slam_replay`** — deterministic detect→match over a PGM sequence.

A live-server HTTP smoke is **on-Pi / non-gating only** (the sandbox
SIGKILLs long-running spawned servers; it adds nothing the headless
tests don't already gate).
