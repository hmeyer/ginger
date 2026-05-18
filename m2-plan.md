# M2 — Calibration + Pinhole Model + Replay Harness (detailed plan)

Detailed plan for the M2 milestone from [`PLAN.md`](../PLAN.md). M2 is
foundation only — no SLAM behaviour ships, but every M3+ step depends on
it. Exit criteria are at the bottom.

## Decisions locked

- **Camera prior: Pi Camera "rev 1.3" standard lens.** We proceed with
  the standard-lens OV5647 module prior. Derived nominal pinhole, FOV-based
  so it is resolution-agnostic for the libcamera `ViewFinder` full-FOV
  mode (`capture.rs:192`, resolution read at runtime `capture.rs:200`):

  ```
  HFOV ≈ 53.5°   VFOV ≈ 41.4°   fixed focus   distortion = 0 (initial)
  fx = (W/2)/tan(HFOV/2)   fy = (H/2)/tan(VFOV/2)   cx = W/2   cy = H/2
  → at 800×600: fx ≈ fy ≈ 794, cx = 400, cy = 300
  ```

  This is an **explicitly-flagged prior, not a calibration.** It unblocks
  M3–M5 against the replay harness and is swapped out without touching
  callers (`CameraModel` is the only intrinsics owner).

- **Math backbone: pure-Rust, no BLAS/LAPACK/Eigen.** Add `nalgebra`
  (pure Rust, aarch64-clean) for small fixed-size matrices + SVD/QR;
  hand-roll a tiny `lie` module (SO3/SE3 `exp`/`log`). Keeps the clean
  aarch64 cross-compile that `crates/fast` exists to protect. OpenBLAS on
  a Pi is large, slow to build, dynamic-dim, and **not** faster than
  well-laid-out small fixed blocks at our problem sizes.

- **Proper calibration is deferred** (no physical target yet). Tracked as
  a follow-up below and in `PLAN.md`. (Note: the standalone `TODO.md`
  asked for in an interrupted message is folded in here instead — say the
  word if you still want a separate file.)

## Hardware-acceleration strategy (RPi 4 / Cortex-A72)

The Pi 4 is a quad-core Cortex-A72, **ARMv8.0-A**. Implications that
shape the math design:

- **NEON is the only SIMD.** 128-bit Advanced SIMD (4×f32 or 2×f64).
  A72 is ARMv8.0 → **no FEAT_DotProd (SDOT/UDOT), no i8mm, no FP16
  arithmetic, no SVE.** Do not design kernels around instructions the
  chip lacks. Hamming popcount stays the `vcntq_u8` + `vaddvq` path
  already used.
- **The VideoCore VI GPU is a dead end for SLAM math.** No usable
  GPGPU stack for sparse/irregular BA; Vulkan-compute ROI is terrible.
  Excluded.
- **Multicore is the biggest single lever**, far more than micro-SIMD.
  ORB-SLAM's 3-thread split (tracking / local mapping / loop closing)
  maps onto the 4 A72 cores: tracking must hold frame rate while heavy
  BA runs off-core at lower cadence. `rayon` is already the data-parallel
  tool in-tree (the Hamming matrix uses it).
- **A72 is memory-bandwidth-bound** (~4–5 GB/s, 1 MB shared L2). For the
  sparse parts, **data layout beats intrinsics**: SoA, contiguous
  observation arrays, block-contiguous Hessian.

Concrete rules for M2 math code:

1. **Reuse the `crates/fast` discipline verbatim.** Portable scalar
   reference (correctness oracle + non-aarch64 build) → `#[cfg(target_arch
   = "aarch64")]` NEON kernel → **differential parity test** (NEON
   bit/eps-identical to scalar) → A/B via an extended `slam_bench`. NEON
   is baseline on aarch64; no runtime feature gate.
2. **Let the compiler vectorize first; hand-write NEON only at a
   measured hotspot.** With `target-cpu=cortex-a72` + `lto=thin` +
   `codegen-units=1` (already set), LLVM auto-vectorizes tight
   fixed-size f32 loops well. Hand-NEON only what the bench proves hot.
3. **f32 for per-observation residual/Jacobian; f64 for accumulators.**
   2× NEON throughput and half the memory traffic on the hot path;
   f64 `JᵀJ` / `Jᵀr` accumulation and global BA where conditioning bites.
4. **Block-sparse, small fixed dense blocks — not a generic sparse
   lib.** BA structure is 6×6 camera blocks, 3×3 point blocks, 6×3
   off-diagonal. The Schur-complement inner kernels (6×6 Cholesky, 3×3
   inverse, 6×3 GEMM) are L1-resident and exactly what NEON +
   auto-vectorization does well. nalgebra supplies the small fixed
   blocks + Lie ops; the block-sparse Schur is hand-written; NEON-tune
   only the inner block kernels, measured.
5. **Dense Gauss-Newton/LM now; sparse Schur deferred to M5.**
   Motion-only BA (M4) is 6-DOF dense — a small dense LM is enough and
   is the right first solver to harden.

## WebUI-visible outputs

M2 is foundation — by design it ships **no SLAM behaviour**, so its
user-facing output is diagnostics, not a demo. What we surface now (cheap,
and it makes M2 progress + risk visible instead of buried in logs), and
what later milestones draw into the same surface:

- **M2 — calibration-status HUD line.** Extend `SlamSnapshot` /
  `/api/slam/stream` with the active intrinsics (`fx fy cx cy`, derived
  FOV) and a `verified` flag; `index.html` `#slam-hud` renders a loud
  `PRIOR · rev 1.3 · UNVERIFIED` badge until a real calibration lands.
- **M2 — sensor-mode confirmation.** Surface the one-time libcamera
  `Model` / `PixelArraySize` and a `full-FOV` vs `crop?` flag, turning
  the biggest M2 risk into something visible.
- **M2 — `/api/slam/map` + 2D top-down canvas, as a stub.** Add the
  transport and an empty toggleable top-down canvas now so M3 has
  somewhere to draw immediately instead of inventing transport
  mid-milestone. Honest cost: a little throwaway-ish UI in M2.
- **M2 — undistortion grid overlay (optional).** A thin warped grid from
  `CameraModel`; a no-op while `k = 0`, so it proves plumbing only.
- **M3+** draw into the M2 canvas: M3 initial point cloud + 2 poses;
  M4 live trajectory + current pose; M5 growing map + keyframes; M6 a
  "loop detected" event with the trajectory snapping straighter.

Existing surface this extends (no new transport invented): `/api/slam/stream`
SSE → `index.html` video overlay + `#slam-hud`.

## Work breakdown

### A. Crate + math scaffold
- New dependency-light crate `crates/slam-core` (no camera/libcamera),
  mirroring `crates/fast` so geometry cross-compiles and unit-tests fast.
  Deps: `nalgebra`, `rayon`.
- `lie`: SO3/SE3 `exp`, `log`, adjoint, compose, inverse. Unit tests:
  `exp∘log` round-trip, small-angle limit, Jacobian numeric check.
- `optimize`: generic dense Gauss-Newton/LM (Huber robustifier),
  trait-based residual/Jacobian. Scalar reference + NEON inner-product
  kernel behind the parity-test pattern.

### B. `CameraModel` (in `slam-core`)
- Pinhole + radial-tangential (k1,k2,p1,p2,k3). `project` (cam→px),
  `unproject` (px→bearing), `undistort_point`, batch keypoint undistort.
- Tests: `project∘unproject` round-trip, known-point projection, zero-
  distortion identity.

### C. Intrinsics config + runtime wiring
- `slam.toml`: `fx,fy,cx,cy,k1,k2,p1,p2,k3,width,height,model,verified`.
  Loader; when absent, derive the rev 1.3 FOV prior from the **live
  stream W/H** and set `verified = false`.
- Log libcamera `Model` / `PixelArraySize` / `UnitCellSize` once at
  startup to pin sensor + pixel pitch from hardware, and a one-time
  warning while `verified = false`.
- Wire `slam::run` to build the `CameraModel` from actual resolution.

### D. Replay harness
- **Recorder:** env-gated dump of live frames (raw YUYV + timestamp).
- **Replay runner:** generalize `examples/slam_bench.rs` to read an
  ordered frame directory deterministically; accept either a recorded Pi
  clip or a standard dataset (TUM/EuRoC/KITTI mono) with its own
  intrinsics TOML. This is what makes M3+ correctness regression-testable
  headless / in CI.
- Commit only a tiny synthetic clip; document fetching a public set
  (sequences are large; keep them gitignored).

### E. Bench + CI
- Extend `slam_bench` with geometry/solver stages so M2 math has the
  same measured A/B loop as the frontend.
- CI keeps `cargo check -p slam-core --target aarch64-unknown-linux-gnu`
  (cross-compile guard) + parity tests.

### F. WebUI status surface
- Extend `SlamSnapshot` with `intrinsics { fx, fy, cx, cy, fov_deg,
  verified }` and an optional sensor-mode string; serialise over the
  existing `/api/slam/stream`.
- `index.html` `#slam-hud`: calibration line + `UNVERIFIED` badge.
- `/api/slam/map` endpoint + a toggleable empty top-down canvas
  (stub for M3+).

## Sequencing

A → (B, C parallel) → D → E. E's recorder feeds the first real Pi clip;
the FOV prior is the placeholder until calibration lands.

## Risks

- **ViewFinder might be a center-crop, not a full-FOV downscale.** If so,
  FOV-derivation is wrong and fx/fy must use the full-res pixel focal.
  Mitigated by the one-time sensor-mode log (task C); verify before
  trusting M3 geometry.
- **Wrong lens variant.** Prior assumes the standard ~54° lens; a
  wide-angle clone breaks it 2–3×. `verified=false` + startup warning
  keeps this loud until calibrated.
- **Local-BA performance on A72 (M5).** De-risked early: harden the
  dense LM and the small-block NEON kernels in M2 so M5 only adds the
  sparse Schur structure, not new numerics.

## Deferred follow-ups

- **Proper camera calibration** (blocked on a physical target). Offline
  `tools/calibrate.py` (OpenCV ChArUco) consuming recorded frames →
  emits `slam.toml` with `verified = true`. Not kalibr (overkill: ROS,
  AprilGrid, built for cam-IMU/multi-cam). Mirrored as the open M2 item
  in `PLAN.md`.

## Exit criteria

- `slam-core` cross-compiles to aarch64; parity tests green.
- `CameraModel` round-trip tests pass; built from live resolution with
  the flagged rev 1.3 prior.
- Replay runner reproduces a deterministic detect→match over a frame
  directory (Pi clip and one public dataset).
- `slam_bench` reports geometry/solver stages.
- No SLAM behaviour yet — that's M3.
