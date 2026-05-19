# ORB-SLAM Roadmap

Plan for evolving the visual SLAM frontend into a full monocular
ORB-SLAM pipeline on the Raspberry Pi 4. The inline roadmap stub lives in
`src/slam/mod.rs`; this document is the detailed version.

## Where the code is today

Monocular SLAM runs end-to-end: detect → two-view bootstrap → **live
6-DoF tracking** (constant-velocity prediction + motion-only BA) →
**local mapping** (keyframes + decoupled triangulation + block-sparse
Schur local BA on a background thread) → **relocalization** (BoW +
PnP-RANSAC on track loss) + **loop closing** (BoW detect → Sim3 verify
→ Essential-graph pose-graph correction), surfaced as a growing,
locally-bundle-adjusted point cloud + keyframes + camera trajectory in
the WebUI that snaps straighter on loop closure. The full monocular
ORB-SLAM pipeline (M2→M6) is in place.

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

### M6 — Relocalization + loop closing ✅

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
- **M6-1c ✅** `sim3`: Sim(3) group (`exp`/`log` Strasdat closed form,
  inverse/compose/action), `sim3_align` (closed-form Umeyama similarity
  with scale), and `optimize_pose_graph` (LM over relative-Sim3
  residuals, gauge-fixed) — closes a detected loop + absorbs monocular
  scale drift.
**M6-2 ⏭ — frontend wiring (the user-visible milestone).** The three
slam-core primitives (bow / pnp / sim3) are done; M6-2 integrates them
into the live `Frontend` + decoupled `LocalMapper` so a lost track
recovers and a revisited place straightens the map. Broken into five
testable sub-steps:

- **M6-2a ✅ — `bow` deferrals: serialization + direct index.**
  `Vocabulary::to_bytes`/`from_bytes` (compact self-describing binary,
  `"GBOW"` magic + version header over a `postcard` payload + structural
  validation) and
  `transform_indexed` (per-feature leaf word alongside the `BowVector`,
  for guided post-BoW matching). Also hardened `transform` to be
  *bitwise* deterministic (sort words before the L1 sum — the `tf`
  HashMap iterates in per-instance-random order). Tests: round-trip,
  garbage/version/truncation rejection, direct-index consistency,
  transform bit-stability. *(Note: PR #34 merged only the PLAN doc, not
  the `bow.rs` code — it is re-landed with M6-2b so it isn't lost.)*
- **M6-2b ✅ — vocabulary source + place-recognition database**
  (`slam/place.rs`). A `PlaceDb` mirroring the `slam.toml` pattern: load
  a shipped `slam_vocab.bin` (`Vocabulary::from_bytes`) if present, else
  **deterministically self-train once** (`Vocabulary::build`) from the
  pooled descriptors of the first `N_VOCAB_KF` keyframes — no external
  asset for headless/CI. Shared `Arc<Mutex<PlaceDb>>` reachable by the
  tracking thread (relocalize, M6-2c) and the `LocalMapper` (loop
  detect, M6-2d); the mapper registers each ingested keyframe's BoW
  (kf-ordered, back-filled once the vocab self-trains), `entry_kf` maps
  a DB entry back to its keyframe. Pipeline test: a long sweep
  self-trains, indexes every keyframe, and a query resolves an earlier
  place to an earlier keyframe (deterministic).
- **M6-2c ✅ — relocalization on track loss.** Added `Stage::Lost {
  since, track }`: a lost track (weak/failed solve, or too few map
  matches) saves the trajectory/map context and transitions to `Lost`
  instead of dead-ending. Each lost frame BoW-queries the place DB →
  collects candidate keyframes' (+ covisible) observed **map points**
  (3D pos + descriptor from `map`) → brute-force match (candidate-
  restricted) → `pnp::pnp_ransac` (bounded iters, conservative
  `RELOC_MIN_INLIERS`) → on success append the recovered pose and
  resume `Tracking`; else stay `Lost` (map + trajectory frozen — no
  corruption). Runs on the tracking thread (bounded). Pipeline test:
  garbage frames lose the track without corruption, a recognizable
  frame relocalizes, tracking continues.
- **M6-2d ✅ — loop closing on the `LocalMapper` thread.** Added
  `sim3::sim3_ransac` (robust Sim3 over 3D↔3D matches, RANSAC + refit;
  slam-core, outlier-tested). After local BA the just-processed
  keyframe BoW-queries the DB **excluding itself / covisible / recent**
  (`LOOP_MIN_GAP`); a candidate over `LOOP_MIN_SCORE` is geometrically
  verified by matching pooled map-point descriptors (kf vs candidate +
  covisible) and `sim3_ransac` (≥ `LOOP_MIN_INLIERS`). On acceptance
  the keyframe-graph poses are snapshotted as `Sim3`, the Essential
  graph is assembled (spanning-tree + covisibility edges measured at
  the current estimate as soft rigidity, plus the high-weight loop edge
  `meas = Sₖ·S⁻¹·S_c⁻¹` that has a non-zero residual pulling the graph),
  `optimize_pose_graph` runs off-lock (origin gauge-fixed), then
  corrected keyframe poses (Sim3→SE3, scale folded into translation) +
  map points (dragged by their reference keyframe's Sim3 correction)
  are written back. Loop count is shared with the frontend (HUD status
  `· loop closed (#n)`). Stays the pumped-synchronously tested seam,
  heavy + rare + off the tracking core. Conservative gates favour a
  miss over a map-destroying false closure.
  - *Caveat:* closure **efficacy** is gated by the slam-core unit tests
    (`optimize_pose_graph` drifted-loop + `sim3_ransac` outliers); the
    synthetic pipeline harness (frame-stable per-landmark descriptors +
    whole-map matching) shares points across a revisit so it can't
    manufacture the drift a closing loop needs — like the M3 synthetic
    caveat, real-scene validation rides the deferred frame-recorder.
    The pipeline test asserts the conservative gates don't misfire on
    straight motion + the path is deterministic + non-corrupting.
- **M6-2e ✅ — surface + tests (consolidation).** The events already
  reach the surface incrementally: the `MapSnapshot` status carries
  `relocalizing… (lost n)` / `relocalized: …` (M6-2c) and
  `· loop closed (#n)` (M6-2d), shown untruncated on the `#slam-hud`
  line; the existing canvas (trajectory + keyframes + points) snaps for
  free when pose-graph moves the poses; the in-canvas caption was
  widened so the event is visible there too. `pipeline_tests` already
  cover (1) track loss → relocalize without map corruption
  (`relocalizes_after_track_loss`, M6-2c) and (2) the loop-closure path
  is gated/deterministic/non-corrupting on straight motion
  (`loop_closing_gated_and_no_false_positive`, M6-2d); closure
  *efficacy* on a genuinely drifted loop is gated by the slam-core unit
  tests (`sim3::optimize_pose_graph` + `sim3_ransac`), the synthetic
  harness limitation documented in M6-2d. README / inline roadmap
  updated.

- **Visible deliverable ✅:** a `relocalized` / `loop closed (#n)`
  event on the `#slam-hud` line and the trajectory + keyframes + point
  cloud snapping straighter on the top-down canvas when a loop closes.

**M6-2 design decisions / risks (durable):**
- *Self-trained vocabulary* is weaker than a pre-trained ORB vocab but
  needs no shipped asset and keeps CI headless + deterministic; the
  `slam_vocab.bin` loader lets a real vocabulary drop in later (same
  opt-in shape as `slam.toml`). Documented caveat, not a blocker.
- *False loops* (perceptual aliasing) corrupt the map irreversibly, so a
  loop fires only on BoW-score gate **and** geometric-inlier gate **and**
  non-covisible/non-recent candidate; favour misses over false closures.
- *Map-point consistency:* after pose-graph correction, points are moved
  by the `Sim3` correction of a chosen observing keyframe (not
  re-triangulated) — cheap and good enough for the visible deliverable;
  a post-loop global BA is a deferred refinement.
- *Concurrency:* relocalization runs on the tracking thread (bounded);
  loop closing + pose-graph run on the `LocalMapper` thread under the
  same coarse world mutex as M5 — rare, off the frame-rate path. A
  finer handoff and post-loop global BA are explicit later steps.

**WebUI surface:** all milestones draw into the single top-down canvas
(`/api/slam/map`, `map` overlay mode) + the `#slam-hud` line; M3 filled
it with the bootstrap cloud + two cameras, M4 extended it with the live
trajectory, M5 grows the locally-bundle-adjusted map + draws keyframe
markers (orange squares), M6 surfaces `relocalizing…` / `relocalized` /
`loop closed (#n)` in the HUD and snaps the canvas on loop closure.

## Sequencing & risks

- **Strict dependency chain:** M2 ✅ → M3 ✅ → M4 ✅ → M5 ✅ → M6 ✅
  (1a/b/c slam-core + 2a–2e wiring all done). The monocular ORB-SLAM
  roadmap is complete; remaining items are the deferred hardware-loop
  follow-ups (real ChArUco calibration, frame recorder) + perf passes
  (guided-index matching, post-loop global BA, finer mapping handoff,
  measured NEON hotspots).
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
