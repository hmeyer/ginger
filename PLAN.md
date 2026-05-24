# PLAN: drive + iterate the SLAM tracking-LOST loop

## Status

Tracking was getting "almost immediately LOST" while driving. Live
journal showed why:

```
21:50:16  slam: relocalization stuck for 151 frames with BoW ready (117 kf, 7055 pts)
  — discarding, re-bootstrapping
21:50:16  slam: two-view init OK — 121 pts, model=essential, R_H=0.39, matches=158
21:50:16  thread 'slam-mapper' panicked at src/slam/mapper.rs:425:66:
          index out of bounds: the len is 115 but the index is 117
21:50:16  thread 'slam'        panicked at src/slam/frontend.rs:567:39:
          called `Result::unwrap()` on an `Err` value: PoisonError { .. }
```

A loop-closure pose-graph optimization was running off-lock when the
frontend's "relocalization gave up" path called `reset_world()`. The
new bootstrap inserted keyframes with ids ≥ `n`; the writeback indexed
`poses[id]` and panicked, poisoning the world mutex and taking the
tracking thread with it. `/api/slam/map` then returned the **last
snapshot before the crash**, which made tracking *look* fine in the
HUD while in fact both SLAM threads were dead — hence "immediately
LOST" the moment the user drove past the (frozen) mapped area.

**The panic is now fixed and shipped** (commit on `main`):

* `apply_loop_correction` extracted as a testable helper.
* Bails if the gauge-fix `origin` is no longer alive (`reset_world()`
  ran underneath the optimization).
* Uses `poses.get(id)` so keyframes inserted after the snapshot are
  skipped, not OOB-indexed.
* Three regression tests cover happy-path, mid-flight insert, and
  reset-under-us.

What we **don't yet know** is what the *real* tracking-loss behaviour
looks like once the threads stay alive. The HUD reported "tracking:
37/39 inliers" with `n_lost: 5` and `last_lost_reason: "weak 0/64
inliers"` — that "0/64" pattern (lots of matches, zero pass the
reprojection gate) is a smell: the const-velocity prediction is wildly
wrong, or the matches are mostly wrong correspondences, or the gate is
too tight for this descriptor noise. But that diagnosis came from a
frozen snapshot, so it might or might not be representative.

## Progress log

**2026-05-24 — session 1 (Claude + Ginger)**

* **Step 1 — fix is live and threads stay up: DONE.** Verified the
  panic-fix binary by polling `/api/slam/map` at ~3 Hz across two
  drive sessions (156 + 606 ticks, **0 frozen snapshots**). Service
  PID stayed put; no panic strings. Bootstrap succeeded during the
  test (kf 0 → 383 across the run), inlier ratios 60–95 %.
* **Step 2 — drive-in-pulses cadence: PARTIALLY DONE.** Did short
  forward / back / turn pulses in the middle of the living room
  (drying-rack + basket ~1 m on either side). `n_lost` stayed at 2
  the entire time — *no new LOST events triggered*. Real translation
  needed `duty≈1500` (≈0.7 s pulses ≈40 cm); 35-raw didn't move the
  wheels at all. That gotcha is now documented in `CLAUDE.md` and the
  step-2 wording above.
* **Step 3 — diagnose first reproducible LOST: BLOCKED.** No LOST
  reproduced in this session. The "0/64 weak inliers" pattern from
  the original journal trace looks more and more like the
  frozen-snapshot artifact: when the threads are actually alive and
  the robot moves at a sane cadence, tracking is healthy. The next
  session needs *deliberately adversarial* driving — see *Next* below.

## Progress log (cont.)

**2026-05-24 — session 2 (Claude + Ginger)**

* **First reproducible LOST: fast in-place spin.** Sequence:
  forward 1 s @ `duty=1800` then *immediately* spin `±1800` for 1.2 s
  (no stop between). Tracked healthily through the forward
  (60–71 inliers, kf 4 → 26) and into the spin (30/48, kf 27 with
  +41 newly triangulated pts), then collapsed in ~14 frames:
  *only 8 map matches → soft-lost → 3 soft-lost frames → Stage::Lost*.
  Lost for 148 frames (~5 s) with the **map preserved** at 29 kf /
  196 pts — BoW-ready give-up budget. `RELOC_MAX_FRAMES_BOW` fired
  at frame 148; `reset_world` + re-bootstrap with zero panic. The
  panic fix is solid; the failure is upstream.
* **Slow spin survives.** Same starting state, spin `±600` for 3 s,
  ended at 15 kf / 187 pts, still `tracking=true`. So the failure
  mode is *rate*-driven, not a structural map issue.
* **Diagnosis (mono-SLAM intrinsic).** Two compounding effects:
  1. *Mapper can't keep up at high angular velocity.* `needs_keyframe`
     fires every couple of frames during a fast pan, but each kf's
     triangulation against its covisible neighbours takes finite time
     (`LOCAL_BA_K=6` window, `LOCAL_BA_ITERS=5`).
  2. *Pure rotation has zero parallax*, so even when the mapper does
     pick up a fast-spin kf, it can't triangulate fresh points against
     the just-previous (also-rotation-only) kf. New geometry only
     appears for pairs that bracket some translation, which a pure
     spin doesn't provide.
  Net effect: the camera sweeps into unmapped scenery faster than map
  points can be created there. Tracking starves on "only N map matches".
  This is a generic mono-SLAM limitation (ORB-SLAM3 et al. also need
  IMU or translation-during-rotation to bridge fast pans).
* **Two-tier exploration limit observed empirically (this room):**
  * Safe / mapping-friendly: differential ≤ ~1200 (e.g. `±600`
    in-place spin, or curved forward turns).
  * Tracking-breaking: differential ≥ ~3000 (e.g. `±1500..±2000`
    in-place spins).
  Threshold not narrowed further this session; ~2000 differential
  is the next data point to gather.
* **Exploration session at "safe" rates also failed — cumulative
  rotation matters.** Three back-to-back gentle right arcs
  (`left=900, right=300`, diff=600, ~0.8 s each) totalled ~90° of
  rotation away from the mapped sector and lost tracking with
  `weak 1/27 inliers`. Per-frame rate was safe; the *total* angular
  drift past the mapped scenery was the killer. So a simple
  supervisor clamp on instantaneous `|left-right|` would *not* have
  helped this case — the failure is "rotated past what's been
  mapped", not "rotated too fast in one moment".
* **Relocalization-on-return works exactly as advertised.** Earlier
  in the same session, a near-collision triggered Lost for 113
  frames; the moment I backed up into the mapped scenery, BoW reloc
  brought us back with `tracking: 94/198 inliers` and `n_lost`
  intact. The recovery path is solid; *avoiding* the loss is what
  needs help.
* **Decision:** the right next move is to add an IMU (BMI160 lying
  around in the user's bin). See *Next* below — pure software fixes
  for the rotation problem all have heavy cost or partial coverage.

## Next: integrate a BMI160 IMU

The session-2 failures are *generic mono-SLAM limits* — pure rotation
has no triangulation parallax, and a const-velocity predict in SE(3)
can't model motion onsets or direction changes well. An IMU is the
standard fix: gyro pre-integration between camera frames gives an
accurate rotation predict regardless of what the map looks like, and
accelerometer + gravity eventually gives metric scale. We have a
BMI160 (6-DOF: 3-axis gyro + 3-axis accel, I²C, 3.3 V) and the Pi's
I²C bus already drives the PCA9685 PWM at `0x40` and the ADS7830 ADC
at `0x48`, so the wiring is incremental.

### Hardware

Pin out (Pi 4, BCM numbering — same bus the existing devices use):

| Pi pin (BCM) | Pi pin (phys) | BMI160 pin | Notes |
| --- | --- | --- | --- |
| 3.3 V | 1 | VCC | **Not 5 V**; BMI160 is 3.3 V. |
| GND | 9 | GND | Any ground pin. |
| GPIO 2 (SDA1) | 3 | SDA | Shared with PCA9685, ADS7830. Pull-ups already on the bus. |
| GPIO 3 (SCL1) | 5 | SCL | Same. |

I²C address: **0x68** if `SDO`/`AD0` is tied to GND (or floating —
internal pull-down on most breakouts), **0x69** if tied to VCC.
Confirm with `i2cdetect -y 1` after wiring — neither `0x68` nor
`0x69` is in use today (only `0x40` and `0x48` show up).

Mounting: rigid attachment to the chassis is critical. The
**IMU-camera relative orientation must not change at runtime** — any
vibration or play injects rotational noise straight into the
tracking-predict. Ideally mount close to the camera and aligned so
the IMU axes are nominally parallel to the camera axes (saves work
on the extrinsic calibration). A small zip-tied perfboard on the
camera mount itself is fine; a flopping wire is not.

### Software stages

Each stage is a separately-mergeable commit / PR. Each adds tests.
The headless DoD (`cargo test --workspace --no-default-features`)
must stay green at every stage — gate the IMU behind the existing
`libcamera` feature or a new `imu` feature, whichever is cleaner;
**do not make non-IMU code paths require the hardware**.

#### Stage 0 — post-solder verification (no Rust yet)

After you finish soldering:

```bash
# Bus is detected, BMI160 shows up at 0x68 or 0x69
i2cdetect -y 1

# Read CHIP_ID (register 0x00) — must be 0xD1 for BMI160
i2cget -y 1 0x68 0x00      # or 0x69 if SDO is tied high
```

If `i2cget` returns `0xd1`, the chip is alive and talking. If it
returns `0x00` or `0xff`, recheck wiring (most often a swapped SDA/
SCL or a 5 V/3.3 V mistake). Don't move to Stage 1 until this works.

#### Stage 1 — HAL driver (`src/hal/bmi160.rs`)

Follow the same shape as `src/hal/pca9685.rs` and `src/hal/adc.rs`:
a `pub struct Bmi160 { i2c: I2c }` with `new(address: u16) -> Result<Self>`
that powers up the gyro (`CMD = 0x15` for "PMU gyro normal") and
accelerometer (`CMD = 0x11` for "PMU accel normal"), waits the
chip's wakeup time (`~80 ms` for gyro, `~3.8 ms` for accel — check
datasheet table 8), and verifies CHIP_ID == 0xD1.

Public API to expose:

* `read_gyro_raw(&mut self) -> Result<[i16; 3]>` — `DATA_8` ..
  `DATA_13`, little-endian.
* `read_accel_raw(&mut self) -> Result<[i16; 3]>` — `DATA_14` ..
  `DATA_19`.
* `read_both(&mut self) -> Result<([i16; 3], [i16; 3])>` — single
  burst read of the 12 data bytes, cheapest and atomic (no
  inter-axis jitter).
* `read_sensortime(&mut self) -> Result<u32>` — chip's 24-bit
  internal time at 39.0625 µs/tick; cheap way to detect dropped
  samples.
* Configurable range/ODR; sensible defaults: gyro `±500°/s` (range
  reg `GYR_RANGE = 0x02`), gyro ODR `200 Hz` (`GYR_CONF = 0x09`),
  accel `±4 g`, accel ODR `200 Hz`. 200 Hz is plenty given the
  camera runs at ~30 Hz; we'll average 6–7 IMU samples per camera
  frame for the pre-integration.

Tests: mock the I²C bus (a `trait I2cBus` over `rppal::i2c::I2c`)
so the driver is unit-testable without hardware — same trick the
existing HAL would use if it had unit tests today (it mostly
doesn't — fine to introduce just for this one).

#### Stage 2 — sample loop + 1 Hz log + new HTTP endpoint

Run a dedicated thread that loops on `read_both` at ~200 Hz, writes
into a lock-free single-producer ring buffer keyed by `Instant`,
and `info!`-logs the latest rate / latest sample once a second.
Expose `GET /api/imu/sample` returning the latest `(gyro_dps,
accel_mps2, t_mono_ns)` as JSON so we can verify the chip is
sensible by hand (`curl /api/imu/sample` while shaking the chassis).

This is the "ship a useful new endpoint and stop" gate. Don't touch
SLAM yet. Confirm:

* Gyro reads ≈ `[0, 0, 0]` when the robot is still (to within bias).
* Accel reads ≈ one g along whichever axis is *up* (depends on
  mounting; if you mounted Z-up it's `[0, 0, +9.81]`).
* Rates: shake along one axis, see only that axis spike on gyro
  (and on accel). If you see *all three* axes spike for a 1-axis
  shake, the chip is mis-oriented or there's loose wiring.

#### Stage 3 — gyro bias calibration

A small CLI binary `bin/calib-imu.rs` (or sub-command of an existing
one): hold the chassis still for ~10 s, average gyro to estimate
the constant bias, write it into `slam.toml` next to the camera
intrinsics. The existing config already has a `[slam]` table —
add an `[imu]` table with `gyro_bias = [bx, by, bz]` (in dps) and a
`temperature` field for sanity. Repeat the calibration if the
ambient temperature shifts a lot — BMI160 gyro bias drifts ~0.05
dps/°C.

Camera-IMU extrinsic calibration is **not needed for stage 4** if
you mount the IMU axis-aligned to the camera. Defer formal Kalibr-
style extrinsics until stage 5+.

#### Stage 4 — gyro-pre-integrated tracking-predict

The current predict in `src/slam/frontend.rs` is:

```rust
let predict = if np >= 2 {
    tracking::constant_velocity(&st.trajectory[np - 2], &st.trajectory[np - 1])
} else {
    st.trajectory[np - 1]
};
```

Replace the *rotational* part of `predict` with the gyro
pre-integration over the camera-frame interval. Translation predict
stays CV for now (accel-based translation predict is much harder —
requires reliable gravity subtraction and double integration that
drifts fast). Concretely:

1. Between camera frame `t_{k-1}` and `t_k`, pull all IMU samples
   from the ring buffer and apply `ΔR = Π exp([ω_i - bias] × Δt_i)`
   in SO(3). Use the slam-core `Lie` helpers — there is already a
   `so3_exp` for this.
2. Combine with the last pose's rotation:
   `R_predict = R_{k-1} * R_camera_imu * ΔR * R_camera_imu^T` (the
   conjugation handles the camera/IMU frame difference; if you
   mounted axis-aligned, `R_camera_imu = I`).
3. Translation predict: keep CV.
4. Feed this predict into `tracking::track_pose` exactly as today.

Add a flag (config or env var) to fall back to vision-only predict
so we can A/B the change and have a kill switch if the IMU integration
introduces a regression.

Headless test: extend the existing tracking unit tests with a
synthetic gyro stream and assert the predict rotates by the
expected amount over a synthesized 1 s spin.

#### Stage 5 — validation against today's failures

Re-run session 2's failing fast-spin sequence with the same duty
values (`±1800` for 1.2 s after a 1 s forward). The pass criteria:

* `tracking=true` throughout the spin.
* Map grows during the spin (kfs accepted at IMU-predicted poses
  give the mapper a baseline to triangulate against later).
* `n_lost` does not increment.

If we still lose tracking, the next-most-likely culprit is camera-
IMU temporal sync (we use `Instant::now()` for both, which can drift
relative to the camera's hardware timestamp). Fall back to using the
libcamera frame timestamp + the BMI160 `sensortime` register and
estimate a constant offset.

### Gotchas (write these in code comments at the touch points)

* **Camera-IMU time alignment.** The libcamera frame timestamp is the
  start-of-exposure (or end-of-exposure — check `request.metadata`),
  not the frame-arrival time at our pipeline. Off-by-one-frame
  alignment shows up as systematic over-rotation. Acceptable for
  stage 4; revisit if stage 5 needs it.
* **Coordinate frames.** Gyro `[ωx, ωy, ωz]` is in the *IMU* frame.
  Camera SE(3) twist uses `[ρ; φ]` (translation, rotation) and the
  rotation is in the *camera* frame. If the IMU is mounted with X
  forward and the camera is mounted with Z forward (typical), a
  90° axis swap is needed before integration. Write down the
  convention in the bmi160 module docstring and unit-test it.
* **Gyro bias drift with temperature.** Don't bake the calibration
  into the binary; keep it in `slam.toml` so it's easy to recalibrate.
* **Headless test must not require hardware.** Gate the actual i2c
  path behind `cfg(feature = "libcamera")` (or a new `cfg(feature =
  "imu")`); have the SLAM consume an `IMU: trait` with a mock
  implementation that the test feeds synthetic gyro samples through.
* **Pi I²C clock.** Default is 100 kHz; with three slaves on the bus
  (PCA9685, ADS7830, BMI160) that's borderline for a 200 Hz gyro
  stream. Bump to 400 kHz via `/boot/config.txt`
  (`dtparam=i2c_arm_baudrate=400000`) — well within everything's
  spec.

### Success criterion for the whole IMU thread

Same as the session-2 goal, just now achievable: the robot can be
driven in a multi-minute exploration of the room — including
in-place direction changes — with `tracking=true` for the whole
session and the map growing monotonically. Save the `/api/slam/map`
snapshot that proves it.

## Workflow

* Drive via the existing HTTP endpoints — short bursts only, and
  always look at `/api/camera/frame` first to confirm the camera sees
  what we think.
* Each iteration: form a hypothesis, write the targeted fix, add a
  test if the failure can be reproduced headlessly (preferred — see
  the `relocalizes_after_track_loss` style in
  `src/slam/frontend.rs`), run the workspace DoD, `make deploy`,
  observe.
* For the IMU thread specifically: every stage merges to `main`
  on its own; do not stack stages in a single PR (each stage
  changes behaviour subtly and should be bisectable).
* Keep the *running TODO* in this PLAN.md and delete it (the whole
  file) when room-scale exploration with tracking succeeds.

## Out of scope

* Camera calibration. The FOV-derived prior is good enough for a
  bench test; a real ChArUco run is a separate work item.
* Loop closure quality tuning. The mapper panic is fixed; observed
  loop closure behaviour can be revisited once tracking is reliable.
* Frontend panic-resilience around the world mutex. With the mapper
  panic gone we shouldn't see a poisoned lock in normal operation;
  defense-in-depth there is a deferred task.
* **Full visual-inertial bundle adjustment.** Stage 4 only uses the
  IMU for the tracking-predict, not as a BA constraint. Adding IMU
  factors to the local BA is the right *next* increment after Stage
  5 lands; it's not a prerequisite for the room-exploration goal.
