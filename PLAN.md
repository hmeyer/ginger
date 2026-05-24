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

**2026-05-24 — session 3 (Claude + Ginger) — IMU is wired and live**

* **Stages 0, 1, 2 are DONE and deployed.** The BMI160 is on the bus at
  `0x69` (SDO tied to VCC), `i2cget` returns `0xd1`. `src/hal/bmi160.rs`
  drives it through an `I2cBus` trait with five unit tests against a
  recording mock. `src/imu/mod.rs` owns a polling thread that publishes
  `latest()` + a 10 s ring (`recent_since(t)`) for Stage 4 to consume,
  plus a 1 Hz info log. `/api/imu/sample` returns the latest reading
  with frame↔sample sync diagnostics. **The WebUI sidebar got an IMU
  card** (gyro x/y/z dps, accel x/y/z m/s², achieved rate, sync gap
  ms with green/amber/red tinting at 110/300 ms thresholds) plus a
  mobile-strip summary (`|gyro|`, Δt). Commits `4e01ecf`, `5f956f2`,
  `de09441`.
* **Frame↔sample sync invariant holds.** `Frame.t_capture` and
  `ImuSample.t_read` are both `Instant::now()` on the host's
  `CLOCK_MONOTONIC`, set at request-complete (libcamera / mock) and
  immediately after the I²C burst respectively. Observed sync gap
  on the live binary is **−2 to +20 ms**, i.e. roughly uniform inside
  one camera period (33 ms at 30 fps), which is the expected
  distribution for a 200 Hz IMU and ~30 Hz camera. Stage 4's
  pre-integration can rely on the two clocks being directly subtractable
  without conversion.
* **At-rest sensor reads check out.** Gyro `~[0.2, -0.4, 0.2] dps`
  (constant bias). Accel `~[9.86, -0.60, 0.08] m/s²` — magnitude
  9.88 ≈ g, so the chip is essentially **Z-up** with a small Y-axis
  tilt (~3.5°). Useful inference: the gyro axis that measures yaw
  in-place is **Z**, since rotation about gravity = rotation about
  +Z_imu given this mounting.
* **Achieved IMU rate is 142 Hz, not 200 Hz.** Bus contention with the
  PCA9685 (motor PWM updates) and ADS7830 (battery polls). The plan's
  "Pi I²C clock" gotcha applies: bumping `/boot/firmware/config.txt`'s
  `dtparam=i2c_arm_baudrate=400000` is the documented fix and the next
  reboot is the right time to do it. **Not blocking Stage 4:** the
  predict only needs more samples than the camera-frame interval
  contains, and 142 Hz ÷ 30 fps ≈ 4.7 samples per frame is plenty
  for an inter-frame rotation integral.
* **Stage 3 (bias persistence) is being downscoped.** BMI160 gyro bias
  drifts ~0.05 dps/°C. Persisting a value into `slam.toml` and reloading
  on next boot bakes in stale temperature; auto-bias on every boot
  (average the first ~1 s of stationary samples) captures the *current*
  temperature with no setup ritual. Cost: a 1 s "hold still after boot"
  window during which the predict is bias-uncorrected — at 0.2 dps that
  is 0.2°/s × 33 ms/frame = 0.007°/frame, negligible.

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

#### Stage 0 — post-solder verification (no Rust yet) — **DONE**

Confirmed `0x69` is alive on the bus and `i2cget -y 1 0x69 0x00`
returns `0xd1`. Chip is wired correctly.

#### Stage 1 — HAL driver (`src/hal/bmi160.rs`) — **DONE** (`4e01ecf`)

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

#### Stage 2 — sample loop + 1 Hz log + new HTTP endpoint — **DONE** (`5f956f2`, WebUI surface in `de09441`)

Polling thread + 10 s ring keyed by `Instant` + `/api/imu/sample`. Frame
↔ sample sync verified live (gap −2 to +20 ms; within one camera
period). Achieved rate **142 Hz** (vs the 200 Hz target) due to PCA9685
/ ADS7830 bus contention — covered by the "Pi I²C clock" gotcha; bumping
to 400 kHz is the documented fix for the next reboot. Not blocking
Stage 4 since 142 Hz still gives ~4.7 samples per 30 fps camera frame.

#### Stage 3 — gyro bias (auto-on-boot, no persistence) — **NEXT**

Downscoped from the original "CLI + `slam.toml` round-trip" plan because
the bias is temperature-dependent (~0.05 dps/°C); persisting yesterday's
value to disk is *less* accurate than re-estimating today's. The polling
thread (`src/imu/mod.rs`) does the work:

1. **Stationary detection** — first `~1 s` after `Imu::open`, watch the
   per-sample gyro magnitude. If the max sample magnitude across the
   window stays below `STATIONARY_DPS = 2.0` (well above the chip's
   noise floor but well below any deliberate motion), accept the
   window's mean as bias. Otherwise log a `warn!` and leave bias at
   zero; the operator should re-call `Imu::recalibrate_bias()` (also
   exposed as `POST /api/imu/calibrate`) once the chassis is still.
2. **Expose** `Imu::gyro_bias_dps() -> [f32; 3]`. SSE handler and
   `/api/imu/sample` subtract it before reporting so the WebUI shows
   "near zero" at rest immediately after the warm-up. Raw samples in
   the ring stay unbiased; Stage 4 subtracts the bias inside the
   integrator so the rotation isn't double-corrected.

Camera-IMU extrinsic: **not needed for Stage 4** under the observed
mounting. At-rest accel = `[~0, ~0, +g]` ⇒ IMU body Z is gravity-up,
which means yaw (rotation about gravity) is `gyro_z`. The robot's
horizontal-plane motion is yaw-dominated, so for Stage 4 we treat
gyro X and Y as small corrections and integrate the full 3-vector
with a configurable `R_camera_imu` whose default is identity. Formal
Kalibr-style extrinsics are deferred unless validation shows axis
misalignment.

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
drifts fast).

**Hook point.** `Frontend::on_frame` is the only entry into the state
machine (lines ~588–592 of `src/slam/frontend.rs`). The cleanest seam
is to extend its signature with `rotation_hint: Option<UnitQuaternion<f64>>`
representing **camera-frame ΔR since the previous frame**. Internally:

* If `rotation_hint` is `Some` and `np >= 1`, use
  `predict.rotation = st.trajectory[np-1].rotation * hint`,
  `predict.translation = constant_velocity(...).translation`
  (or just `st.trajectory[np-1].translation` if `np < 2`).
* Else, fall through to the existing CV predict path unchanged.

This keeps `slam-core` camera-free (no IMU dependency on the inner
crate) and the call site in `src/slam/mod.rs::run` is the only place
that pulls from `imu.recent_since(t_prev_capture)` and applies the
extrinsic.

**Integrator** (in `src/slam/mod.rs::run`):

1. Track `t_prev_capture: Option<Instant>` across frames.
2. On each frame, call `imu.recent_since(t_prev_capture.unwrap_or(now))`.
3. Sum: `ΔR_imu = ∏ exp([(ω_i - bias) · dt_i]_×)` in SO(3), where
   `dt_i` is the gap to the next sample (last sample uses the gap to
   `frame.t_capture`).
4. Apply extrinsic: `ΔR_cam = R_camera_imu · ΔR_imu · R_camera_imu^T`.
   Default `R_camera_imu = I` per the Stage 3 mounting inference;
   override via env var if validation shows axis swap is needed.
5. Pass `Some(UnitQuaternion::from_matrix(&ΔR_cam))` into
   `on_frame(...)`.

**A/B switch.** `GINGER_IMU_PREDICT=0` (env var, read once at startup
in `slam/mod.rs::run`) falls back to vision-only — kill switch if the
IMU integration regresses something.

**Headless tests.** No camera or hardware needed:

* `lie::so3_exp` round-trips: spin a synthetic 1 s 90°/s rotation
  through the integrator and assert recovered `ΔR ≈ R_x(90°)`.
* `Frontend::on_frame` with a `rotation_hint` matching the synthetic
  scene's true motion gives a tighter inlier count than the CV
  predict alone (regression of "tracking through fast motion").

#### Stage 5 — validation: fast spin + room loop

Driven over HTTP with **visual control before each pulse** — i.e.
`curl /api/camera/frame -o /tmp/now.jpg` then look at the image
before sending a `/api/drive` so we don't drive into anything.

1. **Fast-spin reproduction** (the session-2 failure): forward at
   `duty=1500` for 0.7 s, then *immediately* in-place spin `±1800`
   for 1.2 s, no stop between. Watch `/api/slam/map` while doing
   it. Pass criteria:
   * `tracking=true` throughout the spin (was: collapsed in 14 frames).
   * `n_lost` doesn't increment.
   * Map grows during the spin (or at minimum survives it).
2. **Slow room loop** (the session-2 cumulative-rotation failure):
   three back-to-back gentle right arcs (`left=900, right=300`,
   ~0.8 s each) totalling ~90°. Same pass criteria.
3. **Multi-minute exploration** (the actual goal in CLAUDE.md):
   alternating short forward pulses (`±1500` for 0.5–1 s) and gentle
   turns (`differential ≤ 1200`) for several minutes, with visual
   confirmation before each pulse. Save the `/api/slam/map`
   snapshot at the end as evidence.

If tracking still collapses on (1), the most-likely culprits in
order are: (a) wrong extrinsic — re-check `R_camera_imu` by
spinning in-place and asserting `gyro_z` dominates; (b) bias not
yet learned at the moment of the spin (warm-up window collided
with the test); (c) camera-IMU temporal sync drift — fall back to
the libcamera `SensorTimestamp` metadata + BMI160 `sensortime`
constant-offset fit.

### Gotchas

* **Camera-IMU time alignment — current status.** Both stamps come
  from `Instant::now()` on `CLOCK_MONOTONIC`. The libcamera
  `SensorTimestamp` metadata (true start-of-exposure) was deliberately
  *not* used for Stage 2 — `Instant::now()` at request-complete is
  ~one frame later but jitter-free and on the same clock as the IMU.
  Stage 4's integral runs *between* frames, so a constant offset
  cancels out. Live sync gap is −2..+20 ms (well inside one camera
  period). Only revisit if Stage 5 fails (a) and the other two
  hypotheses don't pan out.
* **Coordinate frames — mounting inferred from gravity.** At rest
  `accel ≈ [0, 0, +g]` ⇒ IMU body Z = up. Yaw (rotation about
  gravity) = `gyro_z`. Camera convention is Z-forward, Y-down;
  camera-Y is gravity-aligned. The integrator default extrinsic is
  `R_camera_imu = I` (which assumes gyro X/Y are interpreted as
  camera X/Y, gyro Z as camera-Y); the dominant in-plane motion is
  yaw, which Z handles correctly. If Stage 5 shows the predict
  rotates the wrong sign or axis, **first** dump a 1 s in-place spin
  trace to `/tmp/imu-spin.csv` and check which gyro axis dominates,
  **then** set `R_camera_imu` via env var rather than tweaking math.
* **Gyro bias drift with temperature.** Don't persist to `slam.toml`
  — temperature-dependent (~0.05 dps/°C). Auto-bias on every boot
  (see Stage 3) captures current temperature without any setup ritual.
* **Headless test must not require hardware.** The driver's `I2cBus`
  trait + `MockI2cBus` is in place (`src/hal/bmi160.rs`). The IMU
  thread uses the same trait so the polling loop, ring, and bias
  estimator can be exercised without hardware (`src/imu/mod.rs`
  `tests` module).
* **Pi I²C clock — apply after next reboot.** Default 100 kHz, three
  slaves on the bus → achieved IMU rate is 142 Hz vs 200 Hz target.
  Bump via `/boot/firmware/config.txt` (not `/boot/config.txt` on
  modern RPi OS): `dtparam=i2c_arm_baudrate=400000`. Reboot required.
  Stage 4 works at 142 Hz; this is a polish item.

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
