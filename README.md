# Ginger

[![CI](https://github.com/hmeyer/ginger/actions/workflows/ci.yml/badge.svg)](https://github.com/hmeyer/ginger/actions/workflows/ci.yml)
[![Audit](https://github.com/hmeyer/ginger/actions/workflows/ci.yml/badge.svg?event=push&label=audit)](https://github.com/hmeyer/ginger/actions/workflows/ci.yml)

Rust driver library and web control interface for the [Freenove 4WD Smart Car Kit (FNK0043)](https://docs.freenove.com/projects/fnk0043/en/latest/) running on a Raspberry Pi 4, PCB v2.0.

## Hardware

| Subsystem | Chip / interface | Notes |
|---|---|---|
| Motors (4WD) | PCA9685 → I2C 0x40 | Channels 0–7, H-bridge pairs |
| Pan/tilt servos | PCA9685 → I2C 0x40 | Channels 8 (pan) and 9 (tilt) |
| ADC / battery / light | ADS7830 → I2C 0x48 | Ch 0 = light L, Ch 1 = light R, Ch 2 = battery |
| LED strip | 8× WS2812B → SPI0 | Bit-banged at 6.4 MHz, GRB order |
| Buzzer | GPIO 17 | HIGH = on; bit-banged square wave for pitch |
| Ultrasonic | HC-SR04 → GPIO 27/22 | Trigger / Echo |
| IR line sensors | 3× → GPIO 14/15/23 | Active-high: HIGH = line detected |
| Camera | OV5647 → CSI | Via libcamera, 800×600 YUYV |

## Crate layout

A Cargo workspace: the camera/hardware-coupled binary plus two
dependency-light, camera-free crates (so the SIMD + SLAM math
cross-compile and unit-test without libcamera). The binary is layered
top-down `hal → devices → robot → server → bin`, with `camera`/`video`
as parallel media stacks, `slam` as the visual-SLAM frontend, and `api`
as the shared wire contract.

```
crates/
  fast/             — FAST-9 + grayscale image/pyramid (NEON), no deps but rayon
  slam-core/        — camera-free geometry/optimization: lie (SO3/SE3),
                      camera (pinhole+Brown–Conrady), intrinsics,
                      optimize (LM+Huber), twoview, tracking, map
                      (keyframes/covisibility), local_ba (Schur),
                      triangulation, bow (Bag-of-Words DB),
                      pnp (P3P+RANSAC), sim3 (Sim3+pose-graph), dataset
src/
  error.rs        — crate-wide Error / Result
  api.rs          — wire contract: telemetry, command protocol, request bodies
  hal/            — raw bus/peripheral drivers (rppal-facing)
    pca9685.rs    — I2C PWM driver (PCA9685)
    adc.rs        — ADS7830 battery/voltage/light ADC
    led.rs        — WS2812B strip over SPI
    buzzer.rs     — Buzzer (on/off + bit-banged variable-pitch tone)
    ultrasonic.rs — HC-SR04 distance sensor
    infrared.rs   — 3-sensor line tracker
  devices/        — actuators on the HAL
    motors.rs     — 4WD differential drive
    pan_tilt.rs   — Pan/tilt with invert + trim
  camera/
    capture.rs       — OV5647 via libcamera (streaming background thread)
    mock.rs          — headless frame source (no libcamera; CI / replay)
    auto_exposure.rs — pure software-AE controller (unit-tested)
  video/
    h264.rs       — V4L2 hardware H.264 encoder
    webrtc.rs     — WHEP signalling + adaptive bitrate
  slam/           — visual-SLAM frontend (init → track → map → reloc/loop)
    mod.rs        — Frontend state machine (Stage enum) + run() thread
    mapper.rs     — decoupled local-mapping thread (triangulation, BA, loop close)
    place.rs      — BoW place-recognition DB (relocalization / loop detection)
    fast.rs / brief.rs / image.rs — detection/descriptor/pyramid glue
  robot/          — domain
    car.rs        — Top-level Car struct with obstacle-avoidance safety
    supervisor.rs — Teleop control loop: collision lock, TTC, dead-man stop
  server.rs       — axum router + handlers (incl. /api/slam/stream, /map)
  bin/
    main.rs       — Composition root: wire up state, spawn supervisor, serve
    web/
      index.html  — Embedded mobile-first control UI (+ SLAM overlay/map)
examples/
  slam_bench.rs   — A/B frontend + geometry timing harness
  slam_replay.rs  — deterministic detect→match over a PGM sequence
scripts/
  install-service.sh — Install ginger as a systemd user service
```

## SLAM status

The full monocular ORB-SLAM pipeline is implemented and on `main`:
detection → two-view bootstrap → live 6-DoF tracking → decoupled local
mapping (keyframes, triangulation, block-sparse Schur local BA) → BoW
relocalization on track loss → BoW + Sim3 loop detection →
Essential-graph pose-graph correction, surfaced on the WebUI top-down
canvas (trajectory + keyframes + points, snapping on loop closure) and
gated by deterministic headless tests (`cargo test --workspace
--no-default-features`).

**Deferred — need the physical robot + target in one session:**
- **Proper camera calibration** — an offline OpenCV ChArUco tool
  emitting a verified `slam.toml` (today: the rev 1.3 FOV-derived prior,
  flagged `UNVERIFIED`). Not kalibr.
- **Frame recorder** — dump live libcamera frames to `*.pgm` to feed
  the existing replay harness with real scenes. Real-scene init quality
  (M3) and loop-closure efficacy (M6) are only verified on synthetic /
  unit-test scenes until this lands; the synthetic harness can't
  manufacture the drift a closing loop needs.

**Performance / refinement passes (measure first via `slam_bench`):**
- Wire the BoW direct index into tracking/loop matching (still
  brute-force descriptor matching against the whole map).
- Post-loop global bundle adjustment (loop closure currently only
  pose-graph-corrects + drags points).
- Finer tracking/mapping handoff (the coarse single `Arc<Mutex<Map>>`
  lets heavy local BA / pose-graph stall a tracking frame).
- Analytic Jacobians for `tracking` / `local_ba` (currently central
  finite differences) — parity-tested.
- Hand-written NEON only at a measured hotspot (e.g. BoW Hamming
  `vcntq_u8` + `vaddvq`), keeping the `crates/fast`
  scalar→NEON→parity discipline.

## Dependencies

```bash
sudo apt install libcamera-dev libclang-dev
```

Rust toolchain via [rustup](https://rustup.rs).

## Building

```bash
cargo build --release   # always use release; debug is 10-20× slower for JPEG encoding
# or via make:
make build
```

`libcamera` is an opt-out default feature (the only system-coupled
dep). On a dev machine / CI without libcamera, build & test headless
with a mock camera — this also exercises the full SLAM pipeline:

```bash
cargo test  --workspace --no-default-features   # mock camera, no libcamera
cargo check -p ginger-slam-core --target aarch64-unknown-linux-gnu
```

## Linting

```bash
make lint          # fmt check + clippy (also runs as pre-commit hook)
make audit         # check dependencies for known CVEs (requires cargo-audit)
make install-hooks # install git pre-commit hook
```

## Running as a system service

Install as a systemd user service that starts at boot and **automatically restarts whenever the binary is rebuilt**:

```bash
bash scripts/install-service.sh
```

After that, `cargo build --release` is all you need to deploy — the running server is replaced within a second or two. Logs go to the system journal:

```bash
journalctl --user -u ginger -f
```

## Web interface

The `ginger` binary serves a mobile-first control UI on port 8080:

```bash
cargo run --release --bin ginger
# Open http://<pi-ip>:8080  or  http://ginger.local:8080
```

Features:
- **Live sensor feed** via SSE — battery voltage, light sensors (L/R), IR line tracker (3 dots), ultrasonic distance + time-to-collision estimate, all updating at 200 ms
- **Camera stream** — adaptive JPEG, targeting 30 fps; quality and resolution scale down automatically to stay within budget
- **Onboard FPS** (camera capture rate) and **web FPS** (browser delivery rate) displayed live
- **Software auto-exposure** — single brightness-axis controller (gain-first ramp: gain 1→16× before exposure 0.5→100 ms) with a Smith Predictor that uses each frame's metadata to compensate for libcamera's pipeline delay. Live readback of luma, brightness (EV stops), exposure, and gain — no knobs
- **Drive controls** — spring-back virtual joystick (expo response curve so it isn't near-binary) or keyboard arrow keys, space to stop; server safety-stops motors after 500 ms of silence
- **Forward collision avoidance** — hard stop at 30 cm with hysteresis to 38 cm, obstacle lock prevents the browser's drive-command heartbeat from re-overriding; time-to-collision displayed with colour coding (red < 1 s, orange < 2 s, green)
- **Spring-back camera bracket** — releasing the camera joystick re-centers pan/tilt so the ultrasonic sensor faces forward (replaces the old brittle distance-estimated auto-center)
- **Pan trim** — physical straight-ahead baked in as a servo pulse offset; `set_pan(90°)` always points the sensor forward
- **Pan / tilt joystick** — 0–180° range, spring-back to forward
- **Express** — one-tap synchronized LED + buzzer "expression": a random *mood* (excited / curious / grumpy / alarmed / chatty) fixes the pitch band, tempo, rise/fall contour and a matching colour palette; a random *viz* (scanner / VU meter / flood) makes the lights track pitch in lock-step with the bit-banged buzzer warble
- **Sensor toggles** — enable/disable light, IR, and ultrasonic per sensor

The UI is mobile-first: on phones it stacks camera → scrollable sensor strip → footer controls. On screens ≥ 700 px it switches to a camera + sidebar layout.

## Camera

The `Camera` struct runs a background thread that continuously captures YUYV frames from the OV5647. Frames are published behind an `Arc` — callers either poll or block:

```rust
let cam = Camera::new()?;          // blocks until first real frame (~1 s warmup)
let frame = cam.get_frame();       // latest frame, non-blocking
let frame = cam.wait_frame();      // blocks until next new frame
let rgb = frame.to_rgb();          // YUYV → packed RGB
frame.save_ppm("/tmp/out.ppm")?;   // save without extra deps
```

Exposure is controlled by an internal AE loop that targets luma 128. The current state — luma, brightness (EV stops above darkest), applied exposure, applied gain — is exposed read-only via `cam.exposure_cfg` (`Arc<Mutex<ExposureConfig>>`).

The controller uses a single 1D brightness axis (gain ramps 1→16× before exposure extends 0.5→100 ms) and a Smith Predictor: the luma we measure reflects an *old* sensor setting (libcamera has ~3 frames of pipeline delay), so we predict what luma would be with our latest target applied and step against the predicted error. This converges in well under a second without overshoot or oscillation, and the prediction means we step every frame — no waiting.

## Car safety

`Car::drive()` checks the ultrasonic sensor before and during forward motion and stops automatically if an obstacle is closer than 30 cm.

```rust
let mut car = Car::new()?;
car.forward(2000, Duration::from_secs(1))?;  // stops early if blocked
car.turn_right(2000, Duration::from_millis(400))?;
car.stop()?;
car.close()?;
```

The web server adds a second layer: an obstacle lock in the hardware thread prevents the browser's 150 ms drive-command heartbeat from overriding a collision stop.
