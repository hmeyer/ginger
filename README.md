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
relocalization on track loss (re-bootstrapping a fresh session if
recovery keeps failing) → BoW + Sim3 loop detection →
Essential-graph pose-graph correction, surfaced on the WebUI top-down
canvas (trajectory + keyframes + points, snapping on loop closure) and
gated by deterministic headless tests (`cargo test --workspace
--no-default-features`).

**Verified on real hardware (2026-05-23):** two-view bootstrap +
tracking, end-to-end on the Pi from forward-only driving. The
diagnostic fields on `GET /api/slam/map` (`boot_matches`,
`boot_median_disp_px`, `boot_min_disp_px`, `boot_anchor_age`,
`boot_anchor_resets`, `last_lost_reason`, `n_lost`) drove three
re-tunings the synthetic suite never could have surfaced:

- `INIT_MIN_DISP_FRAC: 0.04 → 0.025` — pure forward motion on a
  differential-drive base produces ~20-26 px median parallax per
  ~25 cm pulse against a same-session anchor; the old 4% gate
  (32 px on an 800 px frame) was unreachable in a single drive.
- `ANCHOR_RESET_MATCHES: 25 → 40` — BRIEF matches against the
  anchor decay through (25, 80) as the scene shifts; below
  `INIT_MIN_MATCHES` parallax can no longer be measured, but the
  old floor of 25 kept the anchor alive uselessly in that gap.
- Tracking accept condition no longer requires `rep.converged`
  (the shared LM optimizer uses `gradient_tol = 1e-10`, calibrated
  for offline BA — motion-only BA is "practically correct" with
  tens of inliers long before that gradient norm).

After all three: sustained tracking through forward + backward
motion, BoW vocabulary self-trains, relocalization recovers from
single-frame losses without re-bootstrapping. Synthetic test count
unchanged; new behavior is on top of the same code paths.

**Deferred — need the physical robot + target in one session:**
- **Proper camera calibration** — an offline OpenCV ChArUco tool
  emitting a verified `slam.toml` (today: the rev 1.3 FOV-derived prior,
  flagged `UNVERIFIED`). Not kalibr.
- **Frame recorder** — dump live libcamera frames to `*.pgm` to feed
  the existing replay harness with real scenes. Loop-closure efficacy
  is still synthetic-only.

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

## Deploying CI builds (no on-Pi compile)

Compiling on the Pi is slow. The [`RPi build`](.github/workflows/rpi-build.yml)
workflow builds the release binary on a GitHub-hosted arm64 runner on every
push to `main` (or via *Run workflow*) and uploads it as a `ginger-aarch64`
artifact. The build runs in a Debian Trixie container so glibc and the
pinned libcamera version match the Pi — the artifact is a drop-in binary.

The Pi-side deploy is trigger-driven, not polled:

```bash
make deploy        # = git push + start the burst pull
```

`scripts/deploy.sh` pushes (any extra args are forwarded — e.g.
`make deploy ARGS="--force-with-lease"`), then `restart`s
`ginger-pull.service`, which runs `scripts/pull-burst.sh`: a 10s loop
that calls `pull-binary.sh` until CI finishes and a new artifact is up
(typically ~3 min), or for at most 15 min. Each pull verifies the
artifact's SHA-256 and atomically replaces `target/release/ginger`;
`ginger-watch.path` then restarts `ginger.service`. There is no
recurring timer — the service is idle between deploys.

Watch a deploy land:

```bash
journalctl --user -u ginger-pull -f
```

GitHub Actions artifacts require auth even on a public repo. The
simplest setup is to be logged in with `gh`:

```bash
gh auth login    # one-time; pull-binary.sh falls through to `gh auth token`
```

Or write a PAT with `Actions: read` to `~/.config/ginger/gh-token`
(chmod 600), or export `GINGER_GH_TOKEN`. Without any of those,
`make deploy` still pushes but the burst exits with `rc=11`.

`pull-binary.sh` needs `jq` and `unzip` (`sudo apt install jq unzip`).
When the Pi's libcamera is upgraded, bump `LIBCAMERA_VERSION` in the
workflow to match `dpkg -l libcamera0.7` so the linked ABI stays
correct. A merge made via the GitHub web UI won't auto-deploy (no local
push to wrap) — run `systemctl --user restart ginger-pull.service` by
hand, or `make deploy` the next time you push.

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
- **Emote** — one-tap synchronized LED + buzzer emote: a random *mood* (excited / curious / grumpy / alarmed / chatty) fixes the pitch band, tempo, rise/fall contour and a matching colour palette; a random *viz* (scanner / VU meter / flood) makes the lights track pitch in lock-step with the bit-banged buzzer warble
- **Sensor toggles** — enable/disable light, IR, and ultrasonic per sensor

The UI is mobile-first: on phones it stacks camera → scrollable sensor strip → footer controls. On screens ≥ 700 px it switches to a camera + sidebar layout.

### Diagnostic endpoints

- `GET /api/camera/frame?w=320&q=70` → single grayscale JPEG of the
  most recent frame (~5 KB at defaults). Useful for headless "is the
  path clear?" checks without negotiating WebRTC. `w` clamps to the
  source width; `q` is JPEG quality 10..100. Returns 503 with
  `Retry-After: 1` before the first frame.
- `GET /api/slam/map` → JSON snapshot: tracking state, init/parallax
  status string, keyframe and map-point counts, BoW state. Also
  carries per-frame bootstrap diagnostics (`boot_matches`,
  `boot_median_disp_px`, `boot_min_disp_px`, `boot_anchor_age`,
  `boot_anchor_resets`) populated while pre-init, and sticky
  tracking-loss telemetry (`last_lost_reason`, `n_lost`) that
  survives the relocalize window so a poller can see *why* tracking
  failed without racing the frame loop.

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
