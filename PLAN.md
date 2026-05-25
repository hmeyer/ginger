# PLAN: hardwood-floor exploration with a learned motor model + depth + ultrasonic

## Why this plan exists

The ORB-SLAM frontend keeps going LOST during room exploration in this
apartment. Six tuning rounds (IMU pre-integration predict, min-inliers
down to 4, longer pauses between pulses, BoW reloc, panic-fix) bought
individual sessions but did not change the underlying problem: hardwood
floors plus blank walls do not give a feature tracker enough persistent
landmarks. Direct / dense visual methods inherit the same scene
limitation.

This plan pivots the exploration path off visual SLAM entirely. The
replacement stack:

* **Motor model** — small, online-learned function
  `(pwm_l, pwm_r) → (v, ω)`, persisted to disk and refit continuously
  while driving. Exploits the fact that the apartment is uniform
  hardwood (one floor coefficient is enough).
* **Gyro** — BMI160 already integrated; gives `ω` as a continuous,
  zero-cost label and overrides the model's `ω` at runtime.
* **Swept ultrasonic** — the HC-SR04 sits on the pan servo, so it can
  build a ~150° local scan at a stationary waypoint. Cheap, no ML.
* **Monocular depth predictor** — on-device neural net at ~1 Hz, gives
  dense structure for mapping and obstacle avoidance. Highest-risk
  component; sequenced last so the rest of the stack works without it.

The existing visual stack (`src/slam`, `crates/slam-core`, `crates/fast`,
`src/imu`) is **not deleted** — `crates/*` are camera-free,
well-tested, and may be useful if we ever add stereo or RGB-D. They
just stop being on the critical path for exploration.

## Architecture target

```
camera frame ──> depth predictor (~1 Hz) ─────┐
                                              │
ultrasonic (panned, swept) ──> local scan ───┤
                                              ├──> 2D occupancy grid
motor PWM ──┐                                 │       (robot frame)
gyro ──┐    │                                 │
       │    └──> motor model (MLP) ──> (v, ω) ──┴──> pose integrator
       └──> ω label                       ▲
ultrasonic Δd/Δt ──> v label ─────────────┘  (online SGD update)
```

Persistence:

* `motor-model.toml` (gitignored) — MLP weights + normaliser
  constants + step count + the battery voltage they were fit at.
  Read at startup, written on graceful shutdown and every ~60 s
  while driving.
* Calibration (`slam.toml`) is unrelated and stays untouched.

## Stages

Each stage is a separately-mergeable commit / PR. Headless DoD
(`cargo test --workspace --no-default-features`) must stay green
at every stage; nothing in the non-libcamera path requires hardware.

### Stage 1 — motor model: MLP, fit, persistence

Wholly camera-free. Builds the foundation everything else needs:
"given a motor command and the chassis's current state, what is it
doing right now."

New module `src/motion/model.rs`. Public API:

```rust
pub struct MotorModel { /* MLP weights + optimizer state + meta */ }

impl MotorModel {
    pub fn load_or_default(path: &Path) -> Result<Self>;
    pub fn save(&self, path: &Path) -> Result<()>;

    /// Forward inference: motor command + current dynamic state → predicted motion.
    pub fn predict(&self, x: ModelInput) -> Motion;

    /// Online update from one labelled window. One SGD step.
    pub fn observe(&mut self, sample: LabelledSample);
}

pub struct ModelInput {
    pub pwm_l: i32, pub pwm_r: i32,
    pub pwm_l_prev: i32, pub pwm_r_prev: i32,  // last tick — captures Δcommand
    pub v_prev: f32,         // last predicted v (or last gyro-confirmed pose Δ)
    pub omega_prev: f32,     // last gyro-measured ω
    pub battery_v: f32,
}

pub struct Motion { pub v: f32, pub omega: f32 }  // m/s, rad/s

pub struct LabelledSample {
    pub input: ModelInput,
    pub omega_meas: f32,           // from gyro, always present
    pub v_meas: Option<f32>,       // from ultrasonic, sometimes
    pub dt_s: f32,
}
```

#### Architecture

A 7 → 16 → 16 → 2 fully-connected MLP. Tanh activations on hidden
layers (symmetric, bounded, well-behaved with signed motion); linear
output. ≈ 430 trainable parameters total — small enough to inspect by
eye in `motor-model.toml`, small enough to forward in ~µs on the Pi,
big enough to capture the nonlinearities we care about (deadband at
low PWM, saturation at high PWM, turning-radius dependence on speed,
inertia from `(v_prev, ω_prev)`, Δcommand transients from
`(pwm_*_prev)`, and battery droop).

Inputs are normalised before entering the network:

* `pwm_l, pwm_r, pwm_l_prev, pwm_r_prev` → divide by `MAX_DUTY` (4095)
  → roughly [-1, 1]
* `v_prev` → divide by 1.0 m/s (a reasonable indoor max)
* `omega_prev` → divide by 2.0 rad/s
* `battery_v` → `(V - 7.8) / 0.6` → roughly [-1, 1] across the
  usable battery range

Normaliser constants live in the file alongside the weights; never
hard-coded in load paths.

#### Training

* **Loss.** Weighted MSE:
  `L = (ω_pred - ω_meas)² + α · 𝟙[v_meas present] · (v_pred - v_meas)²`
  with `α ≈ 3` because v-labelled windows are sparser than ω-labelled
  windows and we want the v head to learn at a comparable rate.
* **Optimiser.** Adam-lite (Adam without bias correction; one extra
  vector per parameter). Learning rate `1e-3`, β₁ `0.9`, β₂ `0.99`.
  Roll our own — no ML dep on the Pi.
* **Backprop.** Hand-written. Two-hidden-layer tanh MLP with MSE loss
  is ~30 lines of careful code; an external autodiff or ML crate is
  overkill and adds aarch64 build pain.
* **Step cadence.** One SGD step per `LabelledSample` (~5 Hz when
  driving). 5 Hz × 60 s = 300 steps/min — fast convergence for a
  400-param model.

#### Bootstrap

Random Xavier init + a synthetic warm-up so a fresh `motor-model.toml`
doesn't make the robot drive blind for the first minute:

1. On first boot (no model file), generate 2000 synthetic samples
   from a hand-coded prior — a linear-with-deadband function that
   matches the operator note in `CLAUDE.md` (PWM 1500 ≈ 0.6 m/s).
2. Run 2000 SGD steps against the synthetic data. Save.
3. From there, the live label stream takes over.

The warm-up is deterministic (seeded RNG) and runs at module-init
time in ~10 ms — no user-visible delay.

#### Staleness check on load

If `motor-model.toml` was saved at a battery voltage > 0.3 V away from
the current reading, **don't reload it.** Re-bootstrap from synthetic
warm-up. Cheap insurance against the model file outliving its
calibration regime.

#### File format

TOML, human-readable, weight matrices as nested arrays:

```toml
[meta]
trained_steps = 14572
last_battery_v = 7.92
last_updated = "2026-05-25T13:42:17Z"
schema_version = 1

[normalizer]
pwm_scale = 4095.0
v_scale = 1.0
omega_scale = 2.0
battery_mean = 7.8
battery_scale = 0.6

[layer_1]  # 7 → 16
weights = [[...], ...]
bias = [...]

[layer_2]  # 16 → 16
weights = [[...], ...]
bias = [...]

[layer_3]  # 16 → 2
weights = [[...], ...]
bias = [...]
```

#### Tests (headless)

* Persistence round-trip: `save` → `load_or_default` gives bit-identical
  predictions on a fixed input set (and the file is human-inspectable).
* **MLP can fit a known nonlinear function.** Generate 5000 samples
  from a synthetic ground-truth: `v = tanh(2·pwm_avg) · 0.6` plus a
  deadband, `ω = 1.5 · (pwm_r - pwm_l) / MAX_DUTY · (1 + 0.3·v_prev)`
  (turning rate depends on speed). After 5000 SGD steps assert
  prediction RMSE < 20 % of target scale on a held-out set.
* **Recurrence reaches steady state.** Feed back `(v_prev, ω_prev)`
  from the previous prediction; with constant inputs the loop must
  converge to a fixed point within 0.5 s of simulated time
  (10 ticks @ 20 Hz).
* **Battery input has measurable effect.** Sweep `battery_v` from
  7.2 to 8.4 with everything else fixed; the output must vary
  (otherwise the model's ignoring the input).
* **Sparse v labels still learn the v head.** Synthetic stream where
  only 1 in 5 samples has `v_meas`; the v RMSE must still drop over
  training.
* **Bootstrap is deterministic.** Two fresh `MotorModel::default()`s
  with the same seed produce bit-identical weights.

#### WebUI surface

A new **"Motor model"** sidebar card. The model is the robot's
self-belief about its own dynamics — the operator needs to see it
both as health telemetry and as a tangible "what will happen if I
drive at PWM X" map.

* **Health row:** `trained_steps`, time since last update, current
  battery_v vs. fit-time battery_v (green / amber / red badge for
  staleness), schema version.
* **Loss row:** running mean ω-residual and v-residual over the
  last 60 samples (sparkline if cheap, otherwise just the number
  with a colour against an absolute target).
* **Prediction heatmap:** a 9×9 grid where each cell `(i, j)`
  shows the model's predicted `(v, ω)` for `pwm_l = -4095 + i·...`,
  `pwm_r = -4095 + j·...` (with `v_prev = 0`, `ω_prev = 0`,
  current `battery_v`). Colour by `v`, text by `ω`. This is the
  one-look "is the model sane" view — the diagonal should show
  smooth forward/reverse gradients, the off-diagonals should turn.
* **Manual probe:** a tiny form (`pwm_l`, `pwm_r` sliders) → live
  predicted `(v, ω)` text. Doesn't drive the motors; pure inspection.

Endpoints:

* `GET /api/motion/model` — `{ trained_steps, last_battery_v,
  last_updated, residual_v, residual_omega, weights_summary }` for
  the health rows.
* `GET /api/motion/model/predict?pwm_l=…&pwm_r=…&...` — runs one
  forward pass; backs both the heatmap and the manual probe.
* `POST /api/motion/model/reset` — re-runs synthetic warm-up;
  WebUI button for emergency recovery if the model goes off the
  rails on bad labels.

#### Stage 1 done when

* All tests green.
* `motor-model.toml` written on shutdown and reloaded on boot on the
  live binary; a manual `cat motor-model.toml` shows recognisable
  structure (the matrix shapes match, values are finite and roughly
  in [-2, 2]).
* WebUI motor-model card renders, prediction heatmap is non-empty,
  manual probe works.
* No behaviour change on the robot yet — model is a passive observer
  fed by Stage 2's labels and read by Stage 3's integrator.

### Stage 2 — label streams: gyro `ω` + ultrasonic `v`

The model from Stage 1 needs supervision. Two sources, both already
wired at the hardware level.

* **`ω` label.** Pre-integrate `gyro_z` over the 200 ms model-update
  window (existing `imu.recent_since` API). Divide by window duration.
* **Whole-window rejection** (drops both ω and v labels for the window
  — anything that produced a bad ω also produced bad v):
  1. Any single-sample |ω| > 5 rad/s — chassis bump.
  2. Any single-sample `| |accel| - g |` > 3 m/s² — collision,
     pickup, kick, or someone setting the robot down. Cheaper and
     earlier signal than gyro for those events; gravity magnitude is
     known and stable on flat hardwood.
* **`v` label.** Δd/Δt from the front-facing ultrasonic, **but only**:
  1. all readings in the window are within the sensor's reliable
     range (8 – 80 cm),
  2. the robot is commanded approximately straight (`|pwm_l - pwm_r|`
     under some threshold — turning windows produce useless v labels),
  3. distances are monotonic across the window (no obstacle moved or
     came into / left the beam).
  Expect maybe 10–20 % of windows to yield a usable v label. That is
  enough for an online fit because the gyro-labelled `ω` stream is
  abundant and the v fit is two parameters per sign.

New `src/motion/labels.rs`. A small worker reads PWM + gyro +
ultrasonic at 5 Hz, emits `LabelledSample`s, feeds them to
`MotorModel::observe`.

#### Tests (headless)

* Synthetic gyro stream + window → expected average ω.
* Synthetic ultrasonic stream: monotonic → correct v; non-monotonic →
  `None`; spike → `None`.
* Whole-window rejection: a window with a gyro spike is dropped;
  a window with `|accel|` 5 m/s² above g is dropped; a clean window
  passes.
* End-to-end: synthetic driver feeds 1 minute of varied commands;
  model converges to ground truth within 15 %.

#### WebUI surface

Extends the Stage 1 "Motor model" card with a **"Labels"** subsection
that lets the operator watch supervision in real time:

* **Counters:** total samples observed; ω-labelled count;
  v-labelled count; v-label rate (% of windows producing a v
  label, rolling 60 s).
* **Rejection breakdown:** counts by reason — whole-window drops
  ("gyro spike", "accel spike") separated from v-only drops ("out
  of range", "not straight", "non-monotonic"). Helps the operator
  notice "I drove for 5 minutes and got zero v labels — why?"
  without having to read logs.
* **Latest-sample table:** scrollable list of the last ~20
  `LabelledSample`s — pwm_l, pwm_r, ω_meas, v_meas (or "—"), age.
  This is the rawest possible view of what the model is being fed.
* **Loss sparkline:** rolling per-step loss (ω-residual,
  v-residual) over the last ~5 min. The "is it converging" view.

Endpoint: `GET /api/motion/labels` returns counters + the last N
samples in a small JSON blob.

#### Stage 2 done when

* On the live binary, `motor-model.toml` changes meaningfully across a
  5-minute drive (weights move off the synthetic-warm-up defaults,
  then stabilise).
* The "Labels" card shows a healthy v-label rate (≥ 10 % during
  straight-driving segments) and the loss sparkline trends down
  over the first ~2 minutes, then sits flat.

### Stage 3 — pose integrator

First user-visible signal that the new stack does something useful.

* `src/motion/pose.rs`. A worker at 20 Hz reads the latest commanded
  PWM, predicts `(v_pred, ω_pred)`, **overrides ω with gyro-integrated
  ω** (gyro is always more accurate; the model's ω prediction is kept
  for monitoring only), and integrates `(v_pred, ω_actual)` into 2D
  pose `(x, y, θ)`.
* Operator endpoints: `GET /api/motion/pose` (current pose +
  trajectory ring), `POST /api/motion/reset` (zero the integrator).

#### Tests (headless)

* Synthetic command sequence through the integrator; assert pose
  matches analytic solution to within float tolerance.
* Reset clears state.

#### WebUI surface

New **"Pose"** card. This is the first time the operator sees the
robot's belief about its own location — the most important world-model
view we have.

* **Live pose readout:** `x` (m), `y` (m), `θ` (deg).
  Plus: `|v|` (m/s) and `ω` (deg/s) live, with the model's prediction
  shown alongside the gyro measurement for ω (visible disagreement
  flags an unhealthy model).
* **Top-down trajectory plot:** SVG canvas, ~5 m × 5 m bounded
  around the robot's start, robot drawn as a triangle pointing
  along θ, trail of the last ~5 min of poses. Auto-zooms if the
  trajectory exits the default bounds.
* **Drift indicator:** distance from origin; useful sanity check
  on a "drive a square and come home" test.
* **"Reset pose" button** wired to `POST /api/motion/reset`.

Endpoint: `GET /api/motion/pose` returns
`{ x, y, theta, v, omega, trail: [{x,y,t}, …] }`. The trail is the
ring buffer; capped at ~1000 points so the JSON stays small.

#### Stage 3 done when

* "Square test" on the live binary: drive a 1 m × 1 m square (forward
  1 m, in-place quarter turn, ×4) and end within ~25 cm of start. The
  trajectory plot in the WebUI shows a recognisable square (some drift
  OK; gross asymmetry indicates a motor-model bug, not drift). Save
  the trace.
* If the square fails badly, that is data: Stage 1/2 needs more
  ultrasonic-labelled v windows, or the model needs more training
  time on varied commands. Iterate before moving to Stage 4.

### Stage 4 — swept-ultrasonic local scan + greedy exploration controller

Cheapest perception we can build, no ML. Validates that "explore the
room without a global map" is reachable on motor-model pose alone.

At a stationary waypoint:

* Pan the ultrasonic from ~15° to ~165° in 10° steps (~15 readings,
  ~3 s total — servo settle dominates).
* Build a 1D polar scan: `distance_cm[angle]`.
* Identify the "best heading": the angular bin (with some neighbour
  width, e.g. ±20°) whose minimum distance is largest. Tiebreak toward
  current heading to reduce thrash.

Controller (`src/motion/explore.rs`), tick every "waypoint":

1. Swept scan at rest.
2. If max free distance < 50 cm in any direction → STOP, log "boxed in".
3. Else: turn toward best heading (motor model predicts the PWMs and
   duration needed to rotate by the required angle; gyro confirms).
4. Drive forward in a series of short pulses (~0.5 s) while the
   forward ultrasonic clears 30 cm. Stop when within 40 cm of
   anything or after ~1 m travelled (motor model integrates distance).
5. Goto 1.

No global plan, no map of where we have been, no return-to-base — just
"keep going somewhere there is room." Loop closure / coverage tracking
are deferred to Stage 6.

#### Tests (headless)

* Mock polar scan → controller picks the correct best-heading angle.
* Boxed-in scan (all readings < 50 cm) → STOP.
* Mock pose evolution under a synthetic command sequence; controller
  switches to the next waypoint at the expected distance.

#### WebUI surface

Two new things, plus an overlay on the existing pose trajectory.

* **"Scan" card.** Polar plot of the most recent swept scan (SVG;
  half-disc, ~150° wedge). Each ray = one ultrasonic reading;
  length = distance in cm. The chosen "best heading" is highlighted
  (e.g. green wedge); rejected directions (< 30 cm) shown red.
  Numerical readout below: `chosen θ`, `chosen d`, `scan age`.
* **"Explore" card.** Toggle button (`POST /api/motion/explore?on=1`),
  current state string (`scanning` / `turning to 47°` / `driving 0.6 m`
  / `boxed in` / `idle`), and the controller's current target
  heading. Plus an "abort" button that calls `POST /api/motion/stop`.
* **Pose-plot overlay (extends Stage 3 plot).** Plot the latest scan
  as fan-shaped rays emanating from the robot's current pose. The
  operator can then see both "where I am" and "what I see right now"
  in one glance.

Endpoints:

* `GET /api/motion/scan` — latest polar scan + timestamp.
* `GET /api/motion/explore` — controller state + chosen heading.
* `POST /api/motion/explore?on=1|0` — toggle.

#### Stage 4 done when

* **The main milestone of this plan**: robot drives autonomously in
  the living room for ≥ 5 minutes without operator commands, without
  bumping into anything, without the supervisor tripping ultrasonic
  stop, with a trajectory from `/api/motion/pose` that shows
  non-degenerate coverage (visits multiple distinct regions of the
  room, not oscillating in a corner).
* The Scan + Explore cards render and update in real time; the
  trajectory overlay shows the live scan wedge.
* Save the trajectory + scan log as evidence on the PR closing Stage 4.

**If Stage 4 is enough for what we actually want, Stages 5–6 can be
deferred indefinitely.** That is by design: stop here unless dense
mapping is needed.

### Stage 5 — on-device monocular depth predictor

Highest-risk stage. Only enter if Stage 4 lands and dense structure is
genuinely useful (mapping, smarter obstacle avoidance than swept-US
allows, return-to-base).

* **Pick model.** Three candidates, profile first:
  * MiDaS-small INT8 (well-trodden path, broad accuracy).
  * FastDepth (designed for mobile, smallest).
  * DepthAnythingV2-Small (newer, more accurate, heavier).
* **Pick runtime.** Two paths, pick after a one-day spike:
  * Rust-native: the `ort` crate (ONNX Runtime). Cleanest integration,
    requires aarch64 ONNX Runtime libs on the Pi.
  * Python sidecar: a small `python3` process loading the model,
    serving depth over a Unix socket. Less elegant but probably the
    fastest way to validate "does any model run fast enough on the
    Pi" — defer the Rust port until we know the answer.
* **Profile gate.** Before integration, run all three models on a
  static image via a throwaway `cargo run --bin depth-bench`. Pass
  criteria: ≥ 1 Hz wall-clock end-to-end on the Pi, subjectively
  reasonable depth on the three test scenes (corridor, living room,
  blank wall). If no candidate clears 1 Hz, **drop Stage 5
  entirely** — Stage 4 is the product.
* **Metric scale.** Mono-depth output is up to unknown scale. Anchor
  with the ground-plane assumption: known camera height H, fit a
  single scalar gain that makes predicted depth at floor pixels
  consistent with H. Refit once per minute. Persist alongside the
  motor model (`motor-model.toml` is already the runtime-state file
  for this machine; add a `[depth_scale]` table to it).
* **Plumbing.** New module `src/depth/`, mirrors `src/imu/`:
  background thread, latest-result snapshot.

#### Tests (headless)

* Mock depth backend returning a known-shape depth from a known-shape
  image; assert ground-plane scaling produces the expected metric
  depth at floor pixels.
* The `--no-default-features` build uses the mock backend so the
  workspace DoD stays green without ONNX or Python.

#### WebUI surface

New **"Depth"** card placed next to the live camera frame so the
operator can compare them visually:

* **Depth heatmap.** Colour-mapped JPEG from `/api/depth/frame`,
  auto-refreshing at the depth predictor's rate (~1 Hz). Use a
  consistent colour map (e.g. viridis: near = yellow, far = purple)
  with a small legend.
* **Metric scale row:** last-fit scale factor (m / depth-unit),
  fit residual on the ground plane, time since last refit, count
  of floor pixels used in the fit. If the fit fails (e.g. no floor
  visible), surface that with an amber badge.
* **Latency / FPS row:** end-to-end depth-predictor latency,
  achieved FPS, queue depth if any.
* **Same-frame overlay (optional, cheap):** thin crosshair on
  both the camera frame and the depth heatmap with the predicted
  depth at the centre pixel — useful sanity check when driving
  toward a wall.

Endpoints:

* `GET /api/depth/frame?...` — JPEG, optional `w` / `q`.
* `GET /api/depth/info` — scale + latency + FPS.

#### Stage 5 done when

* Pi-side: `/api/depth/frame` returns a depth visualisation at ≥ 1 Hz.
* Pi CPU load with depth + IMU + camera + HTTP under ~80 % (room for
  the supervisor and live driving).
* Subjective sanity: in three scenes, the depth map ranks objects in
  the right order (close > far) and the floor is monotonically
  increasing with image row.

### Stage 6 — dense 2D occupancy grid (depth-driven)

Only if Stage 5 lands.

* `OccupancyGrid` in `src/motion/grid.rs`. 5 cm cells, 8 m × 8 m
  bounded around robot start.
* Each depth frame projects to floor-plane occupancy in the current
  motion-model pose. Bayesian update (log-odds) so a single noisy
  prediction doesn't overwrite the grid.
* Exploration controller (Stage 4) gains an "unvisited frontier"
  heading bias: prefer headings toward grid cells marked unknown.

No loop closure. No global consistency across drives. The grid is for
one drive at a time; drift is acceptable as long as it stays locally
coherent.

#### WebUI surface

The occupancy grid becomes the dominant world-model view, replacing
or augmenting Stage 3's trajectory plot.

* **Top-down grid view.** PNG (server-rendered for cheapness)
  at `/api/motion/grid.png`, ~200×200 px for an 8 m × 8 m grid at
  4 cm/px. Colour: white = free, black = occupied, grey = unknown.
  Robot drawn as a triangle on top (same as Stage 3 plot). Trail
  optional.
* **Toggle overlays.** "Show trajectory" / "Show scan rays" /
  "Show frontier cells" (cells flagged for exploration bias).
  All client-side flags on the same canvas.
* **Grid stats row:** % free, % occupied, % unknown, frontier
  count, last update time.
* **"Clear grid" button** wired to `POST /api/motion/grid/reset`
  for starting a fresh drive without restarting the binary.

Endpoints:

* `GET /api/motion/grid.png` — rendered grid image, optional
  query params for size / overlay flags.
* `GET /api/motion/grid` — raw stats + maybe RLE-compressed cells
  for clients that want to render themselves.
* `POST /api/motion/grid/reset` — clear.

#### Stage 6 done when

* A 10-minute autonomous drive in the living room produces an
  occupancy grid in which the room's gross geometry (walls, large
  furniture footprints) is recognisable to a human looking at the
  WebUI top-down view.

## What we are explicitly dropping or deferring

* **ORB-SLAM as the primary localiser.** Crates and `src/slam` keep
  compiling and their tests keep running. Nothing on the exploration
  path calls into them. The mapper / BA / BoW / loop-closure code is
  preserved for a possible future stereo or RGB-D path; if six months
  go by and that future doesn't materialise, delete it then.
* **Loop closure / global map consistency.** Out of scope. Stage 6's
  grid is robot-frame, single-drive.
* **Per-wheel (4D) motor input.** 2D `(pwm_l, pwm_r)` first. The
  hardware supports addressing all four wheels independently
  (`src/devices/motors.rs:33`), but the chassis is skid-steer and the
  theoretical benefit of front/rear differentiation on a symmetric
  chassis is small. Revisit only if Stage 3's square test shows
  systematic asymmetry that 2D inputs can't capture.
* **Bathroom / non-hardwood floor handling.** The model is fit on one
  surface; tile will break it. Manually keep the robot out of the
  bathroom for now. A second model gated on a "floor regime" detector
  is a possible Stage 7 if it matters.
* **Visual-inertial bundle adjustment** (the old PLAN's "next" item).
  Moot now — we're not using the visual stack for pose.

## Gotchas

* **Servo-sweep cost.** A 10°-step pan from 15° to 165° at typical
  settle times is ~3 s per scan. Acceptable as a per-waypoint cost;
  not acceptable as a continuous loop. The controller in Stage 4 must
  not scan more than once per waypoint.
* **Ultrasonic beamwidth.** HC-SR04 cone is ~15° wide; pretending the
  scan has 1° resolution is wrong. Use 10° bins to match.
* **Motor-model staleness across reboots.** Even with the battery
  voltage check, a model fit with a half-charged battery six hours
  ago can be off. If `/api/motion/model` shows the residual jumping
  after a reload, force a re-init (`POST /api/motion/model/reset`).
* **Depth predictor temporal flicker.** Frame-to-frame depth is
  noisy. The Stage 6 Bayesian update absorbs this; don't write a
  hard-overwrite update.
* **Pi CPU contention.** The IMU thread is already starved (142 Hz
  vs 200 Hz target on the contended I²C bus). Adding a depth
  predictor takes a whole core. Pin threads with explicit affinity
  if Stage 5 evidence shows competing latencies.
* **Camera vs ultrasonic vs depth time-alignment.** Stage 6 will need
  the depth frame's `t_capture` aligned to the pose at that moment.
  `Instant::now()` on `CLOCK_MONOTONIC` is already the project-wide
  convention; keep it.

## Definition of done for the whole plan

The bar is **Stage 4**: the robot drives itself around the living room
for ≥ 5 minutes without operator commands, without collisions, with a
non-degenerate trajectory from `/api/motion/pose`. Stages 5–6 are
upgrades, not requirements.

When Stage 4 passes:

1. Save the trajectory and the swept-US scan log as evidence on the
   merging PR.
2. Update `README.md`'s "SLAM status" section to reflect that
   exploration is now motion-model + swept-US driven, with visual SLAM
   demoted to "available but not on the critical path."
3. Delete this `PLAN.md`. Per `CLAUDE.md`, `PLAN.md` is a migration
   roadmap, not a permanent doc.

## Workflow

Same as the prior plan, unchanged:

* Each stage merges to `main` independently; do not stack stages.
* Test headless first (`cargo test --workspace --no-default-features`),
  deploy via `make deploy`, validate on the live binary.
* Every drive command on the live binary is preceded by
  `curl /api/camera/frame -o /tmp/now.jpg` and a look at the image.
* Drive in short pulses; never let the robot run with no stop command
  cued up.
