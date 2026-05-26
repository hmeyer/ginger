//! Forward motor model: predicts the motion that results from a PWM
//! command + chassis state over a fixed 200 ms label window.
//!
//! ## Why forward, not inverse?
//!
//! The previous inverse model (`(v_target, ω_target) → (pwm_l, pwm_r)`)
//! had a self-reinforcing feedback loop: at training time, labels paired
//! each observed PWM with its measured motion as the input target; but
//! at inference time, the model's own (possibly bad) predictions were
//! what got sent to the motors, so it trained against its own output
//! distribution. Once the model's predicted PWMs collapsed below the
//! motor deadband, the chassis stopped moving, every fresh label became
//! `(tiny pwms, no motion)`, and the loop locked the model into "output
//! ~0 for every target." The session that drove this rewrite watched
//! that exact collapse happen within ~30 s of live joystick driving.
//!
//! The forward direction has no such loop. Labels pair PWMs *that were
//! actually sent* with the motion the BNO055 *actually measured*. Both
//! sides are ground truth, observed independently of the model. The
//! driving path is now a pure-math arcade-drive mapping (see
//! [`crate::motion::arcade_drive`]); the forward model is read out for
//! diagnostics, the explore controller's planning, and pose-integrator
//! gain checks — but never on the critical drive path.
//!
//! ## Architecture
//!
//! A 7 → 16 → 16 → 2 fully-connected MLP, tanh on hidden layers, linear
//! output. ~430 parameters total.
//!
//! Inputs:
//! * `pwm_l, pwm_r` — commanded PWMs (current window).
//! * `pwm_l_prev, pwm_r_prev` — previous-window PWMs (captures inertia
//!   and Δcommand effects).
//! * `v_prev, ω_prev` — best estimate of previous-window motion.
//! * `battery_v` — voltage now (lets the model learn voltage droop).
//!
//! Outputs (one fixed window, [`WINDOW_S`] = 0.2 s):
//! * `Δs_m` — forward displacement in robot frame (m).
//! * `Δθ_rad` — rotation about chassis-vertical (rad). The chassis is
//!   left-right symmetric so labels are mirrored during training (see
//!   [`mirror_sample`]).
//!
//! Trained online via Adam-lite SGD with MSE loss. The Δθ channel is
//! always labelled (BNO055 quaternion delta); the Δs channel is missing
//! whenever the ultrasonic Δd/Δt didn't yield a real measurement (most
//! windows in an open room), and its loss is then zeroed — see
//! [`LabelledSample::ds_obs_m`].
//!
//! ## Persistence
//!
//! `motor-model.toml` at the repo root, gitignored (machine-local). The
//! battery-voltage staleness gate is preserved from the prior design.
//! [`SCHEMA_VERSION`] bumped to **2** — the file shape is incompatible
//! with the inverse-model schema; on load the gate triggers a fresh
//! bootstrap if an older file is encountered.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use log::{info, warn};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::hal::pca9685::MAX_DUTY;

// ── Architecture ──────────────────────────────────────────────────────────────

pub const INPUT_DIM: usize = 7;
pub const HIDDEN_1: usize = 16;
pub const HIDDEN_2: usize = 16;
pub const OUTPUT_DIM: usize = 2;
/// Bumped from 1: the on-disk shape (input/output semantics, scale
/// constants) is incompatible with the inverse-model schema.
pub const SCHEMA_VERSION: u32 = 2;

/// Label-window duration. Fixed across the worker, the bootstrap, and
/// the model output's interpretation: the predicted (`Δs_m`, `Δθ_rad`)
/// is the motion across exactly this much wall time.
pub const WINDOW_S: f32 = 0.2;

// Normalizer defaults. The forward output channels need their own
// scales — chosen so a "full pulse" lands roughly at ±1 in normalised
// space, which keeps the tanh-hidden network in its linear regime.
const NORM_PWM_SCALE: f32 = MAX_DUTY as f32; // 4095
const NORM_V_SCALE: f32 = 1.0; // m/s — reasonable indoor max
const NORM_OMEGA_SCALE: f32 = 2.0; // rad/s
const NORM_BATT_MEAN: f32 = 7.8; // V — middle of a 2S LiPo's useful range
const NORM_BATT_SCALE: f32 = 0.6; // V — half-span
/// Max forward displacement per [`WINDOW_S`]: ~1 m/s × 0.2 s = 0.2 m.
const NORM_DS_SCALE: f32 = 0.2;
/// Max rotation per [`WINDOW_S`]: ~2 rad/s × 0.2 s = 0.4 rad.
const NORM_DTHETA_SCALE: f32 = 0.4;

// Optimiser (Adam-lite, no bias correction).
const ADAM_LR: f32 = 1e-3;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.99;
const ADAM_EPS: f32 = 1e-8;

// Bootstrap.
const BOOTSTRAP_SEED: u64 = 0x00C0_FFEE_0000_0002;
const BOOTSTRAP_SAMPLES: usize = 2000;
const BOOTSTRAP_STEPS: usize = 2000;

// Battery staleness gate.
pub const BATTERY_STALE_THRESHOLD_V: f32 = 0.3;

// Running-residual EWMA factor (per-sample). 0.02 → ~50-sample horizon.
const RESIDUAL_ALPHA: f32 = 0.02;

/// Loss weight applied to the Δs channel when no ultrasonic v_meas was
/// available for the window. The Δθ channel is always trusted.
const DS_MISSING_WEIGHT: f32 = 0.0;

// ── Public types ──────────────────────────────────────────────────────────────

/// Inputs to the forward model. All in physical / SI units; the
/// normaliser handles scaling internally.
#[derive(Debug, Clone, Copy)]
pub struct ModelInput {
    /// Current-window commanded PWMs (PCA9685 duty in
    /// `[-MAX_DUTY, MAX_DUTY]`). Positive = forward on this chassis.
    pub pwm_l: i32,
    pub pwm_r: i32,
    /// Previous-window PWMs. Captures inertia and Δcommand effects —
    /// the chassis decelerates differently from a fresh-from-zero pulse.
    pub pwm_l_prev: i32,
    pub pwm_r_prev: i32,
    /// Best estimate of the previous-window forward velocity (m/s).
    /// Ultrasonic-derived when the prior window had a valid Δd/Δt, else
    /// the previous prediction's `Δs_m / WINDOW_S` (closed-loop estimate).
    pub v_prev: f32,
    /// Previous-window angular velocity (rad/s), from BNO055 fusion.
    pub omega_prev: f32,
    /// Battery voltage at this tick (V). Lets the model learn voltage
    /// droop directly.
    pub battery_v: f32,
}

/// Model output. The motion the chassis is expected to make over the
/// next [`WINDOW_S`] seconds, in the robot frame at the window's start.
#[derive(Debug, Clone, Copy)]
pub struct MotionPrediction {
    pub ds_m: f32,
    pub dtheta_rad: f32,
}

/// One training window assembled by `src/motion/labels.rs`.
///
/// * `pwm_l/r_obs` — the actual PWMs commanded during the window
///   (mirrors `ModelInput::pwm_l/r` of the same sample).
/// * `ds_obs_m` — measured forward displacement, **only when** the
///   ultrasonic Δd/Δt window passed the straight + monotonic + in-range
///   gate. `None` otherwise — the Δs loss is then zeroed (gated by
///   [`DS_MISSING_WEIGHT`]) and only Δθ is trained.
/// * `dtheta_obs_rad` — always populated from the BNO055 fusion quaternion
///   delta across the window.
#[derive(Debug, Clone, Copy)]
pub struct LabelledSample {
    pub input: ModelInput,
    pub ds_obs_m: Option<f32>,
    pub dtheta_obs_rad: f32,
    pub dt_s: f32,
}

// ── Normaliser ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalizer {
    pwm_scale: f32,
    v_scale: f32,
    omega_scale: f32,
    battery_mean: f32,
    battery_scale: f32,
    ds_scale: f32,
    dtheta_scale: f32,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self {
            pwm_scale: NORM_PWM_SCALE,
            v_scale: NORM_V_SCALE,
            omega_scale: NORM_OMEGA_SCALE,
            battery_mean: NORM_BATT_MEAN,
            battery_scale: NORM_BATT_SCALE,
            ds_scale: NORM_DS_SCALE,
            dtheta_scale: NORM_DTHETA_SCALE,
        }
    }
}

impl Normalizer {
    fn encode(&self, x: &ModelInput) -> [f32; INPUT_DIM] {
        [
            x.pwm_l as f32 / self.pwm_scale,
            x.pwm_r as f32 / self.pwm_scale,
            x.pwm_l_prev as f32 / self.pwm_scale,
            x.pwm_r_prev as f32 / self.pwm_scale,
            x.v_prev / self.v_scale,
            x.omega_prev / self.omega_scale,
            (x.battery_v - self.battery_mean) / self.battery_scale,
        ]
    }

    fn decode_motion(&self, y: [f32; OUTPUT_DIM]) -> MotionPrediction {
        MotionPrediction {
            ds_m: y[0] * self.ds_scale,
            dtheta_rad: y[1] * self.dtheta_scale,
        }
    }

    /// Encode a measured `(Δs, Δθ)` label into normalised target space.
    /// Saturates inputs that fall outside `[-scale, +scale]` to keep
    /// loss gradients sane.
    fn encode_motion(&self, ds_m: f32, dtheta_rad: f32) -> [f32; OUTPUT_DIM] {
        [
            (ds_m / self.ds_scale).clamp(-1.0, 1.0),
            (dtheta_rad / self.dtheta_scale).clamp(-1.0, 1.0),
        ]
    }
}

// ── Layer ────────────────────────────────────────────────────────────────────

/// A single fully-connected layer with its Adam state. Adam moments are
/// not persisted (`#[serde(skip)]`); they reinitialise to zero on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    #[serde(skip)]
    w_m: Vec<Vec<f32>>,
    #[serde(skip)]
    w_v: Vec<Vec<f32>>,
    #[serde(skip)]
    b_m: Vec<f32>,
    #[serde(skip)]
    b_v: Vec<f32>,
}

impl Layer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut SmallRng) -> Self {
        let bound = (6.0 / (in_dim + out_dim) as f32).sqrt();
        let weights: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| rng.random_range(-bound..bound))
                    .collect()
            })
            .collect();
        Self {
            weights,
            bias: vec![0.0; out_dim],
            w_m: vec![vec![0.0; in_dim]; out_dim],
            w_v: vec![vec![0.0; in_dim]; out_dim],
            b_m: vec![0.0; out_dim],
            b_v: vec![0.0; out_dim],
        }
    }

    fn reset_adam_state(&mut self) {
        let out_dim = self.weights.len();
        let in_dim = self.weights.first().map(|r| r.len()).unwrap_or(0);
        self.w_m = vec![vec![0.0; in_dim]; out_dim];
        self.w_v = vec![vec![0.0; in_dim]; out_dim];
        self.b_m = vec![0.0; out_dim];
        self.b_v = vec![0.0; out_dim];
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let out_dim = self.weights.len();
        let mut y = Vec::with_capacity(out_dim);
        for i in 0..out_dim {
            let row = &self.weights[i];
            let mut acc = self.bias[i];
            for j in 0..row.len() {
                acc += row[j] * x[j];
            }
            y.push(acc);
        }
        y
    }

    fn adam_step(&mut self, dw: &[Vec<f32>], db: &[f32]) {
        for (i, dw_row) in dw.iter().enumerate() {
            for (j, &g) in dw_row.iter().enumerate() {
                self.w_m[i][j] = ADAM_BETA1 * self.w_m[i][j] + (1.0 - ADAM_BETA1) * g;
                self.w_v[i][j] = ADAM_BETA2 * self.w_v[i][j] + (1.0 - ADAM_BETA2) * g * g;
                self.weights[i][j] -= ADAM_LR * self.w_m[i][j] / (self.w_v[i][j].sqrt() + ADAM_EPS);
            }
            let g = db[i];
            self.b_m[i] = ADAM_BETA1 * self.b_m[i] + (1.0 - ADAM_BETA1) * g;
            self.b_v[i] = ADAM_BETA2 * self.b_v[i] + (1.0 - ADAM_BETA2) * g * g;
            self.bias[i] -= ADAM_LR * self.b_m[i] / (self.b_v[i].sqrt() + ADAM_EPS);
        }
    }
}

// ── Persistence schema ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Meta {
    schema_version: u32,
    trained_steps: u64,
    last_battery_v: f32,
    last_updated_unix: i64,
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── MotorModel ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorModel {
    meta: Meta,
    normalizer: Normalizer,
    layer_1: Layer,
    layer_2: Layer,
    layer_3: Layer,

    /// EWMA of per-sample motion-prediction residual (Euclidean norm in
    /// normalised output space). Useful telemetry; not persisted.
    #[serde(skip, default = "default_residual")]
    residual_motion: f32,
}

fn default_residual() -> f32 {
    0.0
}

impl MotorModel {
    /// Freshly bootstrap from synthetic forward dynamics. Deterministic
    /// in [`BOOTSTRAP_SEED`].
    pub fn default_bootstrap(battery_v_now: f32) -> Self {
        let mut rng = SmallRng::seed_from_u64(BOOTSTRAP_SEED);
        let mut model = Self {
            meta: Meta {
                schema_version: SCHEMA_VERSION,
                trained_steps: 0,
                last_battery_v: battery_v_now,
                last_updated_unix: now_unix(),
            },
            normalizer: Normalizer::default(),
            layer_1: Layer::new(INPUT_DIM, HIDDEN_1, &mut rng),
            layer_2: Layer::new(HIDDEN_1, HIDDEN_2, &mut rng),
            layer_3: Layer::new(HIDDEN_2, OUTPUT_DIM, &mut rng),
            residual_motion: 0.0,
        };
        let samples = synthetic_forward_samples(&mut rng, BOOTSTRAP_SAMPLES);
        for _ in 0..BOOTSTRAP_STEPS {
            let i = rng.random_range(0..samples.len());
            model.observe(samples[i]);
        }
        model.meta.last_battery_v = battery_v_now;
        model.meta.trained_steps = BOOTSTRAP_STEPS as u64;
        model
    }

    /// Load from disk if present and non-stale, else bootstrap. A
    /// schema-version mismatch — including any file written by the
    /// previous inverse-model schema — re-bootstraps.
    pub fn load_or_bootstrap(path: &Path, battery_v_now: f32) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => match toml::from_str::<MotorModel>(&s) {
                Ok(mut m) => {
                    if m.meta.schema_version != SCHEMA_VERSION {
                        warn!(
                            "motor-model: schema {} != current {}; re-bootstrapping",
                            m.meta.schema_version, SCHEMA_VERSION
                        );
                        return Self::default_bootstrap(battery_v_now);
                    }
                    let drift = (m.meta.last_battery_v - battery_v_now).abs();
                    if drift > BATTERY_STALE_THRESHOLD_V {
                        warn!(
                            "motor-model: stale ({:.2} V saved vs {:.2} V now, > {:.2}); \
                             re-bootstrapping",
                            m.meta.last_battery_v, battery_v_now, BATTERY_STALE_THRESHOLD_V
                        );
                        return Self::default_bootstrap(battery_v_now);
                    }
                    m.layer_1.reset_adam_state();
                    m.layer_2.reset_adam_state();
                    m.layer_3.reset_adam_state();
                    info!(
                        "motor-model: loaded {} steps trained, last_battery={:.2}V",
                        m.meta.trained_steps, m.meta.last_battery_v
                    );
                    m
                }
                Err(e) => {
                    warn!("motor-model: unparseable ({e}); re-bootstrapping");
                    Self::default_bootstrap(battery_v_now)
                }
            },
            Err(_) => {
                info!("motor-model: no file at {}; bootstrapping", path.display());
                Self::default_bootstrap(battery_v_now)
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut s = self.clone();
        s.meta.last_updated_unix = now_unix();
        let body = toml::to_string_pretty(&s)
            .map_err(|e| crate::Error::Motion(format!("serialize: {e}")))?;
        std::fs::write(path, body)?;
        Ok(())
    }

    pub fn trained_steps(&self) -> u64 {
        self.meta.trained_steps
    }

    pub fn last_battery_v(&self) -> f32 {
        self.meta.last_battery_v
    }

    pub fn last_updated_unix(&self) -> i64 {
        self.meta.last_updated_unix
    }

    /// EWMA of recent prediction error in normalised output units.
    /// Roughly: 0.0 = predictions match labels, 1.0 = full-scale miss.
    pub fn residual_motion(&self) -> f32 {
        self.residual_motion
    }

    /// Forward inference: PWMs + state → predicted (Δs, Δθ) over the
    /// next [`WINDOW_S`] seconds.
    pub fn predict(&self, x: ModelInput) -> MotionPrediction {
        let z0 = self.normalizer.encode(&x);
        let a1 = self.layer_1.forward(&z0);
        let h1: Vec<f32> = a1.iter().map(|v| v.tanh()).collect();
        let a2 = self.layer_2.forward(&h1);
        let h2: Vec<f32> = a2.iter().map(|v| v.tanh()).collect();
        let a3 = self.layer_3.forward(&h2);
        self.normalizer.decode_motion([a3[0], a3[1]])
    }

    /// One labelled window → two SGD steps (sample + left/right mirror;
    /// the chassis is mechanically symmetric so each label is also a
    /// mirror-axis label, which doubles the data and forces the learned
    /// mapping to stay L/R-symmetric — important when one direction of
    /// turning happens to dominate the recent driving distribution).
    pub fn observe(&mut self, s: LabelledSample) {
        self.step(&s);
        let mirrored = mirror_sample(&s);
        self.step(&mirrored);
        self.meta.trained_steps = self.meta.trained_steps.saturating_add(1);
        self.meta.last_battery_v = s.input.battery_v;
    }

    fn step(&mut self, s: &LabelledSample) {
        let z0 = self.normalizer.encode(&s.input);
        let a1 = self.layer_1.forward(&z0);
        let h1: Vec<f32> = a1.iter().map(|v| v.tanh()).collect();
        let a2 = self.layer_2.forward(&h1);
        let h2: Vec<f32> = a2.iter().map(|v| v.tanh()).collect();
        let a3 = self.layer_3.forward(&h2);

        // Build the normalised target. The Δs target is *known* only
        // when the labeller actually observed it; otherwise we set the
        // ds-channel loss weight to zero so the model is trained only
        // on Δθ this step.
        let ds_label = s.ds_obs_m.unwrap_or(0.0);
        let target = self.normalizer.encode_motion(ds_label, s.dtheta_obs_rad);
        let ds_weight = if s.ds_obs_m.is_some() {
            1.0
        } else {
            DS_MISSING_WEIGHT
        };

        // dL/da3 — MSE with per-channel weight. Factor of 2 absorbed into LR.
        let d_a3: [f32; OUTPUT_DIM] = [ds_weight * (a3[0] - target[0]), (a3[1] - target[1])];

        // Residual in normalised output space, Euclidean norm.
        let r2: f32 = d_a3.iter().map(|&v| v * v).sum::<f32>();
        let r = (r2 / OUTPUT_DIM as f32).sqrt();
        self.residual_motion = (1.0 - RESIDUAL_ALPHA) * self.residual_motion + RESIDUAL_ALPHA * r;

        let (dw3, db3, d_h2) = backprop_linear(&self.layer_3.weights, &d_a3, &h2);
        let d_a2: Vec<f32> = (0..HIDDEN_2)
            .map(|i| d_h2[i] * (1.0 - h2[i] * h2[i]))
            .collect();
        let (dw2, db2, d_h1) = backprop_linear(&self.layer_2.weights, &d_a2, &h1);
        let d_a1: Vec<f32> = (0..HIDDEN_1)
            .map(|i| d_h1[i] * (1.0 - h1[i] * h1[i]))
            .collect();
        let (dw1, db1, _d_z0) = backprop_linear(&self.layer_1.weights, &d_a1, &z0);

        self.layer_1.adam_step(&dw1, &db1);
        self.layer_2.adam_step(&dw2, &db2);
        self.layer_3.adam_step(&dw3, &db3);
    }
}

/// Mirror a labelled window across the chassis's left/right axis: swap
/// `pwm_l`/`pwm_r` (current + previous), negate `omega_prev` and the
/// observed Δθ. Δs (forward) and `v_prev` don't have a left/right sign
/// and stay put. The chassis kinematics are mechanically symmetric so
/// this is a genuinely valid second sample.
fn mirror_sample(s: &LabelledSample) -> LabelledSample {
    LabelledSample {
        input: ModelInput {
            pwm_l: s.input.pwm_r,
            pwm_r: s.input.pwm_l,
            pwm_l_prev: s.input.pwm_r_prev,
            pwm_r_prev: s.input.pwm_l_prev,
            v_prev: s.input.v_prev,
            omega_prev: -s.input.omega_prev,
            battery_v: s.input.battery_v,
        },
        ds_obs_m: s.ds_obs_m,
        dtheta_obs_rad: -s.dtheta_obs_rad,
        dt_s: s.dt_s,
    }
}

/// Backprop through one fully-connected layer with grad-of-output `d_y`
/// and input `x_in`. Returns `(dW, db, dX_in)`.
fn backprop_linear(
    weights: &[Vec<f32>],
    d_y: &[f32],
    x_in: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let out_dim = weights.len();
    let in_dim = weights[0].len();
    let mut dw = vec![vec![0.0_f32; in_dim]; out_dim];
    let mut db = vec![0.0_f32; out_dim];
    let mut dx = vec![0.0_f32; in_dim];
    for i in 0..out_dim {
        db[i] = d_y[i];
        for j in 0..in_dim {
            dw[i][j] = d_y[i] * x_in[j];
            dx[j] += weights[i][j] * d_y[i];
        }
    }
    (dw, db, dx)
}

// ── Synthetic forward prior (used only for bootstrap) ─────────────────────────

/// Hand-coded forward dynamics, **wiring-quirk-aware**: on this chassis
/// `pwm_l > pwm_r` produces physical CCW = positive Δθ (chassis convention).
/// The synthetic prior bakes this in so a fresh bootstrap is already in
/// the correct sign convention without waiting for labels to flip it.
///
/// Per-window magnitudes calibrated from live observation:
/// * Forward sensitivity: PWM 1500 ≈ 0.6 m/s ≈ 0.12 m / 200 ms window.
/// * Turning sensitivity: PWM diff 1400 ≈ 0.26 rad/s ≈ 0.052 rad / window.
///   → K_dtheta ≈ 3.78e-5 rad per (PWM-unit · per 200 ms window).
fn synthetic_forward(pwm_l: i32, pwm_r: i32) -> (f32, f32) {
    let pwm_avg = (pwm_l + pwm_r) as f32 * 0.5;
    let pwm_diff = (pwm_l - pwm_r) as f32; // wiring-correct sign: l > r → CCW positive
    let dead = 200.0;
    let v_mps = if pwm_avg.abs() < dead {
        0.0
    } else {
        pwm_avg.signum() * (pwm_avg.abs() - dead) * 4.6e-4
    };
    let ds_m = v_mps * WINDOW_S;
    let dtheta_rad = pwm_diff * 3.78e-5;
    (ds_m, dtheta_rad)
}

/// Generate forward training samples for bootstrap: pick a random PWM
/// pair, compute the synthetic (Δs, Δθ), package as a labelled sample.
/// Both Δs and Δθ are "present" in synthetic samples — bootstrap trains
/// against a complete distribution.
fn synthetic_forward_samples(rng: &mut SmallRng, n: usize) -> Vec<LabelledSample> {
    let max = MAX_DUTY as i32;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let pwm_l = rng.random_range(-max..=max);
        let pwm_r = rng.random_range(-max..=max);
        let (ds_m, dtheta_rad) = synthetic_forward(pwm_l, pwm_r);
        let battery_v = rng.random_range(7.2..8.4);
        out.push(LabelledSample {
            input: ModelInput {
                pwm_l,
                pwm_r,
                pwm_l_prev: 0,
                pwm_r_prev: 0,
                v_prev: 0.0,
                omega_prev: 0.0,
                battery_v,
            },
            ds_obs_m: Some(ds_m),
            dtheta_obs_rad: dtheta_rad,
            dt_s: WINDOW_S,
        });
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn input(pwm_l: i32, pwm_r: i32, battery_v: f32) -> ModelInput {
        ModelInput {
            pwm_l,
            pwm_r,
            pwm_l_prev: 0,
            pwm_r_prev: 0,
            v_prev: 0.0,
            omega_prev: 0.0,
            battery_v,
        }
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let a = MotorModel::default_bootstrap(7.8);
        let b = MotorModel::default_bootstrap(7.8);
        assert_eq!(a.layer_1.weights, b.layer_1.weights);
        assert_eq!(a.layer_2.weights, b.layer_2.weights);
        assert_eq!(a.layer_3.weights, b.layer_3.weights);
        for l in [-2000, 0, 2000] {
            for r in [-2000, 0, 2000] {
                let pa = a.predict(input(l, r, 7.8));
                let pb = b.predict(input(l, r, 7.8));
                assert_eq!(pa.ds_m, pb.ds_m);
                assert_eq!(pa.dtheta_rad, pb.dtheta_rad);
            }
        }
    }

    #[test]
    fn persistence_round_trip() {
        let m = MotorModel::default_bootstrap(7.8);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let toml_str = toml::to_string_pretty(&m).unwrap();
        f.write_all(toml_str.as_bytes()).unwrap();
        f.flush().unwrap();
        let loaded = MotorModel::load_or_bootstrap(f.path(), 7.8);
        assert_eq!(loaded.meta.trained_steps, m.meta.trained_steps);
        assert_eq!(loaded.layer_1.weights, m.layer_1.weights);
        for l in [-1500, 0, 1500] {
            for r in [-1500, 0, 1500] {
                let pa = m.predict(input(l, r, 7.8));
                let pb = loaded.predict(input(l, r, 7.8));
                assert_eq!(pa.ds_m, pb.ds_m);
                assert_eq!(pa.dtheta_rad, pb.dtheta_rad);
            }
        }
    }

    #[test]
    fn battery_staleness_triggers_rebootstrap() {
        let m = MotorModel::default_bootstrap(7.2);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let toml_str = toml::to_string_pretty(&m).unwrap();
        f.write_all(toml_str.as_bytes()).unwrap();
        f.flush().unwrap();
        let loaded = MotorModel::load_or_bootstrap(f.path(), 8.4);
        // 1.2 V drift past the 0.3 V threshold → fresh bootstrap (with
        // last_battery_v anchored on the new voltage).
        assert!((loaded.meta.last_battery_v - 8.4).abs() < 1e-3);
    }

    /// Schema-version mismatch (e.g. an old inverse-model file) must
    /// trigger a re-bootstrap rather than crash.
    #[test]
    fn schema_version_mismatch_rebootstraps() {
        let mut m = MotorModel::default_bootstrap(7.8);
        m.meta.schema_version = 1; // pretend an old inverse-model file
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let toml_str = toml::to_string_pretty(&m).unwrap();
        f.write_all(toml_str.as_bytes()).unwrap();
        f.flush().unwrap();
        let loaded = MotorModel::load_or_bootstrap(f.path(), 7.8);
        assert_eq!(loaded.meta.schema_version, SCHEMA_VERSION);
    }

    /// The bootstrap should already match the chassis's wiring quirk:
    /// `pwm_l > pwm_r` drives CCW (positive Δθ) and a centred forward
    /// PWM drives positive Δs. This is the "do something sensible
    /// before any labels arrive" check.
    #[test]
    fn bootstrap_signs_match_wiring() {
        let m = MotorModel::default_bootstrap(7.8);
        // Pure forward at firm PWM → positive Δs, ~zero Δθ.
        let p = m.predict(input(1500, 1500, 7.8));
        assert!(p.ds_m > 0.0, "expected forward ds_m > 0, got {}", p.ds_m);
        assert!(
            p.dtheta_rad.abs() < 0.05,
            "expected near-zero dθ, got {}",
            p.dtheta_rad
        );
        // Pure backward → negative Δs.
        let p = m.predict(input(-1500, -1500, 7.8));
        assert!(p.ds_m < 0.0, "expected backward ds_m < 0, got {}", p.ds_m);
        // Pure CCW spin (pwm_l > pwm_r) → positive Δθ.
        let p = m.predict(input(1500, -1500, 7.8));
        assert!(
            p.dtheta_rad > 0.0,
            "expected CCW dθ > 0, got {}",
            p.dtheta_rad
        );
        // Pure CW spin → negative Δθ.
        let p = m.predict(input(-1500, 1500, 7.8));
        assert!(
            p.dtheta_rad < 0.0,
            "expected CW dθ < 0, got {}",
            p.dtheta_rad
        );
    }

    /// Training on synthetic forward labels should land predictions
    /// close to those labels (low residual).
    #[test]
    fn trains_on_synthetic_forward_labels() {
        let mut m = MotorModel::default_bootstrap(7.8);
        // Bootstrap already trained 2000 steps; add 1000 more on a
        // fresh draw and confirm residual is small.
        let mut rng = SmallRng::seed_from_u64(0xCAFE_F00D);
        let samples = synthetic_forward_samples(&mut rng, 1000);
        for s in &samples {
            m.observe(*s);
        }
        // After 3000 SGD steps the residual on this synthetic
        // distribution should be well under 0.05 (~ 1 cm Δs error,
        // 0.02 rad Δθ error).
        assert!(
            m.residual_motion() < 0.05,
            "residual too high: {}",
            m.residual_motion()
        );
    }

    /// Δs-missing samples should only train the Δθ channel — the Δs
    /// loss weight is zero. Test: train on (random PWM, no Δs, Δθ=0)
    /// many times; Δθ predictions should converge to ~0, Δs predictions
    /// should stay roughly at their pre-training values (bootstrap state).
    #[test]
    fn ds_missing_samples_only_train_dtheta() {
        let mut m = MotorModel::default_bootstrap(7.8);
        let pwm_l = 1500;
        let pwm_r = 1500;
        let pre = m.predict(input(pwm_l, pwm_r, 7.8));

        // Train 500 times on the same (pwm_l=pwm_r=1500, dθ=0, no ds).
        // dθ should converge to 0; ds should not move much.
        for _ in 0..500 {
            m.observe(LabelledSample {
                input: input(pwm_l, pwm_r, 7.8),
                ds_obs_m: None,
                dtheta_obs_rad: 0.0,
                dt_s: WINDOW_S,
            });
        }
        let post = m.predict(input(pwm_l, pwm_r, 7.8));
        assert!(
            post.dtheta_rad.abs() < 0.005,
            "dθ should have trained to ~0, got {}",
            post.dtheta_rad
        );
        // Δs preserved within a tight tolerance.
        let ds_drift = (post.ds_m - pre.ds_m).abs();
        assert!(
            ds_drift < 0.02,
            "ds drifted by {ds_drift} despite missing-ds samples"
        );
    }

    /// Mirror sample property: swapping L/R PWMs and negating ω/Δθ
    /// produces a sample the model should predict identically up to
    /// reflection. After symmetric training this is a structural
    /// invariant — verify it directly on the bootstrap-only model.
    #[test]
    fn mirror_predictions_are_symmetric_after_bootstrap() {
        let m = MotorModel::default_bootstrap(7.8);
        let cases = [(1500, 500), (-1200, 800), (1000, -1000), (500, 1500)];
        for (l, r) in cases {
            let p = m.predict(input(l, r, 7.8));
            let p_mirror = m.predict(input(r, l, 7.8));
            // Δs symmetric (L/R-flip preserves forward).
            assert!(
                (p.ds_m - p_mirror.ds_m).abs() < 0.005,
                "Δs not symmetric for ({l},{r}): {} vs {}",
                p.ds_m,
                p_mirror.ds_m
            );
            // Δθ flips sign.
            assert!(
                (p.dtheta_rad + p_mirror.dtheta_rad).abs() < 0.01,
                "Δθ not anti-symmetric for ({l},{r}): {} vs {}",
                p.dtheta_rad,
                p_mirror.dtheta_rad
            );
        }
    }
}
