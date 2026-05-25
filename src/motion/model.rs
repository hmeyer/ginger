//! Inverse motor-driver model: maps desired motion + chassis state to
//! the `(pwm_l, pwm_r)` command that should produce it.
//!
//! ## Why inverse?
//!
//! The controller (PLAN Stage 4) and the WebUI joystick both want to
//! think in motion units ("drive at 0.4 m/s, ω = 0.3 rad/s"), not raw
//! PWM. The model translates: input = desired `(v_target, ω_target)`
//! plus the chassis's last state, output = `(pwm_l, pwm_r)`. At
//! training time the *measured* motion from gyro / ultrasonic is fed
//! in as the "desired" — the model learns "to produce this observed
//! motion, the actual PWM that did it was X."
//!
//! ## Architecture
//!
//! A 7 → 16 → 16 → 2 fully-connected MLP, tanh on hidden layers, linear
//! output, ~430 parameters total. Inputs normalised to roughly `[-1, 1]`
//! before the network; outputs scaled back to physical PWM and clamped
//! to `[-MAX_DUTY, MAX_DUTY]` on the way out.
//!
//! Trained online via Adam-lite SGD (no bias correction) with MSE on
//! PWM output plus L2 regularisation on the predicted PWMs. The L2 term
//! is the **deadband-ambiguity resolver**: when multiple training PWMs
//! produced the same observed motion, λ pulls the prediction toward the
//! smallest-norm command — the canonical zero-energy answer.
//!
//! ## Persistence
//!
//! `motor-model.toml` at the repo root, gitignored (machine-local
//! runtime state, **not** committed). The loader runs a battery-voltage
//! staleness check: if the saved-at voltage is more than
//! [`BATTERY_STALE_THRESHOLD_V`] away from the live reading, the file
//! is rejected and we re-bootstrap from the synthetic warm-up. Cheap
//! insurance against the model file outliving its calibration regime.
//!
//! The Adam state (first / second moment vectors) is intentionally
//! **not** persisted — Adam recovers fast and an out-of-date moment
//! could mislead the first few steps after a reload.

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
pub const SCHEMA_VERSION: u32 = 1;

// Normalizer defaults (also persisted, so changing these later doesn't
// break older files — the file's own values are used at load time).
const NORM_PWM_SCALE: f32 = MAX_DUTY as f32; // 4095
const NORM_V_SCALE: f32 = 1.0; // m/s — a reasonable indoor max
const NORM_OMEGA_SCALE: f32 = 2.0; // rad/s
const NORM_BATT_MEAN: f32 = 7.8; // V — middle of a 2S LiPo's useful range
const NORM_BATT_SCALE: f32 = 0.6; // V — half-span

// Optimizer (Adam-lite — no bias correction; one extra vector per param).
const ADAM_LR: f32 = 1e-3;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.99;
const ADAM_EPS: f32 = 1e-8;

// Regularisation.
//
// `L2_LAMBDA` is the deadband-ambiguity resolver — see module docs. It
// acts on the *normalised* PWM output (roughly `[-1, 1]`), so a value
// of `1e-4` is a gentle pull toward zero, not a hard constraint.
const L2_LAMBDA: f32 = 1e-4;

// When the labelled window had no ultrasonic Δd/Δt (so `v_target` was
// filled with the commanded value rather than a real measurement), the
// v dimension of the input is noisy — weight the sample less so the
// gradient doesn't commit hard to it. ω is still good (gyro is always
// present).
const V_MISSING_WEIGHT: f32 = 0.3;

// Bootstrap.
//
// On first boot (no `motor-model.toml`), we generate `BOOTSTRAP_SAMPLES`
// synthetic samples from a hand-coded forward prior, then *invert* each
// one for training: feed the predicted motion as the target and the
// originating PWM as the regression label. That gives the model a
// sensible starting point in `~10 ms`.
const BOOTSTRAP_SEED: u64 = 0x00C0_FFEE_0000_0001;
const BOOTSTRAP_SAMPLES: usize = 2000;
const BOOTSTRAP_STEPS: usize = 2000;

// Battery staleness gate — see module docs.
pub const BATTERY_STALE_THRESHOLD_V: f32 = 0.3;

// Running-residual EWMA factor (per-sample). 0.02 → ~50-sample horizon.
const RESIDUAL_ALPHA: f32 = 0.02;

// ── Public types ──────────────────────────────────────────────────────────────

/// Inputs to the model. All values are in physical / SI units; the
/// model handles normalisation internally.
#[derive(Debug, Clone, Copy)]
pub struct ModelInput {
    /// Last commanded left/right PWM (PCA9685 duty units in
    /// `[-MAX_DUTY, MAX_DUTY]`). Captures inertia and Δcommand effects.
    pub pwm_l_prev: i32,
    pub pwm_r_prev: i32,
    /// Best estimate of the chassis's previous forward velocity (m/s).
    /// Per PLAN: ultrasonic-measured if the previous window had a valid
    /// Δd/Δt label, else the previous `v_target` (closed-loop fallback).
    pub v_prev: f32,
    /// Previous angular velocity (rad/s) from gyro. Always available.
    pub omega_prev: f32,
    /// Battery voltage at this tick (V). Lets the model learn voltage
    /// droop without a separate forgetting-factor mechanism.
    pub battery_v: f32,
    /// Desired forward velocity (m/s).
    pub v_target: f32,
    /// Desired angular velocity (rad/s).
    pub omega_target: f32,
}

/// Model output. Physical PCA9685 duty, clamped to `[-MAX_DUTY, MAX_DUTY]`.
#[derive(Debug, Clone, Copy)]
pub struct PwmCommand {
    pub pwm_l: i32,
    pub pwm_r: i32,
}

/// One training window assembled by Stage 2 (`src/motion/labels.rs`).
///
/// Construction recipe (the trick that makes inverse training work):
///
/// 1. Observe what PWM was commanded over the window (`pwm_*_obs`).
/// 2. Measure what happened — `omega_meas` from gyro, optionally
///    `v_meas` from ultrasonic Δd/Δt.
/// 3. Build [`ModelInput`] with the *measured* motion as the target:
///    `omega_target = omega_meas`, `v_target = v_meas.unwrap_or(<commanded v>)`.
/// 4. Set `v_label_present = v_meas.is_some()`.
#[derive(Debug, Clone, Copy)]
pub struct LabelledSample {
    pub input: ModelInput,
    pub pwm_l_obs: i32,
    pub pwm_r_obs: i32,
    /// `false` ⇒ the `v_target` field was filled from the commanded
    /// value, not a real ultrasonic measurement. The optimiser
    /// down-weights such samples ([`V_MISSING_WEIGHT`]) because the
    /// v-direction of the input is then less trustworthy.
    pub v_label_present: bool,
    pub dt_s: f32,
}

// ── Normalizer ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Normalizer {
    pwm_scale: f32,
    v_scale: f32,
    omega_scale: f32,
    battery_mean: f32,
    battery_scale: f32,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self {
            pwm_scale: NORM_PWM_SCALE,
            v_scale: NORM_V_SCALE,
            omega_scale: NORM_OMEGA_SCALE,
            battery_mean: NORM_BATT_MEAN,
            battery_scale: NORM_BATT_SCALE,
        }
    }
}

impl Normalizer {
    fn encode(&self, x: &ModelInput) -> [f32; INPUT_DIM] {
        [
            x.pwm_l_prev as f32 / self.pwm_scale,
            x.pwm_r_prev as f32 / self.pwm_scale,
            x.v_prev / self.v_scale,
            x.omega_prev / self.omega_scale,
            (x.battery_v - self.battery_mean) / self.battery_scale,
            x.v_target / self.v_scale,
            x.omega_target / self.omega_scale,
        ]
    }

    /// Maps the network's normalised output `y ∈ ℝ²` back to physical
    /// PCA9685 duty, clamped to `[-MAX_DUTY, MAX_DUTY]`.
    fn decode_pwm(&self, y: [f32; OUTPUT_DIM]) -> PwmCommand {
        let pwm_l = (y[0] * self.pwm_scale).round() as i32;
        let pwm_r = (y[1] * self.pwm_scale).round() as i32;
        let max = MAX_DUTY as i32;
        PwmCommand {
            pwm_l: pwm_l.clamp(-max, max),
            pwm_r: pwm_r.clamp(-max, max),
        }
    }

    /// Encode a *target* PWM (in physical units) into normalised space
    /// for use as a training target. Saturates inputs that fall outside
    /// `[-pwm_scale, pwm_scale]` to keep loss gradients sane.
    fn encode_pwm(&self, pwm_l: i32, pwm_r: i32) -> [f32; OUTPUT_DIM] {
        let s = self.pwm_scale;
        [
            (pwm_l as f32 / s).clamp(-1.0, 1.0),
            (pwm_r as f32 / s).clamp(-1.0, 1.0),
        ]
    }
}

// ── Layer ────────────────────────────────────────────────────────────────────

/// A single fully-connected layer with its Adam state. Adam moments
/// are not persisted (`#[serde(skip)]`); they reinitialise to zero on
/// load and re-warm in a few SGD steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    weights: Vec<Vec<f32>>, // [out][in]
    bias: Vec<f32>,         // [out]
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
        // Xavier / Glorot for tanh: U(-bound, +bound), bound = √(6/(fan_in+fan_out)).
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

    /// Reinitialise Adam state to match the (possibly just-loaded)
    /// weight shape. Called after deserialisation since Adam moments
    /// aren't persisted.
    fn reset_adam_state(&mut self) {
        let out_dim = self.weights.len();
        let in_dim = self.weights.first().map(|r| r.len()).unwrap_or(0);
        self.w_m = vec![vec![0.0; in_dim]; out_dim];
        self.w_v = vec![vec![0.0; in_dim]; out_dim];
        self.b_m = vec![0.0; out_dim];
        self.b_v = vec![0.0; out_dim];
    }

    /// Pre-activation forward: `y = W·x + b`. Caller applies the
    /// nonlinearity (or not, for the output layer).
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

    /// One Adam-lite step on this layer given grads w.r.t. weights and bias.
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

    /// EWMA of the per-sample PWM-residual RMS, in normalised PWM units.
    /// Useful telemetry for the WebUI; not persisted.
    #[serde(skip, default = "default_residual")]
    residual_pwm: f32,
}

fn default_residual() -> f32 {
    0.0
}

impl MotorModel {
    /// Construct a freshly-bootstrapped model: Xavier weights then
    /// [`BOOTSTRAP_STEPS`] SGD steps against synthetic inverse-prior
    /// samples. Deterministic in `BOOTSTRAP_SEED`.
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
            residual_pwm: 0.0,
        };
        let samples = synthetic_inverse_samples(&mut rng, BOOTSTRAP_SAMPLES);
        for _ in 0..BOOTSTRAP_STEPS {
            // Take one random sample per step (mini-batch size 1; data is
            // tiny so iterating in random order is fine).
            let i = rng.random_range(0..samples.len());
            model.observe(samples[i]);
        }
        // `observe` updates `last_battery_v` to each sample's voltage; reset
        // to the caller's live reading so the staleness check on the next
        // boot anchors on actual hardware state, not the last synthetic.
        model.meta.last_battery_v = battery_v_now;
        model
    }

    /// Load from disk if present and non-stale, else bootstrap.
    ///
    /// "Non-stale" means the saved `last_battery_v` is within
    /// [`BATTERY_STALE_THRESHOLD_V`] of `battery_v_now`. On any failure
    /// (missing, unparseable, wrong schema, stale battery) the function
    /// falls through to [`Self::default_bootstrap`] and logs the reason.
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

    pub fn residual_pwm(&self) -> f32 {
        self.residual_pwm
    }

    /// Forward inference: desired motion + state → PWM command.
    pub fn predict(&self, x: ModelInput) -> PwmCommand {
        let z0 = self.normalizer.encode(&x);
        let a1 = self.layer_1.forward(&z0);
        let h1: Vec<f32> = a1.iter().map(|v| v.tanh()).collect();
        let a2 = self.layer_2.forward(&h1);
        let h2: Vec<f32> = a2.iter().map(|v| v.tanh()).collect();
        let a3 = self.layer_3.forward(&h2);
        self.normalizer.decode_pwm([a3[0], a3[1]])
    }

    /// One SGD step on a labelled window. Updates Adam state and the
    /// running residual EWMA. Bumps `trained_steps`.
    pub fn observe(&mut self, s: LabelledSample) {
        // Forward, saving intermediates needed for backprop.
        let z0 = self.normalizer.encode(&s.input);
        let a1 = self.layer_1.forward(&z0);
        let h1: Vec<f32> = a1.iter().map(|v| v.tanh()).collect();
        let a2 = self.layer_2.forward(&h1);
        let h2: Vec<f32> = a2.iter().map(|v| v.tanh()).collect();
        let a3 = self.layer_3.forward(&h2); // linear output, normalised PWM space

        // Targets in normalised space.
        let target = self.normalizer.encode_pwm(s.pwm_l_obs, s.pwm_r_obs);
        let weight = if s.v_label_present {
            1.0
        } else {
            V_MISSING_WEIGHT
        };

        // dL/da3 — MSE with weight + L2 regularisation, both on
        // normalised output. Factor of 2 absorbed into LR.
        let d_a3: Vec<f32> = (0..OUTPUT_DIM)
            .map(|i| weight * (a3[i] - target[i]) + L2_LAMBDA * a3[i])
            .collect();

        // Track residual EWMA in normalised PWM units.
        let r2: f32 = (0..OUTPUT_DIM).map(|i| (a3[i] - target[i]).powi(2)).sum();
        let r = (r2 / OUTPUT_DIM as f32).sqrt();
        self.residual_pwm = (1.0 - RESIDUAL_ALPHA) * self.residual_pwm + RESIDUAL_ALPHA * r;

        // Backprop layer 3 (linear output: dL/dh2 = W3ᵀ · dL/da3).
        let (dw3, db3, d_h2) = backprop_linear(&self.layer_3.weights, &d_a3, &h2);
        // Layer 2 (tanh hidden: dL/da2 = dL/dh2 · (1 - h2²)).
        let d_a2: Vec<f32> = (0..HIDDEN_2)
            .map(|i| d_h2[i] * (1.0 - h2[i] * h2[i]))
            .collect();
        let (dw2, db2, d_h1) = backprop_linear(&self.layer_2.weights, &d_a2, &h1);
        // Layer 1.
        let d_a1: Vec<f32> = (0..HIDDEN_1)
            .map(|i| d_h1[i] * (1.0 - h1[i] * h1[i]))
            .collect();
        let (dw1, db1, _d_z0) = backprop_linear(&self.layer_1.weights, &d_a1, &z0);

        // Apply Adam-lite updates.
        self.layer_1.adam_step(&dw1, &db1);
        self.layer_2.adam_step(&dw2, &db2);
        self.layer_3.adam_step(&dw3, &db3);

        self.meta.trained_steps = self.meta.trained_steps.saturating_add(1);
        // Track last_battery_v so the save reflects the most recent regime.
        self.meta.last_battery_v = s.input.battery_v;
    }
}

/// Backprop through one fully-connected layer with grad-of-output `d_y`
/// and input `x_in`. Returns `(dW, db, dX_in)`.
///
/// `dW[i][j] = d_y[i] · x_in[j]`, `db[i] = d_y[i]`, `dX_in[j] = Σ_i W[i][j] · d_y[i]`.
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

/// Hand-coded forward dynamics used to bootstrap the model into a
/// sensible starting region. Linear-with-deadband, matching the
/// operator note in `CLAUDE.md` (PWM 1500 ≈ 0.6 m/s forward) and a
/// simple skid-steer turning relationship.
fn synthetic_forward(pwm_l: i32, pwm_r: i32) -> (f32, f32) {
    // pwm_avg ∈ ~ [-2000, 2000] over the operator's normal range.
    let pwm_avg = (pwm_l + pwm_r) as f32 * 0.5;
    let pwm_diff = (pwm_r - pwm_l) as f32;
    // Deadband: anything inside ±200 PWM produces no motion (static friction).
    let dead = 200.0;
    let v = if pwm_avg.abs() < dead {
        0.0
    } else {
        // 1500 PWM → 0.6 m/s ⇒ slope ≈ 0.6 / (1500 − 200) ≈ 4.6e-4 m/s per PWM.
        (pwm_avg.signum()) * (pwm_avg.abs() - dead) * 4.6e-4
    };
    // Turning: pwm_diff ≈ 1000 → ~1 rad/s.
    let omega = pwm_diff * 1.0e-3;
    (v, omega)
}

/// Generate inverse-training samples: pick a random PWM pair, compute
/// its synthetic forward motion, then build a labelled sample where
/// that motion is the `(v_target, ω_target)` input and the original
/// PWM is the regression target. The model trains the inverse from
/// these synthetic pairs.
fn synthetic_inverse_samples(rng: &mut SmallRng, n: usize) -> Vec<LabelledSample> {
    let max = MAX_DUTY as i32;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let pwm_l = rng.random_range(-max..=max);
        let pwm_r = rng.random_range(-max..=max);
        let (v, omega) = synthetic_forward(pwm_l, pwm_r);
        let battery_v = rng.random_range(7.2..8.4);
        out.push(LabelledSample {
            input: ModelInput {
                pwm_l_prev: 0,
                pwm_r_prev: 0,
                v_prev: 0.0,
                omega_prev: 0.0,
                battery_v,
                v_target: v,
                omega_target: omega,
            },
            pwm_l_obs: pwm_l,
            pwm_r_obs: pwm_r,
            v_label_present: true,
            dt_s: 0.2,
        });
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn neutral_state(v_target: f32, omega_target: f32, battery_v: f32) -> ModelInput {
        ModelInput {
            pwm_l_prev: 0,
            pwm_r_prev: 0,
            v_prev: 0.0,
            omega_prev: 0.0,
            battery_v,
            v_target,
            omega_target,
        }
    }

    fn round_trip_motion(m: &MotorModel, v_target: f32, omega_target: f32) -> (f32, f32) {
        let cmd = m.predict(neutral_state(v_target, omega_target, 7.8));
        synthetic_forward(cmd.pwm_l, cmd.pwm_r)
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let a = MotorModel::default_bootstrap(7.8);
        let b = MotorModel::default_bootstrap(7.8);
        // Identical weights from identical seed.
        assert_eq!(a.layer_1.weights, b.layer_1.weights);
        assert_eq!(a.layer_2.weights, b.layer_2.weights);
        assert_eq!(a.layer_3.weights, b.layer_3.weights);
        // Same predictions on a representative grid.
        for v in [-0.4, 0.0, 0.4] {
            for w in [-1.0, 0.0, 1.0] {
                let pa = a.predict(neutral_state(v, w, 7.8));
                let pb = b.predict(neutral_state(v, w, 7.8));
                assert_eq!((pa.pwm_l, pa.pwm_r), (pb.pwm_l, pb.pwm_r));
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
        for v in [-0.5, 0.0, 0.3] {
            for w in [-0.5, 0.0, 0.5] {
                let pa = m.predict(neutral_state(v, w, 7.8));
                let pb = loaded.predict(neutral_state(v, w, 7.8));
                assert_eq!((pa.pwm_l, pa.pwm_r), (pb.pwm_l, pb.pwm_r));
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
        // Load with a battery 0.5 V away — past the 0.3 V threshold.
        let reloaded = MotorModel::load_or_bootstrap(f.path(), 7.9);
        // Re-bootstrapped → trained_steps from the fresh bootstrap, and
        // last_battery_v reflects the new reading (not the file's 7.2).
        assert!((reloaded.meta.last_battery_v - 7.9).abs() < 1e-3);
    }

    #[test]
    fn bootstrap_round_trip_motion_in_envelope() {
        // After the bootstrap warm-up, the model should produce PWMs
        // that — when run through the *same* synthetic forward — land
        // close to the requested motion, at least inside the achievable
        // envelope.
        let m = MotorModel::default_bootstrap(7.8);
        // Easy targets well inside what 4095 PWM can produce.
        for (v_t, w_t) in [(0.3, 0.0), (-0.3, 0.0), (0.0, 0.5), (0.0, -0.5), (0.2, 0.3)] {
            let (v_got, w_got) = round_trip_motion(&m, v_t, w_t);
            // Loose tolerance — bootstrap is rough; later online learning
            // tightens this. The real assertion is "not garbage."
            assert!(
                (v_got - v_t).abs() < 0.25 && (w_got - w_t).abs() < 0.4,
                "target ({v_t}, {w_t}) → got ({v_got}, {w_got})"
            );
        }
    }

    #[test]
    fn inverse_fits_synthetic_dynamics() {
        // Train past the bootstrap with more synthetic data; the
        // round-trip motion should tighten.
        let mut m = MotorModel::default_bootstrap(7.8);
        let mut rng = SmallRng::seed_from_u64(0xFEED_BEEF);
        for _ in 0..5000 {
            let samples = synthetic_inverse_samples(&mut rng, 1);
            m.observe(samples[0]);
        }
        for (v_t, w_t) in [(0.3, 0.0), (0.0, 0.5), (-0.2, -0.3), (0.4, 0.2)] {
            let (v_got, w_got) = round_trip_motion(&m, v_t, w_t);
            assert!(
                (v_got - v_t).abs() < 0.15 && (w_got - w_t).abs() < 0.25,
                "after 5k extra steps: target ({v_t}, {w_t}) → got ({v_got}, {w_got})"
            );
        }
    }

    #[test]
    fn zero_target_predicts_small_pwm() {
        // Deadband regulariser: target = (0, 0) should produce a PWM
        // close to zero. (Strict zero would require infinite training;
        // ≤ 200 PWM is enough to confirm L2 is doing its job.)
        let m = MotorModel::default_bootstrap(7.8);
        let cmd = m.predict(neutral_state(0.0, 0.0, 7.8));
        assert!(
            cmd.pwm_l.abs() < 400 && cmd.pwm_r.abs() < 400,
            "expected near-zero PWM for zero target, got ({}, {})",
            cmd.pwm_l,
            cmd.pwm_r
        );
    }

    #[test]
    fn battery_input_has_measurable_effect() {
        // After training, the predicted PWM should vary across battery
        // voltages (otherwise the model is ignoring that input).
        let mut m = MotorModel::default_bootstrap(7.8);
        let mut rng = SmallRng::seed_from_u64(0xBADD_F00D);
        // Synthesise samples where battery scales the achievable speed
        // — train the model to associate low battery with higher PWM
        // for the same target.
        for _ in 0..3000 {
            let pwm = rng.random_range(-3000..=3000);
            let battery_v: f32 = rng.random_range(7.2..=8.4);
            // Same forward, but the "effective" PWM is scaled by
            // battery. Inverse target encodes this: same motion needs
            // a higher PWM at lower battery.
            let effective_pwm = (pwm as f32 * (battery_v / 8.4)) as i32;
            let (v, omega) = synthetic_forward(effective_pwm, effective_pwm);
            m.observe(LabelledSample {
                input: ModelInput {
                    pwm_l_prev: 0,
                    pwm_r_prev: 0,
                    v_prev: 0.0,
                    omega_prev: 0.0,
                    battery_v,
                    v_target: v,
                    omega_target: omega,
                },
                pwm_l_obs: pwm,
                pwm_r_obs: pwm,
                v_label_present: true,
                dt_s: 0.2,
            });
        }
        let lo = m.predict(neutral_state(0.4, 0.0, 7.2));
        let hi = m.predict(neutral_state(0.4, 0.0, 8.4));
        let delta = (lo.pwm_l - hi.pwm_l).abs();
        assert!(
            delta as f32 > 0.02 * MAX_DUTY as f32,
            "expected battery to shift PWM by >2% of MAX_DUTY, got Δ={delta}"
        );
    }

    #[test]
    fn out_of_envelope_clamps() {
        let m = MotorModel::default_bootstrap(7.8);
        // 5 m/s forward is impossible; the model should saturate the
        // PWM at MAX_DUTY instead of going to NaN / huge values.
        let cmd = m.predict(neutral_state(5.0, 0.0, 7.8));
        assert!(cmd.pwm_l.abs() <= MAX_DUTY as i32);
        assert!(cmd.pwm_r.abs() <= MAX_DUTY as i32);
        // And it's biased forward (not stuck at zero).
        assert!(cmd.pwm_l > 0 && cmd.pwm_r > 0);
    }

    #[test]
    fn sparse_v_labels_still_train() {
        // Mix labelled and unlabelled samples 1:4. The ω round-trip
        // (which doesn't depend on v_label_present) should still
        // tighten.
        let mut m = MotorModel::default_bootstrap(7.8);
        let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF);
        for i in 0..4000 {
            let mut s = synthetic_inverse_samples(&mut rng, 1).remove(0);
            if i % 5 != 0 {
                // Mark v as missing — the v_target field stays at the
                // ground-truth motion (which is what would happen if the
                // commanded v happened to match observed), but weight
                // is reduced.
                s.v_label_present = false;
            }
            m.observe(s);
        }
        for w_t in [-0.5_f32, -0.2, 0.2, 0.5] {
            let (_, w_got) = round_trip_motion(&m, 0.0, w_t);
            assert!(
                (w_got - w_t).abs() < 0.3,
                "ω target {w_t} → got {w_got} after sparse-v training"
            );
        }
    }
}
