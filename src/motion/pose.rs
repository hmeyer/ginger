//! Stage 3 of `PLAN.md`: 2D pose integrator.
//!
//! A 20 Hz worker reads the latest `(v_target, ω_target)` issued via
//! `/api/motion/drive` and integrates it into chassis pose `(x, y, θ)`.
//! Two corrections are applied on every tick:
//!
//! 1. **ω override.** Gyro `gyro_z` is more accurate than any motor
//!    model could ever be — the integrator always trusts it for the
//!    rotational component. The model-derived `ω_target` is kept only
//!    for the WebUI residual display ("model said X, gyro says Y").
//! 2. **v override when ultrasonic is valid.** Same gate as Stage 2's
//!    label worker (straight + monotonic + in 8–80 cm range); when the
//!    window passes, `−Δd/Δt` substitutes for the commanded `v_target`
//!    as the integrator's `v_used`. This keeps the pose anchored to
//!    real-world translation whenever the sensor has a usable reading;
//!    between such windows, the integrator falls back to whatever was
//!    last commanded.
//!
//! Pose is **robot-frame** — `(0, 0, 0)` at the integrator's last
//! reset, no global consistency, drifts over time. That's fine for the
//! exploration use cases in Stages 4–6; if absolute localisation
//! matters later, a fiducial / RGB-D anchor is the right addition,
//! not a tweak to this integrator.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::warn;
use serde::Serialize;

use crate::api::SensorSnapshot;
use crate::imu::Imu;

// ── Cadence & trail ───────────────────────────────────────────────────────────

/// Integrator tick interval. 20 Hz → 50 ms.
const TICK: Duration = Duration::from_millis(50);

/// Push a trail point every `TRAIL_EVERY` ticks (≈4 Hz at 20 Hz tick).
/// Lower-frequency trail keeps JSON payloads small while preserving
/// visible path shape on the WebUI plot.
const TRAIL_EVERY: u32 = 5;

/// Cap on the in-memory trail. ~5 min × 4 Hz = 1200 points; clip past
/// that. The WebUI plot only needs the recent history.
const TRAIL_CAP: usize = 1200;

// ── v-override gate (mirrors src/motion/labels.rs) ────────────────────────────

const US_MIN_CM: f32 = 8.0;
const US_MAX_CM: f32 = 80.0;
const STRAIGHT_PWM_DIFF_THRESHOLD: i32 = 200;
const STATIC_PWM_AVG_THRESHOLD: i32 = 200;
const US_NOISE_FLOOR_CM: f32 = 1.0;

// ── Shared types ──────────────────────────────────────────────────────────────

/// Latest desired motion as set by `POST /api/motion/drive`. The pose
/// integrator reads this every tick; the drive endpoint also stores the
/// PWM the model predicted from it, so the WebUI can show "commanded
/// vs. predicted PWM" without re-running the model.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MotionTarget {
    pub v_target: f32,
    pub omega_target: f32,
    pub pwm_l: i32,
    pub pwm_r: i32,
}

/// One trail point — (x, y, unix milliseconds).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TrailPoint {
    pub x: f32,
    pub y: f32,
    pub t_ms: u64,
}

/// Surfaced at `GET /api/motion/pose`. Includes both the latest
/// commanded and measured motion so the WebUI can show residuals
/// (model vs gyro / ultrasonic) without a separate endpoint.
#[derive(Clone, Debug, Serialize)]
pub struct PoseState {
    pub x: f32,
    pub y: f32,
    /// Heading in radians. `0` = chassis forward along +X. CCW positive
    /// (right-hand rule about chassis-Z, matches gyro convention).
    pub theta: f32,
    /// Last commanded `v_target` (m/s).
    pub v_cmd: f32,
    /// Last gyro-measured ω (rad/s). Source of truth used in integration.
    pub omega_gyro: f32,
    /// Last commanded ω_target — kept for residual visualisation.
    pub omega_cmd: f32,
    /// Last ultrasonic-derived v (m/s). `None` when the window failed
    /// the gate; the integrator then used `v_cmd` as `v_used`.
    pub v_us: Option<f32>,
    /// Drift indicator: Euclidean distance from origin to current pose.
    pub drift_m: f32,
    pub trail: VecDeque<TrailPoint>,
}

impl Default for PoseState {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            v_cmd: 0.0,
            omega_gyro: 0.0,
            omega_cmd: 0.0,
            v_us: None,
            drift_m: 0.0,
            trail: VecDeque::with_capacity(TRAIL_CAP),
        }
    }
}

impl PoseState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ── Integrator ────────────────────────────────────────────────────────────────

struct PoseWorker {
    sensors: Arc<RwLock<SensorSnapshot>>,
    imu: Arc<Imu>,
    target: Arc<RwLock<MotionTarget>>,
    pose: Arc<RwLock<PoseState>>,

    last_tick: Instant,
    /// Ultrasonic distance at `last_tick`. Carried forward so each
    /// 50 ms window has a `(start, end)` pair for the Δd/Δt v-override.
    initial_us_cm: Option<f32>,
    trail_counter: u32,
}

impl PoseWorker {
    fn new(
        sensors: Arc<RwLock<SensorSnapshot>>,
        imu: Arc<Imu>,
        target: Arc<RwLock<MotionTarget>>,
        pose: Arc<RwLock<PoseState>>,
    ) -> Self {
        let initial_us_cm = sensors.read().unwrap().us_cm;
        Self {
            sensors,
            imu,
            target,
            pose,
            last_tick: Instant::now(),
            initial_us_cm,
            trail_counter: 0,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let dt_s = (now - self.last_tick).as_secs_f32();
        if dt_s < 0.005 {
            return;
        }

        // ── ω: gyro-integrated mean over the window ──────────────────
        let samples = self.imu.recent_since(self.last_tick);
        let bias_dps = self.imu.gyro_bias_dps();
        let omega_gyro = if samples.is_empty() {
            0.0
        } else {
            let sum: f32 = samples
                .iter()
                .map(|s| (s.raw.gyro_dps()[2] - bias_dps[2]).to_radians())
                .sum();
            sum / samples.len() as f32
        };

        let target = *self.target.read().unwrap();
        let snap = self.sensors.read().unwrap().clone();
        let us_end = snap.us_cm;

        // ── v override: ultrasonic Δd/Δt when valid ──────────────────
        let v_us = try_v_us(
            snap.pwm_l_cmd,
            snap.pwm_r_cmd,
            self.initial_us_cm,
            us_end,
            dt_s,
        );
        let v_used = v_us.unwrap_or(target.v_target);

        // ── integrate ────────────────────────────────────────────────
        let mut pose = self.pose.write().unwrap();
        // Use midpoint heading for second-order accuracy at modest dt.
        let theta_mid = pose.theta + 0.5 * omega_gyro * dt_s;
        pose.x += v_used * theta_mid.cos() * dt_s;
        pose.y += v_used * theta_mid.sin() * dt_s;
        pose.theta = wrap_pi(pose.theta + omega_gyro * dt_s);
        pose.v_cmd = target.v_target;
        pose.omega_cmd = target.omega_target;
        pose.omega_gyro = omega_gyro;
        pose.v_us = v_us;
        pose.drift_m = (pose.x * pose.x + pose.y * pose.y).sqrt();

        self.trail_counter = self.trail_counter.wrapping_add(1);
        if self.trail_counter.is_multiple_of(TRAIL_EVERY) {
            let t_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let (x, y) = (pose.x, pose.y);
            pose.trail.push_back(TrailPoint { x, y, t_ms });
            while pose.trail.len() > TRAIL_CAP {
                pose.trail.pop_front();
            }
        }
        drop(pose);

        self.last_tick = now;
        self.initial_us_cm = us_end;
    }
}

/// Wrap an angle to `(-π, π]`. The integrator keeps θ in that range so
/// JSON consumers don't see ever-growing values after many laps.
fn wrap_pi(a: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let pi = std::f32::consts::PI;
    let mut x = a % two_pi;
    if x > pi {
        x -= two_pi;
    } else if x <= -pi {
        x += two_pi;
    }
    x
}

/// Same v-label gate as `src/motion/labels.rs::try_v_label`, returning
/// the metric velocity rather than tagging a training sample. Keeping
/// this duplicated rather than pulling into a shared module — it's a
/// dozen lines and the two callers want slightly different semantics
/// (the pose integrator doesn't increment rejection counters).
fn try_v_us(
    pwm_l: i32,
    pwm_r: i32,
    us_start: Option<f32>,
    us_end: Option<f32>,
    dt_s: f32,
) -> Option<f32> {
    if (pwm_l - pwm_r).abs() >= STRAIGHT_PWM_DIFF_THRESHOLD {
        return None;
    }
    let us_start = us_start?;
    let us_end = us_end?;
    if !(US_MIN_CM..=US_MAX_CM).contains(&us_start) || !(US_MIN_CM..=US_MAX_CM).contains(&us_end) {
        return None;
    }
    let pwm_avg = (pwm_l + pwm_r) / 2;
    let dd = us_end - us_start;
    let expected_sign = if pwm_avg > STATIC_PWM_AVG_THRESHOLD {
        -1.0_f32
    } else if pwm_avg < -STATIC_PWM_AVG_THRESHOLD {
        1.0
    } else {
        0.0
    };
    if expected_sign != 0.0 && dd.signum() != expected_sign && dd.abs() > US_NOISE_FLOOR_CM {
        return None;
    }
    if dt_s < 0.01 {
        return None;
    }
    Some(-dd * 0.01 / dt_s)
}

/// Spawn the integrator thread. Detached; lives as long as the binary.
pub fn spawn(
    sensors: Arc<RwLock<SensorSnapshot>>,
    imu: Arc<Imu>,
    target: Arc<RwLock<MotionTarget>>,
    pose: Arc<RwLock<PoseState>>,
) {
    thread::Builder::new()
        .name("motion-pose".into())
        .spawn(move || {
            let mut worker = PoseWorker::new(sensors, imu, target, pose);
            loop {
                thread::sleep(TICK);
                worker.tick();
            }
        })
        .map(|_| ())
        .unwrap_or_else(|e| warn!("motion-pose: could not spawn thread: {e}"));
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure-function `tick` mirror used by tests — same integration
    /// math as `PoseWorker::tick` without the IMU / sensor side. Lets
    /// us exercise the integrator math headlessly.
    fn integrate_tick(pose: &mut PoseState, v: f32, omega: f32, dt_s: f32) {
        let theta_mid = pose.theta + 0.5 * omega * dt_s;
        pose.x += v * theta_mid.cos() * dt_s;
        pose.y += v * theta_mid.sin() * dt_s;
        pose.theta = wrap_pi(pose.theta + omega * dt_s);
        pose.drift_m = (pose.x * pose.x + pose.y * pose.y).sqrt();
    }

    #[test]
    fn pure_forward_increments_x() {
        // 0.3 m/s, no rotation, 1.0 s total → expect x ≈ 0.3.
        let mut pose = PoseState::default();
        for _ in 0..20 {
            integrate_tick(&mut pose, 0.3, 0.0, 0.05);
        }
        assert!((pose.x - 0.3).abs() < 1e-4, "x = {}", pose.x);
        assert!(pose.y.abs() < 1e-4, "y = {}", pose.y);
        assert!(pose.theta.abs() < 1e-4);
    }

    #[test]
    fn pure_rotation_keeps_position() {
        // Spin at π/2 rad/s for 2 s — θ should land at π.
        let mut pose = PoseState::default();
        for _ in 0..40 {
            integrate_tick(&mut pose, 0.0, std::f32::consts::PI / 2.0, 0.05);
        }
        assert!(pose.x.abs() < 1e-4);
        assert!(pose.y.abs() < 1e-4);
        assert!(
            (pose.theta - std::f32::consts::PI).abs() < 1e-3
                || (pose.theta + std::f32::consts::PI).abs() < 1e-3,
            "θ = {} (expected ±π)",
            pose.theta
        );
    }

    #[test]
    fn square_path_closes() {
        // Drive a unit square: forward 1m, +90° turn, ×4. Expect to
        // end at origin with θ = 0 (mod 2π).
        let mut pose = PoseState::default();
        // Discretise the spec: 1 m forward at 0.5 m/s = 2 s = 40 ticks.
        for _side in 0..4 {
            for _ in 0..40 {
                integrate_tick(&mut pose, 0.5, 0.0, 0.05);
            }
            // 90° turn at π/2 rad/s = 1 s = 20 ticks.
            for _ in 0..20 {
                integrate_tick(&mut pose, 0.0, std::f32::consts::PI / 2.0, 0.05);
            }
        }
        assert!(pose.drift_m < 1e-3, "drift = {} m", pose.drift_m);
        assert!(
            pose.theta.abs() < 1e-3 || (pose.theta.abs() - std::f32::consts::TAU).abs() < 1e-3,
            "θ = {}",
            pose.theta
        );
    }

    #[test]
    fn reset_clears() {
        let mut pose = PoseState::default();
        integrate_tick(&mut pose, 0.3, 0.5, 0.5);
        assert!(pose.drift_m > 0.0);
        pose.reset();
        assert_eq!(pose.x, 0.0);
        assert_eq!(pose.y, 0.0);
        assert_eq!(pose.theta, 0.0);
        assert_eq!(pose.drift_m, 0.0);
        assert!(pose.trail.is_empty());
    }

    #[test]
    fn theta_stays_in_range() {
        // Spin a lot — θ must stay in (-π, π].
        let mut pose = PoseState::default();
        for _ in 0..1000 {
            integrate_tick(&mut pose, 0.0, 3.0, 0.05);
        }
        assert!(
            pose.theta > -std::f32::consts::PI && pose.theta <= std::f32::consts::PI,
            "θ = {}",
            pose.theta
        );
    }

    #[test]
    fn v_us_gate_forward_straight() {
        // Forward, in-range, monotonic Δd → returns metric v.
        let v = try_v_us(1500, 1500, Some(30.0), Some(20.0), 0.05);
        match v {
            Some(x) => assert!((x - 2.0).abs() < 1e-3, "v = {x}"),
            None => panic!("expected Some, got None"),
        }
    }

    #[test]
    fn v_us_gate_turning_rejects() {
        assert!(try_v_us(2000, 1000, Some(30.0), Some(29.0), 0.05).is_none());
    }

    #[test]
    fn v_us_gate_range_rejects() {
        assert!(try_v_us(1500, 1500, Some(5.0), Some(4.0), 0.05).is_none());
        assert!(try_v_us(1500, 1500, Some(70.0), Some(85.0), 0.05).is_none());
    }

    #[test]
    fn v_us_gate_non_monotonic_rejects() {
        // Commanded forward, distance grew — reject.
        assert!(try_v_us(1500, 1500, Some(30.0), Some(35.0), 0.05).is_none());
    }

    #[test]
    fn v_us_gate_static_command_tolerates_jitter() {
        // Zero command, distance jittered → accept.
        assert!(try_v_us(0, 0, Some(30.0), Some(30.5), 0.05).is_some());
    }
}
