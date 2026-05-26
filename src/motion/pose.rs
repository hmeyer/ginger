//! Stage 3 of `PLAN.md`: 2D pose integrator.
//!
//! A 20 Hz worker reads the latest `(v_target, ω_target)` issued via
//! `/api/motion/drive` and integrates it into chassis pose `(x, y, θ)`.
//! Two corrections are applied on every tick:
//!
//! 1. **θ from BNO055 fusion.** The chip's IMUPLUS fusion engine
//!    produces a drift-compensated absolute orientation; we read its
//!    quaternion every tick and subtract the yaw captured at the last
//!    pose reset. No software-side gyro integration, no bias tracking.
//!    `omega_meas` (the "ω the IMU saw" residual the WebUI shows next
//!    to `ω_target`) is derived as `Δyaw / Δt` from the same source.
//! 2. **v override when ultrasonic is valid.** Same gate as Stage 2's
//!    label worker (straight + monotonic + in 8–80 cm range); when the
//!    window passes, `−Δd/Δt` substitutes for the commanded `v_target`
//!    as the integrator's `v_used`. This keeps the pose anchored to
//!    real-world translation whenever the sensor has a usable reading;
//!    between such windows, the integrator falls back to whatever was
//!    last commanded.
//!
//! Pose is **robot-frame** — `(0, 0, 0)` at the integrator's last
//! reset, no global consistency. The yaw component now barely drifts
//! (the BNO055 absorbs it); the x/y components still drift through the
//! v-channel because mono ultrasonic only gives v when the window
//! passes the straight-and-in-range gate.
//!
//! ## Chip → chassis yaw sign
//!
//! [`YAW_SIGN`] flips the chip's yaw to match the chassis convention
//! "CCW about chassis-Z is positive." Default `+1.0` assumes the BNO055
//! is mounted upright with chip-Z ≈ chassis-Z. Validate on the live
//! robot by rotating the chassis ~90° CCW and confirming `pose.theta`
//! moves toward `+π/2`; flip to `-1.0` if it moves the other way.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::warn;
use serde::Serialize;

use crate::api::SensorSnapshot;
use crate::imu::Imu;
use crate::motion::{YAW_SIGN, wrap_pi};

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
/// Reject v_us if the pan servo is not within this many degrees of
/// 90° (chassis-forward). Without this gate, the Stage-4 explore
/// swept scan moves the ultrasonic across the room while the chassis
/// is stationary — each pose tick sees a different `us_cm` (different
/// ray) and the integrator reads the jumps as forward velocity,
/// drifting pose.x by several metres in 30 s of explore.
const PAN_CENTERED_TOLERANCE_DEG: f32 = 5.0;
const PAN_FORWARD_DEG: f32 = 90.0;

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
/// (model vs fusion / ultrasonic) without a separate endpoint.
#[derive(Clone, Debug, Serialize)]
pub struct PoseState {
    pub x: f32,
    pub y: f32,
    /// Heading in radians. `0` = chassis forward along +X. CCW positive
    /// (right-hand rule about chassis-Z).
    pub theta: f32,
    /// Last commanded `v_target` (m/s).
    pub v_cmd: f32,
    /// Last IMU-measured ω (rad/s). Derived as `Δθ / Δt` from the BNO055
    /// fusion engine's quaternion delta across the tick — i.e. the same
    /// θ source the integrator uses, just differentiated. Field name
    /// kept (`omega_gyro`) for wire compatibility with the WebUI's
    /// existing Pose-card parser; the semantics are now fusion, not raw
    /// gyro.
    pub omega_gyro: f32,
    /// Last commanded ω_target — kept for residual visualisation.
    pub omega_cmd: f32,
    /// Last ultrasonic-derived v (m/s). `None` when the window failed
    /// the gate; the integrator then used `v_cmd` as `v_used`.
    pub v_us: Option<f32>,
    /// Drift indicator: Euclidean distance from origin to current pose.
    pub drift_m: f32,
    pub trail: VecDeque<TrailPoint>,
    /// Bumped each time [`Self::reset`] runs. The pose worker watches
    /// for changes and re-captures the BNO055's current yaw as the new
    /// origin on the next tick — that's how "reset" actually zeros θ
    /// even though θ is read absolutely from the chip's fusion engine.
    pub reset_seq: u32,
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
            reset_seq: 0,
        }
    }
}

impl PoseState {
    pub fn reset(&mut self) {
        let seq = self.reset_seq.wrapping_add(1);
        *self = Self::default();
        self.reset_seq = seq;
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

    /// Chassis-frame yaw at the moment the pose was last reset
    /// (chip yaw at that instant, times [`YAW_SIGN`]). `None` until the
    /// first usable fusion sample; re-set to `None` when `PoseState`'s
    /// `reset_seq` advances.
    yaw_origin: Option<f32>,
    /// Last θ value computed — used to derive `omega_meas` as
    /// `Δθ / Δt` and as the previous endpoint for the midpoint heading.
    last_theta: Option<f32>,
    /// Tracks `PoseState::reset_seq` so the worker notices the operator
    /// pressed "Reset pose" and re-captures `yaw_origin`.
    last_seen_reset_seq: u32,
}

impl PoseWorker {
    fn new(
        sensors: Arc<RwLock<SensorSnapshot>>,
        imu: Arc<Imu>,
        target: Arc<RwLock<MotionTarget>>,
        pose: Arc<RwLock<PoseState>>,
    ) -> Self {
        let initial_us_cm = sensors.read().unwrap().us_cm;
        let last_seen_reset_seq = pose.read().unwrap().reset_seq;
        Self {
            sensors,
            imu,
            target,
            pose,
            last_tick: Instant::now(),
            initial_us_cm,
            trail_counter: 0,
            yaw_origin: None,
            last_theta: None,
            last_seen_reset_seq,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let dt_s = (now - self.last_tick).as_secs_f32();
        if dt_s < 0.005 {
            return;
        }

        // Watch for an operator-triggered pose reset. Re-capture
        // `yaw_origin` on the next valid sample so θ goes back to 0
        // relative to the chassis's current heading.
        {
            let current = self.pose.read().unwrap().reset_seq;
            if current != self.last_seen_reset_seq {
                self.yaw_origin = None;
                self.last_theta = None;
                self.last_seen_reset_seq = current;
            }
        }

        // No fusion sample yet (chip warming up, or warm-up gate
        // closed). Don't integrate — advance the tick and bail.
        let Some(sample) = self.imu.latest() else {
            self.last_tick = now;
            self.initial_us_cm = self.sensors.read().unwrap().us_cm;
            return;
        };

        // BNO055 yaw → chassis-frame θ, relative to the pose's origin.
        let (_, _, yaw_chip) = sample.orientation.euler_angles();
        let yaw_chassis = yaw_chip * YAW_SIGN;
        let yaw_origin = *self.yaw_origin.get_or_insert(yaw_chassis);
        let theta = wrap_pi(yaw_chassis - yaw_origin);

        // `Δθ / Δt` on the same wrapped manifold = the chip's reported ω.
        let (omega_meas, theta_prev) = match self.last_theta {
            Some(prev) => (wrap_pi(theta - prev) / dt_s, prev),
            None => (0.0, theta),
        };
        self.last_theta = Some(theta);

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
            snap.pan,
        );
        let v_used = v_us.unwrap_or(target.v_target);

        // Midpoint heading for second-order accuracy on x/y. Wrap-aware
        // so a θ that crossed ±π between ticks doesn't put the midpoint
        // on the wrong side of the unit circle.
        let dtheta = wrap_pi(theta - theta_prev);
        let theta_mid = wrap_pi(theta_prev + 0.5 * dtheta);

        let mut pose = self.pose.write().unwrap();
        pose.x += v_used * theta_mid.cos() * dt_s;
        pose.y += v_used * theta_mid.sin() * dt_s;
        pose.theta = theta;
        pose.v_cmd = target.v_target;
        pose.omega_cmd = target.omega_target;
        pose.omega_gyro = omega_meas;
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
    pan_deg: f32,
) -> Option<f32> {
    if (pwm_l - pwm_r).abs() >= STRAIGHT_PWM_DIFF_THRESHOLD {
        return None;
    }
    // Pan not centred → the ultrasonic isn't reading distance along
    // chassis-forward, so Δd/Δt is not chassis forward velocity. Also
    // catches the Stage-4 explore scan: pan sweeps 15°→165° while the
    // chassis is stationary; each tick sees a wildly different
    // `us_cm`, and naive integration drifts pose.x by metres in
    // seconds (live observation, 30 s explore → ~7 m fake travel).
    if (pan_deg - PAN_FORWARD_DEG).abs() > PAN_CENTERED_TOLERANCE_DEG {
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
        // Commanded near zero — chassis should be stationary. If we
        // returned `Some` here we'd integrate ultrasonic noise as
        // motion. Live observation (post PR #65 deploy): pose.x
        // wandered 0.84 → 1.72 → 1.17 over 30 s of explore's idle
        // phase because each tick saw 1-2 cm of US jitter and computed
        // v ≈ ±0.3 m/s from it. Returning `None` falls back to v_cmd
        // (zero) in the integrator — ground truth for a stopped robot.
        return None;
    };
    if dd.signum() != expected_sign && dd.abs() > US_NOISE_FLOOR_CM {
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

    /// Pan-centred (90°) is the precondition for v_us to mean
    /// "chassis-forward velocity". Helper to keep tests readable.
    const PAN: f32 = 90.0;

    #[test]
    fn v_us_gate_forward_straight() {
        // Forward, in-range, monotonic Δd → returns metric v.
        let v = try_v_us(1500, 1500, Some(30.0), Some(20.0), 0.05, PAN);
        match v {
            Some(x) => assert!((x - 2.0).abs() < 1e-3, "v = {x}"),
            None => panic!("expected Some, got None"),
        }
    }

    #[test]
    fn v_us_gate_turning_rejects() {
        assert!(try_v_us(2000, 1000, Some(30.0), Some(29.0), 0.05, PAN).is_none());
    }

    #[test]
    fn v_us_gate_range_rejects() {
        assert!(try_v_us(1500, 1500, Some(5.0), Some(4.0), 0.05, PAN).is_none());
        assert!(try_v_us(1500, 1500, Some(70.0), Some(85.0), 0.05, PAN).is_none());
    }

    #[test]
    fn v_us_gate_non_monotonic_rejects() {
        // Commanded forward, distance grew — reject.
        assert!(try_v_us(1500, 1500, Some(30.0), Some(35.0), 0.05, PAN).is_none());
    }

    #[test]
    fn v_us_gate_zero_command_rejects() {
        // Zero command → no v_us. Returning anything here would
        // integrate ultrasonic noise as fake forward motion during
        // idle, which is the bug this assertion guards against.
        assert!(try_v_us(0, 0, Some(30.0), Some(30.5), 0.05, PAN).is_none());
        // Even on a perfectly-clean (zero Δd) reading: still None.
        // Truth at zero command is "no motion" and the integrator
        // should use v_cmd (= 0), not the sensor.
        assert!(try_v_us(0, 0, Some(30.0), Some(30.0), 0.05, PAN).is_none());
    }

    #[test]
    fn v_us_gate_rejects_when_pan_off_centre() {
        // Even an otherwise-valid window (straight, in-range, monotonic)
        // must reject when the pan servo is sweeping. This is the
        // Stage-4 fix — without it, swept-scan ultrasonic jumps drift
        // pose.x by metres in 30 s.
        assert!(try_v_us(1500, 1500, Some(30.0), Some(20.0), 0.05, 60.0).is_none());
        assert!(try_v_us(1500, 1500, Some(30.0), Some(20.0), 0.05, 120.0).is_none());
        // Within the ±5° tolerance still passes.
        assert!(try_v_us(1500, 1500, Some(30.0), Some(20.0), 0.05, 87.0).is_some());
        assert!(try_v_us(1500, 1500, Some(30.0), Some(20.0), 0.05, 93.0).is_some());
    }
}
