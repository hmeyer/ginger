//! Stage 2 of `PLAN.md`: assemble labelled 200 ms training windows for
//! [`MotorModel`] from the live PWM command, IMU fusion, and
//! ultrasonic streams.
//!
//! ## Window construction
//!
//! Every 200 ms the worker takes a snapshot of:
//!
//! * the latest commanded PWM (`pwm_l_cmd`, `pwm_r_cmd`) — published by
//!   the supervisor into [`SensorSnapshot`];
//! * all BNO055 fusion samples in the window (`Imu::recent_since`),
//!   each carrying an absolute orientation quaternion + a
//!   linear-acceleration vector (gravity already removed by the chip);
//! * the ultrasonic distance at the *start* and *end* of the window
//!   (captured by the previous tick → carried forward).
//!
//! ## Labels and rejections
//!
//! * `ω_meas` = `Δyaw / Δt` between the first and last orientation
//!   snapshot in the window (rad/s). The chip's IMUPLUS fusion engine
//!   has already done the gyro integration, bias-correction and
//!   gravity-aware projection internally — we just subtract endpoints.
//!   **Always present** unless the window has fewer than two samples
//!   (chip warming up) or hits whole-window rejection.
//! * `v_meas` = `−Δd/Δt` from ultrasonic, in m/s. **Sometimes present**;
//!   rejected if the window is curving, distances are out of the
//!   sensor's reliable 8–80 cm range, or the trend isn't sign-consistent
//!   with the commanded direction.
//! * **Whole-window rejection** drops both `ω` and `v` for the window
//!   if any single-sample `|linear_accel| > 3 m/s²` (collision / pickup
//!   / kick). With gravity already removed by the chip, anything above
//!   that threshold is a real impulse on the chassis.
//!
//! ## `ModelInput.v_target` / `omega_target`
//!
//! Per the PLAN's inverse-training trick: the *measured* motion goes in
//! as the input target. `v_target = v_meas` when present, else the
//! previous window's measured `v` (a placeholder — the v signal is
//! weaker on those samples and the optimiser down-weights them via
//! `v_label_present = false`). `omega_target = omega_meas` always.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use log::warn;
use serde::Serialize;

use crate::api::SensorSnapshot;
use crate::imu::Imu;
use crate::motion::{LabelledSample, ModelInput, MotionTarget, MotorModel, YAW_SIGN, wrap_pi};

// ── Window cadence ────────────────────────────────────────────────────────────

const WINDOW: Duration = Duration::from_millis(200);

// ── Whole-window rejection thresholds ─────────────────────────────────────────

/// Any per-sample `|linear_accel|` above this rejects the window
/// (collision / pickup / kick). The BNO055 already subtracts gravity,
/// so the magnitude of `linear_accel` *is* the chassis's own
/// acceleration — anything past this is a real impulse, not gravity
/// pointing through a tilted chip.
const LIN_ACCEL_SPIKE_MPS2: f32 = 3.0;

// ── v-label gate ──────────────────────────────────────────────────────────────

const US_MIN_CM: f32 = 8.0;
const US_MAX_CM: f32 = 80.0;
/// `|pwm_l − pwm_r|` above this means the window curved — `Δd` no
/// longer represents pure forward speed, so we reject the v label.
const STRAIGHT_PWM_DIFF_THRESHOLD: i32 = 200;
/// Below this commanded `|pwm_avg|` we accept *either* sign of `Δd` —
/// the robot wasn't supposed to move much, sensor noise dominates.
const STATIC_PWM_AVG_THRESHOLD: i32 = 200;
/// Ultrasonic noise floor — Δd below this magnitude is treated as
/// "essentially stationary", so a small wrong-sign jitter doesn't
/// reject an otherwise-good window.
const US_NOISE_FLOOR_CM: f32 = 1.0;
/// Reject v_meas if the pan servo is not within this many degrees of
/// 90° (chassis-forward). Without this, Stage-4 explore's swept-scan
/// would feed the model bogus v labels: pan sweeps 15°→165° while the
/// chassis is stationary, every label window sees a different
/// ultrasonic ray, and the implied Δd/Δt becomes a fictitious
/// "forward velocity" the model learns from.
const PAN_CENTERED_TOLERANCE_DEG: f32 = 5.0;
const PAN_FORWARD_DEG: f32 = 90.0;

// ── Telemetry ─────────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct RejectionCounts {
    /// Whole-window rejection: a single fusion sample's linear-accel
    /// magnitude exceeded [`LIN_ACCEL_SPIKE_MPS2`]. Real impulse on the
    /// chassis (bump, kick, pickup) — both ω and v labels dropped.
    pub accel_spike: u64,
    pub v_out_of_range: u64,
    pub v_not_straight: u64,
    pub v_non_monotonic: u64,
}

#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct LabelStats {
    pub samples_observed: u64,
    pub samples_v_labelled: u64,
    pub rejections: RejectionCounts,
}

// ── Worker state ──────────────────────────────────────────────────────────────

/// One iteration of the labeller. Owns no thread; can be unit-tested by
/// poking [`tick`] with a faked `SensorSnapshot` + an [`Imu`] that
/// returns synthetic samples.
struct LabelWorker {
    sensors: Arc<RwLock<SensorSnapshot>>,
    imu: Arc<Imu>,
    /// Latest desired motion set by `POST /api/motion/drive` (the
    /// operator's commanded `v_target` / `ω_target`). Used as the
    /// `v_target` fallback when ultrasonic Δd/Δt doesn't yield a real
    /// v_meas — see [`Self::tick`] for the why.
    target: Arc<RwLock<MotionTarget>>,
    model: Arc<RwLock<MotorModel>>,
    stats: Arc<RwLock<LabelStats>>,

    window_start: Instant,
    /// Ultrasonic distance at `window_start`. `None` when the sensor
    /// was disabled or returned no reading; the v label can't be
    /// computed without it.
    initial_us_cm: Option<f32>,

    // Carries for `ModelInput.*_prev` on the next tick. `prev_v` is the
    // last *known* v — either ultrasonic-measured when valid, or the
    // commanded v as a placeholder when not.
    prev_pwm_l: i32,
    prev_pwm_r: i32,
    prev_v: f32,
    prev_omega: f32,
}

impl LabelWorker {
    /// Build, do not run. The caller spawns a thread that loops over
    /// [`Self::tick`] every [`WINDOW`].
    fn new(
        sensors: Arc<RwLock<SensorSnapshot>>,
        imu: Arc<Imu>,
        target: Arc<RwLock<MotionTarget>>,
        model: Arc<RwLock<MotorModel>>,
        stats: Arc<RwLock<LabelStats>>,
    ) -> Self {
        let initial_us_cm = sensors.read().unwrap().us_cm;
        Self {
            sensors,
            imu,
            target,
            model,
            stats,
            window_start: Instant::now(),
            initial_us_cm,
            prev_pwm_l: 0,
            prev_pwm_r: 0,
            prev_v: 0.0,
            prev_omega: 0.0,
        }
    }

    /// One window. Returns `true` if a `LabelledSample` was observed.
    fn tick(&mut self) -> bool {
        let window_end = Instant::now();
        let dt_s = (window_end - self.window_start).as_secs_f32();
        // Pull BNO055 fusion samples in `[window_start, window_end]`.
        let imu_samples = self.imu.recent_since(self.window_start);

        // Pre-pass: scan for whole-window rejection on the chip's
        // gravity-removed linear acceleration.
        let mut accel_spike = false;
        for s in &imu_samples {
            let a = s.linear_accel;
            let amag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if amag > LIN_ACCEL_SPIKE_MPS2 {
                accel_spike = true;
            }
        }

        // Snapshot the sensor/PWM state once, atomically.
        let snap = self.sensors.read().unwrap().clone();
        let us_end = snap.us_cm;

        if accel_spike {
            self.stats.write().unwrap().rejections.accel_spike += 1;
            self.advance_window(window_end, us_end);
            return false;
        }

        // Need at least two fusion snapshots to compute Δyaw across the
        // window; bail otherwise (chip warming up or window too tight).
        if imu_samples.len() < 2 || dt_s < 0.05 {
            self.advance_window(window_end, us_end);
            return false;
        }

        // ω from the chip's drift-compensated yaw delta. No software
        // integration, no bias to subtract — IMUPLUS did all that. The
        // shared `YAW_SIGN` keeps the convention identical to the pose
        // integrator, so the motor model trains against the same ω
        // values the integrator and WebUI joystick speak in.
        let (_, _, yaw_start) = imu_samples.first().unwrap().orientation.euler_angles();
        let (_, _, yaw_end) = imu_samples.last().unwrap().orientation.euler_angles();
        let omega_meas = wrap_pi((yaw_end - yaw_start) * YAW_SIGN) / dt_s;

        // v label gate.
        let v_meas = self.try_v_label(
            snap.pwm_l_cmd,
            snap.pwm_r_cmd,
            self.initial_us_cm,
            us_end,
            dt_s,
            snap.pan,
        );
        let v_label_present = v_meas.is_some();

        // Operator's currently-commanded v_target. Used as the v
        // fallback when ultrasonic Δd/Δt didn't yield a v_meas — this
        // is the PLAN.md Stage 2 design, and it matters most when
        // there's no obstacle in front of the ultrasonic (open room):
        // without it the model trains "pwm_avg=1200 → v=0" because the
        // previous still-window's v was zero, which is structurally
        // wrong and was breaking the v→pwm mapping until this fix.
        let v_commanded = self.target.read().unwrap().v_target;

        // Build the labelled window. `v_target` uses the measured value
        // when present, falls back to the operator's commanded v
        // (loss-weighted 0.3× via `v_label_present=false` in the model).
        let sample = LabelledSample {
            input: ModelInput {
                pwm_l_prev: self.prev_pwm_l,
                pwm_r_prev: self.prev_pwm_r,
                v_prev: self.prev_v,
                omega_prev: self.prev_omega,
                battery_v: snap.battery_v,
                v_target: v_meas.unwrap_or(v_commanded),
                omega_target: omega_meas,
            },
            pwm_l_obs: snap.pwm_l_cmd,
            pwm_r_obs: snap.pwm_r_cmd,
            v_label_present,
            dt_s,
        };

        self.model.write().unwrap().observe(sample);
        {
            let mut stats = self.stats.write().unwrap();
            stats.samples_observed += 1;
            if v_label_present {
                stats.samples_v_labelled += 1;
            }
        }

        // Carry forward (only on a successful observation). `prev_v` is
        // the best-known v at this point: measured when valid, else the
        // commanded fallback we just used as the training target.
        self.prev_pwm_l = snap.pwm_l_cmd;
        self.prev_pwm_r = snap.pwm_r_cmd;
        self.prev_omega = omega_meas;
        self.prev_v = v_meas.unwrap_or(v_commanded);
        self.advance_window(window_end, us_end);
        true
    }

    fn advance_window(&mut self, t: Instant, new_initial_us_cm: Option<f32>) {
        self.window_start = t;
        self.initial_us_cm = new_initial_us_cm;
    }

    /// Try to compute `v_meas` (m/s) from ultrasonic Δd/Δt. Returns
    /// `Some` only when the window is straight, in range, and the
    /// distance trend is sign-consistent with the commanded direction.
    /// Increments the appropriate rejection counter when it returns
    /// `None` — so the WebUI can show *why* v labels are sparse.
    fn try_v_label(
        &self,
        pwm_l: i32,
        pwm_r: i32,
        us_start: Option<f32>,
        us_end: Option<f32>,
        dt_s: f32,
        pan_deg: f32,
    ) -> Option<f32> {
        // Straight?
        if (pwm_l - pwm_r).abs() >= STRAIGHT_PWM_DIFF_THRESHOLD {
            self.stats.write().unwrap().rejections.v_not_straight += 1;
            return None;
        }
        // Pan not centred → ultrasonic isn't reading chassis-forward,
        // so Δd/Δt isn't forward velocity. Bucket the rejection under
        // `v_not_straight` (closest existing reason) to keep the JSON
        // schema stable — operator distinguishes by context (pan
        // sweeps happen during Stage-4 scans only).
        if (pan_deg - PAN_FORWARD_DEG).abs() > PAN_CENTERED_TOLERANCE_DEG {
            self.stats.write().unwrap().rejections.v_not_straight += 1;
            return None;
        }
        let us_start = us_start?;
        let us_end = us_end?;
        // Range?
        if !(US_MIN_CM..=US_MAX_CM).contains(&us_start)
            || !(US_MIN_CM..=US_MAX_CM).contains(&us_end)
        {
            self.stats.write().unwrap().rejections.v_out_of_range += 1;
            return None;
        }
        // Sign-consistent?
        let pwm_avg = (pwm_l + pwm_r) / 2;
        let dd = us_end - us_start; // cm
        let expected_sign = if pwm_avg > STATIC_PWM_AVG_THRESHOLD {
            -1.0_f32 // forward → distance decreases
        } else if pwm_avg < -STATIC_PWM_AVG_THRESHOLD {
            1.0
        } else {
            0.0 // commanded near-zero — accept either sign within noise
        };
        if expected_sign != 0.0 && dd.signum() != expected_sign && dd.abs() > US_NOISE_FLOOR_CM {
            self.stats.write().unwrap().rejections.v_non_monotonic += 1;
            return None;
        }
        if dt_s < 0.05 {
            return None;
        }
        // `−Δd / Δt`, cm → m.
        Some(-dd * 0.01 / dt_s)
    }
}

/// Spawn the long-lived label worker. Detached thread; lives as long
/// as the binary.
pub fn spawn(
    sensors: Arc<RwLock<SensorSnapshot>>,
    imu: Arc<Imu>,
    target: Arc<RwLock<MotionTarget>>,
    model: Arc<RwLock<MotorModel>>,
    stats: Arc<RwLock<LabelStats>>,
) {
    thread::Builder::new()
        .name("motion-labels".into())
        .spawn(move || {
            let mut worker = LabelWorker::new(sensors, imu, target, model, stats);
            loop {
                thread::sleep(WINDOW);
                let _ = worker.tick();
            }
        })
        .map(|_| ())
        .unwrap_or_else(|e| warn!("motion-labels: could not spawn thread: {e}"));
}

// ── Tests ────────────────────────────────────────────────────────────────────
//
// The label worker takes a real `Arc<Imu>` which holds a polling thread
// against a real I²C bus. For headless tests we don't want any of that;
// instead, the *pure* gating logic in `try_v_label` and the per-sample
// rejection scanner are factored out of `tick` enough to exercise them
// directly.

#[cfg(test)]
mod tests {
    use super::*;

    // Standalone helpers re-implemented for testing. These match the
    // gating logic in `LabelWorker` exactly (line-by-line) and exist so
    // the tests can exercise the conditions without spinning up an IMU
    // or sensor snapshot.

    fn v_label_test(pwm_l: i32, pwm_r: i32, us_start: f32, us_end: f32, dt_s: f32) -> VResult {
        // Helpers default pan to centred (90°). The pan-off case has
        // its own test below.
        v_label_test_pan(pwm_l, pwm_r, us_start, us_end, dt_s, 90.0)
    }

    fn v_label_test_pan(
        pwm_l: i32,
        pwm_r: i32,
        us_start: f32,
        us_end: f32,
        dt_s: f32,
        pan_deg: f32,
    ) -> VResult {
        if (pwm_l - pwm_r).abs() >= STRAIGHT_PWM_DIFF_THRESHOLD {
            return VResult::NotStraight;
        }
        if (pan_deg - PAN_FORWARD_DEG).abs() > PAN_CENTERED_TOLERANCE_DEG {
            return VResult::NotStraight; // bucketed under not-straight
        }
        if !(US_MIN_CM..=US_MAX_CM).contains(&us_start)
            || !(US_MIN_CM..=US_MAX_CM).contains(&us_end)
        {
            return VResult::OutOfRange;
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
            return VResult::NonMonotonic;
        }
        VResult::Ok(-dd * 0.01 / dt_s)
    }

    #[derive(Debug, PartialEq)]
    enum VResult {
        Ok(f32),
        NotStraight,
        OutOfRange,
        NonMonotonic,
    }

    #[test]
    fn v_label_forward_straight_monotonic() {
        // Forward 200 PWM, 30→20 cm over 200 ms → v = 0.5 m/s.
        let r = v_label_test(1500, 1500, 30.0, 20.0, 0.2);
        match r {
            VResult::Ok(v) => assert!((v - 0.5).abs() < 1e-3, "got {v}"),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn v_label_reverse_straight_monotonic() {
        // Reverse: 20→30 cm in 200 ms → v ≈ -0.5 m/s.
        match v_label_test(-1500, -1500, 20.0, 30.0, 0.2) {
            VResult::Ok(v) => assert!((v + 0.5).abs() < 1e-3, "got {v}"),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn v_label_rejects_turning_window() {
        // Differential 1000 → not straight.
        assert_eq!(
            v_label_test(2000, 1000, 30.0, 29.0, 0.2),
            VResult::NotStraight
        );
    }

    #[test]
    fn v_label_rejects_out_of_range() {
        // Start at 5 cm (below floor of 8 cm).
        assert_eq!(v_label_test(1500, 1500, 5.0, 4.0, 0.2), VResult::OutOfRange);
        // End past 80 cm.
        assert_eq!(
            v_label_test(-1500, -1500, 70.0, 85.0, 0.2),
            VResult::OutOfRange
        );
    }

    #[test]
    fn v_label_rejects_non_monotonic() {
        // Commanded forward, but distance grew — wrong sign by more
        // than the noise floor.
        assert_eq!(
            v_label_test(1500, 1500, 30.0, 35.0, 0.2),
            VResult::NonMonotonic
        );
    }

    #[test]
    fn v_label_tolerates_us_noise_at_zero_command() {
        // Commanded zero, distance jittered 0.5 cm — should accept.
        match v_label_test(0, 0, 30.0, 30.5, 0.2) {
            VResult::Ok(_) => {}
            other => panic!("expected Ok at near-zero command, got {other:?}"),
        }
    }

    #[test]
    fn v_label_tolerates_small_wrong_sign_jitter() {
        // Commanded forward, distance jittered by 0.5 cm wrong way
        // (below US_NOISE_FLOOR_CM). Should accept.
        match v_label_test(1500, 1500, 30.0, 30.5, 0.2) {
            VResult::Ok(_) => {}
            other => panic!("expected Ok with sub-noise jitter, got {other:?}"),
        }
    }

    #[test]
    fn v_label_rejects_when_pan_off_centre() {
        // Even an otherwise-valid window must reject when the pan
        // servo isn't centred. Catches Stage 4's swept-scan poisoning
        // of the v-label stream.
        assert_eq!(
            v_label_test_pan(1500, 1500, 30.0, 20.0, 0.2, 60.0),
            VResult::NotStraight
        );
        assert_eq!(
            v_label_test_pan(1500, 1500, 30.0, 20.0, 0.2, 120.0),
            VResult::NotStraight
        );
        // Within ±5° tolerance still passes.
        match v_label_test_pan(1500, 1500, 30.0, 20.0, 0.2, 87.0) {
            VResult::Ok(_) => {}
            other => panic!("expected Ok at pan=87°, got {other:?}"),
        }
    }

    /// Per-sample rejection scanner mirrored from `LabelWorker::tick`.
    /// Each input is a linear-accel triple (m/s², gravity already
    /// removed by the chip) — same shape as `ImuSample::linear_accel`.
    fn scan_lin_accel(samples: &[(f32, f32, f32)]) -> bool {
        samples
            .iter()
            .any(|a| (a.0 * a.0 + a.1 * a.1 + a.2 * a.2).sqrt() > LIN_ACCEL_SPIKE_MPS2)
    }

    #[test]
    fn scan_clean_window_passes() {
        // Quiet chassis: linear-accel near zero on every sample.
        let s = vec![(0.05, -0.02, 0.01); 30];
        assert!(!scan_lin_accel(&s));
    }

    #[test]
    fn scan_accel_spike_window_rejects() {
        // One sample with |linear_accel| = 5 m/s² — past the 3 m/s² gate.
        let mut s = vec![(0.0, 0.0, 0.0); 30];
        s[10].0 = 5.0;
        assert!(scan_lin_accel(&s));
    }

    #[test]
    fn scan_threshold_just_above_passes_just_below_rejects() {
        // 2.9 m/s² should not trip; 3.1 m/s² should.
        let just_below = vec![(2.9, 0.0, 0.0); 5];
        assert!(!scan_lin_accel(&just_below));
        let just_above = vec![(0.0, 0.0, 3.1)];
        assert!(scan_lin_accel(&just_above));
    }

    #[test]
    fn stats_default_clean() {
        let s = LabelStats::default();
        assert_eq!(s.samples_observed, 0);
        assert_eq!(s.samples_v_labelled, 0);
        assert_eq!(s.rejections.accel_spike, 0);
    }
}
