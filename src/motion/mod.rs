//! Motion stack: arcade-drive mapping, learned forward motor model,
//! pose integrator, label worker, exploration controller.
//!
//! Drive path: WebUI joystick / autonomous controller → [`arcade_drive`]
//! → raw motor PWMs. Pure math, no learned-model dependency.
//!
//! Learning loop: [`labels`] worker observes (commanded PWMs, measured
//! Δs from US, measured Δθ from BNO055 fusion) every 200 ms and feeds
//! the [`model`] forward predictor. The model is read out for
//! diagnostics and (later) explore planning; it is *not* on the drive
//! path. See `motion::model` for the why-not-inverse story.

pub mod explore;
pub mod labels;
pub mod model;
pub mod pose;

pub use explore::{ExploreHandle, ExplorePhase, ExploreStatus, PolarScan, ScanRay};
pub use labels::{LabelStats, RejectionCounts};
pub use model::{LabelledSample, ModelInput, MotionPrediction, MotorModel};
pub use pose::{MotionTarget, PoseState, TrailPoint};

use crate::hal::pca9685::MAX_DUTY;

// ── Arcade drive (joystick (v, ω) → motor PWMs) ───────────────────────────────

/// PWM per (m/s) of commanded forward velocity. Calibrated from
/// CLAUDE.md: "PWM 1500 ≈ 0.6 m/s" → ~2500 PWM/(m/s). The WebUI joystick
/// saturates at ±2000 PWM, so this puts full deflection at ~0.8 m/s of
/// commanded v — a healthy indoor cruise.
const ARCADE_K_V: f32 = 2500.0;

/// PWM differential per (rad/s) of commanded angular velocity, with
/// wiring-quirk-aware sign: on this chassis `pwm_l > pwm_r` drives
/// physical CCW (chassis-frame positive ω). Calibrated from the
/// live-session data: PWM diff 1400 produced ≈ 0.26 rad/s → 5400
/// PWM/(rad/s). Rounded down to leave headroom against the clamp.
const ARCADE_K_OMEGA: f32 = 5000.0;

/// Map an operator's `(v_target, omega_target)` intent to raw motor
/// PWMs. Pure math, no learned model — robust by construction.
///
/// Wiring convention (this chassis): `pwm_l > pwm_r` → physical CCW.
/// The mapping is `pwm_l = K_v · v + K_ω · ω; pwm_r = K_v · v − K_ω · ω`
/// which makes `+ω` increase `pwm_l` relative to `pwm_r` — matching the
/// wiring quirk so commanded CCW produces physical CCW.
///
/// Outputs clamped to `[-MAX_DUTY, MAX_DUTY]` per side.
pub fn arcade_drive(v_target: f32, omega_target: f32) -> (i32, i32) {
    let v_pwm = v_target * ARCADE_K_V;
    let w_pwm = omega_target * ARCADE_K_OMEGA;
    let max = MAX_DUTY as f32;
    let l = (v_pwm + w_pwm).round().clamp(-max, max) as i32;
    let r = (v_pwm - w_pwm).round().clamp(-max, max) as i32;
    (l, r)
}

// ── Shared helpers used by both `pose` and `labels` ──────────────────────────

/// Wrap an angle to `(-π, π]`. Shared by the pose integrator and the
/// label worker so a yaw delta that crosses ±π is handled identically
/// in both consumers (any divergence would let the motor model train
/// against a yaw signal the integrator never uses).
pub(crate) fn wrap_pi(a: f32) -> f32 {
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

/// Chip-frame → chassis-frame yaw sign. `+1.0` if the BNO055's reported
/// CCW yaw matches the chassis's right-hand-rule "CCW about chassis-Z
/// is positive" convention; `-1.0` otherwise. Default `+1.0`; validate
/// on the live robot by rotating the chassis ~90° CCW and confirming
/// `pose.theta` moves toward `+π/2`.
pub(crate) const YAW_SIGN: f32 = 1.0;

#[cfg(test)]
mod arcade_tests {
    use super::*;

    #[test]
    fn pure_forward_symmetric() {
        let (l, r) = arcade_drive(0.5, 0.0);
        assert_eq!(l, r, "forward should be symmetric");
        assert!(l > 0, "forward should be positive PWMs");
    }

    #[test]
    fn pure_reverse_symmetric() {
        let (l, r) = arcade_drive(-0.5, 0.0);
        assert_eq!(l, r);
        assert!(l < 0);
    }

    #[test]
    fn ccw_intent_drives_pwm_l_greater_than_pwm_r() {
        let (l, r) = arcade_drive(0.0, 0.5);
        assert!(
            l > r,
            "CCW intent should yield pwm_l > pwm_r (wiring); got L={l}, R={r}"
        );
        assert_eq!(l, -r, "pure rotation should be anti-symmetric");
    }

    #[test]
    fn cw_intent_drives_pwm_l_less_than_pwm_r() {
        let (l, r) = arcade_drive(0.0, -0.5);
        assert!(l < r);
        assert_eq!(l, -r);
    }

    #[test]
    fn clamps_at_max_duty() {
        let max = MAX_DUTY as i32;
        // Saturating forward + CCW would land beyond ±MAX_DUTY on the
        // left side; output must clamp.
        let (l, r) = arcade_drive(5.0, 5.0);
        assert_eq!(l, max);
        assert!(r >= -max && r <= max);
        let (l, r) = arcade_drive(-5.0, 5.0);
        assert!(l >= -max && l <= max);
        assert_eq!(r, -max);
    }

    #[test]
    fn zero_intent_zero_pwm() {
        assert_eq!(arcade_drive(0.0, 0.0), (0, 0));
    }
}
