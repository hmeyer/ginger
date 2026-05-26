//! Motion stack: the learned inverse motor model and (later in PLAN
//! stages) the pose integrator, label assembler, and exploration
//! controller.
//!
//! Stage 1 (this commit): only [`model`] — a tiny MLP that takes
//! desired motion `(v_target, ω_target)` plus chassis state and emits
//! a `(pwm_l, pwm_r)` command. Camera-free, no hardware dependency;
//! the whole module unit-tests headless.

pub mod explore;
pub mod labels;
pub mod model;
pub mod pose;

pub use explore::{ExploreHandle, ExplorePhase, ExploreStatus, PolarScan, ScanRay};
pub use labels::{LabelStats, RejectionCounts};
pub use model::{LabelledSample, ModelInput, MotorModel, PwmCommand};
pub use pose::{MotionTarget, PoseState, TrailPoint};

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
