//! Motion stack: the learned inverse motor model and (later in PLAN
//! stages) the pose integrator, label assembler, and exploration
//! controller.
//!
//! Stage 1 (this commit): only [`model`] — a tiny MLP that takes
//! desired motion `(v_target, ω_target)` plus chassis state and emits
//! a `(pwm_l, pwm_r)` command. Camera-free, no hardware dependency;
//! the whole module unit-tests headless.

pub mod labels;
pub mod model;
pub mod pose;

pub use labels::{LabelStats, RejectionCounts};
pub use model::{LabelledSample, ModelInput, MotorModel, PwmCommand};
pub use pose::{MotionTarget, PoseState, TrailPoint};
