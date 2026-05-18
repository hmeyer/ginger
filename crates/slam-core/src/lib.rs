//! Camera-free geometry + optimization core for the SLAM pipeline (M2).
//!
//! Split into its own dependency-light crate (`nalgebra` + `serde` +
//! `toml`, no camera/libcamera) so the numerically-sensitive math
//! cross-compiles and unit-tests without the hardware stack:
//!
//! ```text
//! cargo test  -p ginger-slam-core
//! cargo check -p ginger-slam-core --target aarch64-unknown-linux-gnu
//! ```
//!
//! Modules:
//! - [`lie`] — SO3/SE3 `exp`/`log`, hand-rolled (nalgebra is not a
//!   Lie-group lib) so the conventions are pinned and tested.
//! - [`camera`] — pinhole + Brown–Conrady [`camera::CameraModel`].
//! - [`intrinsics`] — serde config + the rev 1.3 FOV-derived prior.
//! - [`optimize`] — dense Levenberg–Marquardt with a Huber robustifier,
//!   the first solver to harden per the M2 plan.
//! - [`dataset`] — camera-free half of the replay harness: PGM
//!   sequence + intrinsics loading.

pub mod camera;
pub mod dataset;
pub mod intrinsics;
pub mod lie;
pub mod optimize;
