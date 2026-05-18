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
//! - [`twoview`] — two-view monocular initialization (M3).
//! - [`tracking`] — constant-velocity prediction + motion-only BA (M4).
//! - [`map`] — keyframes / map points / covisibility graph (M5).
//! - [`local_ba`] — block-sparse Schur local bundle adjustment (M5).
//! - [`triangulation`] — gated two-view new-point triangulation (M5).
//! - [`bow`] — binary Bag-of-Words vocabulary + database for place
//!   recognition / loop closure / relocalization (M6).
//! - [`pnp`] — P3P + RANSAC pose recovery from 3D↔2D matches (M6).
//! - [`sim3`] — Sim(3) group + Essential-graph pose-graph optimization
//!   for loop-closure / scale-drift correction (M6).

pub mod bow;
pub mod camera;
pub mod dataset;
pub mod intrinsics;
pub mod lie;
pub mod local_ba;
pub mod map;
pub mod optimize;
pub mod pnp;
pub mod sim3;
pub mod tracking;
pub mod triangulation;
pub mod twoview;
