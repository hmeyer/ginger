//! FAST-9 corner detection and the grayscale image / pyramid types it
//! operates on.
//!
//! Split into its own dependency-light crate (only `rayon`) so the
//! aarch64 NEON detector can be type-checked in CI with a plain
//! `cargo check -p ginger-fast --target aarch64-unknown-linux-gnu`,
//! without dragging in the camera / hardware stack (libcamera, v4l).

pub mod fast;
pub mod image;
