//! FAST-9 corner detector.
//!
//! The implementation (scalar + aarch64 NEON) lives in the
//! dependency-light `ginger-fast` crate so the SIMD path can be
//! cross-checked for aarch64 in CI without the camera/hardware stack.
//! This module just re-exports it; callers use `crate::slam::fast::*`
//! unchanged.

pub use ginger_fast::fast::{Corner, detect, non_max_suppress};
