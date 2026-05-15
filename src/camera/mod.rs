//! Camera subsystem: libcamera capture plus a software auto-exposure loop.

pub mod auto_exposure;
pub mod capture;

pub use auto_exposure::ExposureConfig;
pub use capture::{Camera, Frame};
