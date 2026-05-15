//! Actuator abstractions built on top of the HAL.
//!
//! Both modules drive servos/motors through a shared `Pca9685`, so they
//! take a short-lived `&mut Pca9685` borrow rather than owning the bus.

pub mod motors;
pub mod pan_tilt;
