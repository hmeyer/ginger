//! Hardware abstraction layer: raw bus and peripheral drivers.
//!
//! These modules talk directly to the Raspberry Pi's I2C/SPI/GPIO via
//! `rppal` and know nothing about the robot above them.

pub mod adc;
pub mod bno055;
pub mod buzzer;
pub mod infrared;
pub mod led;
pub mod pca9685;
pub mod ultrasonic;
