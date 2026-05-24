//! Crate-wide error type.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I2C error: {0}")]
    I2c(#[from] rppal::i2c::Error),
    #[error("SPI error: {0}")]
    Spi(#[from] rppal::spi::Error),
    #[error("GPIO error: {0}")]
    Gpio(#[from] rppal::gpio::Error),
    #[error("Timeout waiting for {0}")]
    Timeout(&'static str),
    #[error("Camera: {0}")]
    Camera(String),
    #[error("IMU: {0}")]
    Imu(String),
}

pub type Result<T> = std::result::Result<T, Error>;
