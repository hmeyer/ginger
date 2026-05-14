//! PCA9685 16-channel 12-bit PWM driver over I2C.

use std::thread::sleep;
use std::time::Duration;

use rppal::i2c::I2c;

use crate::Result;

const MODE1: u8 = 0x00;
const PRESCALE: u8 = 0xFE;
const LED0_ON_L: u8 = 0x06;

const OSC_FREQ: u32 = 25_000_000;
const PWM_RES: u32 = 4096;

pub const MAX_DUTY: u16 = (PWM_RES - 1) as u16; // 4095
pub const FULL_OFF: u16 = PWM_RES as u16; // 4096 — sets FULL-OFF bit in OFF_H register

pub struct Pca9685 {
    i2c: I2c,
}

impl Pca9685 {
    pub fn new(address: u16) -> Result<Self> {
        let mut i2c = I2c::new()?;
        i2c.set_slave_address(address)?;
        i2c.smbus_write_byte(MODE1, 0x00)?;
        Ok(Self { i2c })
    }

    pub fn set_pwm_freq(&mut self, freq: f32) -> Result<()> {
        let prescale =
            ((OSC_FREQ as f32 / (PWM_RES as f32 * freq)).round() as u8).saturating_sub(1);
        let old_mode = self.i2c.smbus_read_byte(MODE1)?;
        self.i2c.smbus_write_byte(MODE1, (old_mode & 0x7F) | 0x10)?; // sleep
        self.i2c.smbus_write_byte(PRESCALE, prescale)?;
        self.i2c.smbus_write_byte(MODE1, old_mode)?;
        sleep(Duration::from_millis(5));
        self.i2c.smbus_write_byte(MODE1, old_mode | 0x80)?; // restart
        Ok(())
    }

    pub fn set_pwm(&mut self, channel: u8, on: u16, off: u16) -> Result<()> {
        let base = LED0_ON_L + 4 * channel;
        self.i2c.smbus_write_byte(base, (on & 0xFF) as u8)?;
        self.i2c.smbus_write_byte(base + 1, (on >> 8) as u8)?;
        self.i2c.smbus_write_byte(base + 2, (off & 0xFF) as u8)?;
        self.i2c.smbus_write_byte(base + 3, (off >> 8) as u8)?;
        Ok(())
    }

    /// pulse_us: pulse width in microseconds (500–2500 for standard servos at 50 Hz).
    pub fn set_servo_pulse_us(&mut self, channel: u8, pulse_us: f32) -> Result<()> {
        let off = (pulse_us * PWM_RES as f32 / 20_000.0) as u16;
        self.set_pwm(channel, 0, off)
    }
}
