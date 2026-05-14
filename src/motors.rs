//! 4WD motor control via PCA9685.
//!
//! Channel pairs (forward, reverse) per wheel:
//!   left_front (7,6), left_rear (5,4), right_front (1,0), right_rear (2,3)

use crate::{
    Result,
    pca9685::{FULL_OFF, MAX_DUTY, Pca9685},
};

struct WheelChannels {
    fwd: u8,
    rev: u8,
}

const WHEELS: [WheelChannels; 4] = [
    WheelChannels { fwd: 7, rev: 6 }, // left_front
    WheelChannels { fwd: 5, rev: 4 }, // left_rear
    WheelChannels { fwd: 1, rev: 0 }, // right_front
    WheelChannels { fwd: 2, rev: 3 }, // right_rear
];

pub struct Motors<'a> {
    pwm: &'a mut Pca9685,
}

impl<'a> Motors<'a> {
    pub fn new(pwm: &'a mut Pca9685) -> Self {
        Self { pwm }
    }

    /// duty: -4095 (full reverse) to 4095 (full forward), 0 = brake.
    pub fn set_wheel(&mut self, wheel_idx: usize, duty: i32) -> Result<()> {
        let duty = duty.clamp(-(MAX_DUTY as i32), MAX_DUTY as i32);
        let ch = &WHEELS[wheel_idx];
        if duty > 0 {
            self.pwm.set_pwm(ch.fwd, 0, FULL_OFF)?; // inactive pin: guaranteed LOW
            self.pwm.set_pwm(ch.rev, 0, duty as u16)?;
        } else if duty < 0 {
            self.pwm.set_pwm(ch.rev, 0, FULL_OFF)?; // inactive pin: guaranteed LOW
            self.pwm.set_pwm(ch.fwd, 0, (-duty) as u16)?;
        } else {
            self.pwm.set_pwm(ch.fwd, 0, MAX_DUTY)?;
            self.pwm.set_pwm(ch.rev, 0, MAX_DUTY)?;
        }
        Ok(())
    }

    /// Set all wheels: left side / right side duty (-4095..4095).
    pub fn drive(&mut self, left: i32, right: i32) -> Result<()> {
        self.set_wheel(0, left)?; // left_front
        self.set_wheel(1, left)?; // left_rear
        self.set_wheel(2, right)?; // right_front
        self.set_wheel(3, right)?; // right_rear
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        self.drive(0, 0)
    }
}
