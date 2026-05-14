//! Pan/tilt servo control via PCA9685.
//!
//! Pan channel 8, tilt channel 9. Pan is wired inverted.
//! Pulse: 500 µs = 0°, 1500 µs = 90°, 2500 µs = 180°.

use crate::{Result, pca9685::Pca9685};

const PAN_CHANNEL: u8 = 8;
const TILT_CHANNEL: u8 = 9;
const PULSE_MIN_US: f32 = 500.0;
const PULSE_MAX_US: f32 = 2500.0;

// Physical straight-ahead is at 83.5° rather than the nominal 90°.
// Trim shifts set_pan(90.0) to the correct pulse for actual straight.
const PAN_TRUE_CENTER_DEG: f32 = 83.5;
pub const PAN_TRIM_US: f32 = (90.0 - PAN_TRUE_CENTER_DEG) / 180.0 * (PULSE_MAX_US - PULSE_MIN_US);

fn angle_to_pulse(angle: f32, invert: bool, trim_us: f32) -> f32 {
    let angle = angle.clamp(0.0, 180.0);
    let pulse = PULSE_MIN_US + angle / 180.0 * (PULSE_MAX_US - PULSE_MIN_US);
    let pulse = if invert {
        PULSE_MIN_US + PULSE_MAX_US - pulse
    } else {
        pulse
    };
    pulse + trim_us
}

pub struct PanTilt<'a> {
    pwm: &'a mut Pca9685,
    pan_trim_us: f32,
    tilt_trim_us: f32,
}

impl<'a> PanTilt<'a> {
    pub fn new(pwm: &'a mut Pca9685, pan_trim_us: f32, tilt_trim_us: f32) -> Self {
        Self {
            pwm,
            pan_trim_us,
            tilt_trim_us,
        }
    }

    /// angle: 0–180°, 90 = center.
    pub fn set_pan(&mut self, angle: f32) -> Result<()> {
        let pulse = angle_to_pulse(angle, true, self.pan_trim_us);
        self.pwm.set_servo_pulse_us(PAN_CHANNEL, pulse)
    }

    /// angle: 0–180°, 90 = center.
    pub fn set_tilt(&mut self, angle: f32) -> Result<()> {
        let pulse = angle_to_pulse(angle, false, self.tilt_trim_us);
        self.pwm.set_servo_pulse_us(TILT_CHANNEL, pulse)
    }

    pub fn center(&mut self) -> Result<()> {
        self.set_pan(90.0)?;
        self.set_tilt(90.0)
    }
}
