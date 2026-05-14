//! 3-sensor infrared line tracker.
//!
//! GPIO: left=14, center=15, right=23. Active-low: High = line detected.

use rppal::gpio::{Gpio, InputPin, Level};

use crate::Result;

const PIN_LEFT:   u8 = 14;
const PIN_CENTER: u8 = 15;
const PIN_RIGHT:  u8 = 23;

pub struct InfraredSensors {
    left:   InputPin,
    center: InputPin,
    right:  InputPin,
}

impl InfraredSensors {
    pub fn new() -> Result<Self> {
        let gpio = Gpio::new()?;
        Ok(Self {
            left:   gpio.get(PIN_LEFT)?.into_input(),
            center: gpio.get(PIN_CENTER)?.into_input(),
            right:  gpio.get(PIN_RIGHT)?.into_input(),
        })
    }

    /// Returns (left, center, right): true = line detected.
    pub fn read_all(&self) -> (bool, bool, bool) {
        (
            self.left.read()   == Level::High,
            self.center.read() == Level::High,
            self.right.read()  == Level::High,
        )
    }
}
