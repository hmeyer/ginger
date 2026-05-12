//! Active buzzer on GPIO 17 (HIGH = on).

use std::thread::sleep;
use std::time::Duration;

use rppal::gpio::{Gpio, OutputPin};

use crate::Result;

const BUZZER_PIN: u8 = 17;

pub struct Buzzer {
    pin: OutputPin,
}

impl Buzzer {
    pub fn new() -> Result<Self> {
        let pin = Gpio::new()?.get(BUZZER_PIN)?.into_output_low();
        Ok(Self { pin })
    }

    pub fn on(&mut self) {
        self.pin.set_high();
    }

    pub fn off(&mut self) {
        self.pin.set_low();
    }

    pub fn beep(&mut self, count: u32, on_ms: u64, off_ms: u64) {
        for _ in 0..count {
            self.on();
            sleep(Duration::from_millis(on_ms));
            self.off();
            sleep(Duration::from_millis(off_ms));
        }
    }
}

impl Drop for Buzzer {
    fn drop(&mut self) {
        self.off();
    }
}
