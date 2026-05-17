//! Buzzer on GPIO 17 (HIGH = on).
//!
//! The pin is also bit-banged as a square wave by [`Buzzer::tone`] to get
//! a variable pitch (an R2D2-ish warble). A strictly *active* buzzer has
//! its own fixed oscillator and may not track the pitch, but most kit
//! buzzers follow the drive frequency well enough.

use std::thread::sleep;
use std::time::{Duration, Instant};

use rppal::gpio::{Gpio, OutputPin};

use crate::Result;

const BUZZER_PIN: u8 = 17;

pub struct Buzzer {
    pin: OutputPin,
}

/// Busy-wait for `d`. `thread::sleep` granularity (~50–100 µs on Linux)
/// is too coarse for audio half-periods, so spin instead.
fn spin(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {}
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

    /// Drive the pin as a ~50 % square wave at `freq_hz` for `dur_ms`.
    /// `freq_hz == 0` is a silent rest of the same length.
    pub fn tone(&mut self, freq_hz: u32, dur_ms: u64) {
        let dur = Duration::from_millis(dur_ms);
        if freq_hz == 0 {
            self.pin.set_low();
            sleep(dur);
            return;
        }
        let half = Duration::from_nanos(500_000_000 / freq_hz as u64);
        let start = Instant::now();
        while start.elapsed() < dur {
            self.pin.set_high();
            spin(half);
            self.pin.set_low();
            spin(half);
        }
        self.pin.set_low();
    }
}

impl Drop for Buzzer {
    fn drop(&mut self) {
        self.off();
    }
}
