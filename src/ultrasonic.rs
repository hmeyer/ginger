//! HC-SR04 ultrasonic distance sensor.
//!
//! Trigger: GPIO 27 (output), Echo: GPIO 22 (input). Max range ~3 m.

use std::time::{Duration, Instant};

use rppal::gpio::{Gpio, InputPin, Level, OutputPin};

use crate::Result;

const TRIGGER_PIN: u8 = 27;
const ECHO_PIN: u8 = 22;

const ECHO_IDLE_TIMEOUT: Duration = Duration::from_millis(100);
const ECHO_START_TIMEOUT: Duration = Duration::from_millis(10);
const ECHO_END_TIMEOUT: Duration = Duration::from_millis(50);

pub struct Ultrasonic {
    trigger: OutputPin,
    echo: InputPin,
}

impl Ultrasonic {
    pub fn new() -> Result<Self> {
        let gpio = Gpio::new()?;
        let trigger = gpio.get(TRIGGER_PIN)?.into_output_low();
        let echo = gpio.get(ECHO_PIN)?.into_input();
        Ok(Self { trigger, echo })
    }

    /// Return distance in cm, or None on timeout / out of range.
    pub fn distance_cm(&mut self) -> Option<f32> {
        // Wait for echo idle (LOW)
        let deadline = Instant::now() + ECHO_IDLE_TIMEOUT;
        while self.echo.read() == Level::High {
            if Instant::now() > deadline {
                return None;
            }
        }

        // 10 µs trigger pulse
        self.trigger.set_high();
        std::thread::sleep(Duration::from_micros(10));
        self.trigger.set_low();

        // Wait for echo HIGH
        let deadline = Instant::now() + ECHO_START_TIMEOUT;
        while self.echo.read() == Level::Low {
            if Instant::now() > deadline {
                return None;
            }
        }
        let t_start = Instant::now();

        // Wait for echo LOW
        let deadline = Instant::now() + ECHO_END_TIMEOUT;
        while self.echo.read() == Level::High {
            if Instant::now() > deadline {
                return None;
            }
        }
        let elapsed = t_start.elapsed().as_secs_f32();

        Some((elapsed * 34300.0 / 2.0 * 10.0).round() / 10.0)
    }
}
