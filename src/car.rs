//! Top-level Car struct — coordinates all hardware with built-in safety.

use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::{
    Result, adc::Adc, buzzer::Buzzer, infrared::InfraredSensors, led::LedStrip, motors::Motors,
    pca9685::Pca9685, servo::PanTilt, ultrasonic::Ultrasonic,
};

const SAFE_DISTANCE_CM: f32 = 30.0;
const PAN_SETTLE: Duration = Duration::from_millis(300);
const DRIVE_CHECK_INTERVAL: Duration = Duration::from_millis(100);

pub struct Car {
    pwm: Pca9685,
    pub adc: Adc,
    pub ir: InfraredSensors,
    pub leds: LedStrip,
    pub buzzer: Buzzer,
    us: Ultrasonic,
    // motors and pan_tilt borrow `pwm`, so we call them via helpers that hold short borrows
}

impl Car {
    pub fn new() -> Result<Self> {
        let mut pwm = Pca9685::new(0x40)?;
        pwm.set_pwm_freq(50.0)?;

        // Center servos on startup
        {
            let mut pt = PanTilt::new(&mut pwm, 0.0, 0.0);
            pt.center()?;
        }

        Ok(Self {
            pwm,
            adc: Adc::new()?,
            ir: InfraredSensors::new()?,
            leds: LedStrip::new(255)?,
            buzzer: Buzzer::new()?,
            us: Ultrasonic::new()?,
        })
    }

    // ------------------------------------------------------------------
    // Low-level accessors (short-lived borrows into pwm)

    pub fn motors(&mut self) -> Motors<'_> {
        Motors::new(&mut self.pwm)
    }

    pub fn pan_tilt(&mut self) -> PanTilt<'_> {
        PanTilt::new(&mut self.pwm, 0.0, 0.0)
    }

    pub fn us(&mut self) -> &mut Ultrasonic {
        &mut self.us
    }

    // ------------------------------------------------------------------
    // Safety

    /// Center pan, wait for settle, read ultrasonic. Returns (safe, distance_cm).
    pub fn clear_ahead(&mut self) -> Result<(bool, Option<f32>)> {
        self.pan_tilt().set_pan(90.0)?;
        sleep(PAN_SETTLE);
        let dist = self.us.distance_cm();
        let safe = dist.is_none_or(|d| d > SAFE_DISTANCE_CM);
        Ok((safe, dist))
    }

    // ------------------------------------------------------------------
    // Movement

    /// Drive for up to `duration`. Checks clearance when going forward.
    /// Returns true if completed, false if stopped early by obstacle.
    pub fn drive(&mut self, left: i32, right: i32, duration: Duration) -> Result<bool> {
        if left > 0 && right > 0 {
            let (safe, dist) = self.clear_ahead()?;
            if !safe {
                println!("STOP: obstacle at {:.1}cm", dist.unwrap_or(0.0));
                return Ok(false);
            }
        }

        self.motors().drive(left, right)?;
        let deadline = Instant::now() + duration;

        loop {
            if Instant::now() >= deadline {
                break;
            }
            if left > 0
                && right > 0
                && let Some(d) = self.us.distance_cm()
                && d < SAFE_DISTANCE_CM
            {
                println!("STOP: obstacle at {d:.1}cm");
                self.motors().stop()?;
                return Ok(false);
            }
            sleep(DRIVE_CHECK_INTERVAL);
        }

        self.motors().stop()?;
        Ok(true)
    }

    pub fn forward(&mut self, duty: i32, duration: Duration) -> Result<bool> {
        self.drive(duty, duty, duration)
    }

    pub fn backward(&mut self, duty: i32, duration: Duration) -> Result<bool> {
        self.drive(-duty, -duty, duration)
    }

    pub fn turn_left(&mut self, duty: i32, duration: Duration) -> Result<bool> {
        self.drive(-duty, duty, duration)
    }

    pub fn turn_right(&mut self, duty: i32, duration: Duration) -> Result<bool> {
        self.drive(duty, -duty, duration)
    }

    pub fn stop(&mut self) -> Result<()> {
        self.motors().stop()
    }

    pub fn battery_v(&mut self) -> Result<f32> {
        self.adc.read_battery()
    }

    /// Returns (left, right) ambient light voltages. Higher = brighter.
    pub fn light(&mut self) -> Result<(f32, f32)> {
        Ok((self.adc.read_light_left()?, self.adc.read_light_right()?))
    }

    // ------------------------------------------------------------------
    // Lifecycle

    pub fn close(&mut self) -> Result<()> {
        self.motors().stop()?;
        self.pan_tilt().center()?;
        self.leds.clear()?;
        self.buzzer.off();
        Ok(())
    }
}
