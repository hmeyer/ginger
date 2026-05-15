//! Wire contract shared between the web server and the robot supervisor.
//!
//! One source of truth for the JSON the UI depends on and the command
//! protocol the hardware loop consumes.

use serde::{Deserialize, Serialize};

// ── Battery model ─────────────────────────────────────────────────────────────

// 2S LiPo: 8.4 V full, 6.0 V cutoff. Log data to refine these constants.
const BAT_FULL_V: f32 = 8.4;
const BAT_EMPTY_V: f32 = 6.0;

pub fn battery_pct(v: f32) -> u8 {
    ((v - BAT_EMPTY_V) / (BAT_FULL_V - BAT_EMPTY_V) * 100.0).clamp(0.0, 100.0) as u8
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
pub struct SensorSnapshot {
    pub battery_v: f32,
    pub battery_pct: u8,
    pub light_left: Option<f32>,
    pub light_right: Option<f32>,
    pub ir: Option<[bool; 3]>,
    pub us_cm: Option<f32>,
    pub ttc_s: Option<f32>, // estimated seconds to collision (None = not closing)
    pub explore_state: String,
    pub camera_fps: f32,
    pub exposure_us: i32,
    pub gain: f32,
    pub brightness: f32,
    pub luma: u8,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SensorConfig {
    pub light: bool,
    pub ir: bool,
    pub us: bool,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            light: true,
            ir: true,
            us: true,
        }
    }
}

// ── Command protocol ──────────────────────────────────────────────────────────

/// A request from the web layer to the robot supervisor (hardware thread).
pub enum Command {
    SetMotors { left: i32, right: i32 },
    Stop,
    SetPan(f32),
    SetTilt(f32),
    SetLed { r: u8, g: u8, b: u8 },
    LedOff,
    Buzzer(bool),
    SetSensors(SensorConfig),
    Scan,
    ExploreStart,
}

// ── Request bodies ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DriveBody {
    pub left: i32,
    pub right: i32,
}

#[derive(Deserialize)]
pub struct AngleBody {
    pub angle: f32,
}

#[derive(Deserialize)]
pub struct LedBody {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Deserialize)]
pub struct BuzzerBody {
    pub on: bool,
}
