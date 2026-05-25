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
    pub pan: f32,           // current pan-bracket angle, degrees (UI joystick sync)
    pub tilt: f32,          // current tilt-bracket angle, degrees
    pub camera_fps: f32,
    pub exposure_us: i32,
    pub gain: f32,
    pub brightness: f32,
    pub luma: u8,
    /// IMU gyro vector in deg/s (IMU body frame). `None` when the BMI160
    /// wasn't reachable at boot — same gate as `/api/imu/sample` 503.
    pub imu_gyro_dps: Option<[f32; 3]>,
    /// IMU accelerometer vector in m/s² (IMU body frame, gravity included).
    pub imu_accel_mps2: Option<[f32; 3]>,
    /// Achieved polling rate (EWMA). Should sit ~200 Hz; sagging means
    /// the I²C bus is busy (PCA9685/ADS7830 contention).
    pub imu_rate_hz: Option<f32>,
    /// Signed ms gap between latest camera frame arrival and latest IMU
    /// sample, on the same monotonic clock: `frame_ago_ms − sample_ago_ms`.
    /// Positive = frame is older (typical with a 200 Hz IMU and ~10 Hz
    /// camera, so this should sit in roughly `[0, camera_period_ms]`).
    /// A steady drift here would break Stage 4's gyro pre-integration —
    /// this is the canary surfaced in the WebUI.
    pub imu_frame_sync_ms: Option<f32>,
    /// Most recently applied motor PWM (PCA9685 duty in
    /// `[-MAX_DUTY, MAX_DUTY]`). Surfaced so the Stage-2 label worker
    /// (`src/motion/labels.rs`) can pair each 200 ms training window
    /// with the command that produced its observed motion — without
    /// subscribing to the supervisor's command channel directly. Zero
    /// at boot until the first `Command::SetMotors`.
    pub pwm_l_cmd: i32,
    pub pwm_r_cmd: i32,
}

impl SensorSnapshot {
    /// Startup state before the first sensor poll. Exposure/gain seed the
    /// camera AE loop's initial guess.
    pub fn initial() -> Self {
        Self {
            battery_v: 0.0,
            battery_pct: 0,
            light_left: None,
            light_right: None,
            ir: None,
            us_cm: None,
            ttc_s: None,
            pan: 90.0,
            tilt: 90.0,
            camera_fps: 0.0,
            exposure_us: 8_000,
            gain: 8.0,
            brightness: 0.0,
            luma: 0,
            imu_gyro_dps: None,
            imu_accel_mps2: None,
            imu_rate_hz: None,
            imu_frame_sync_ms: None,
            pwm_l_cmd: 0,
            pwm_r_cmd: 0,
        }
    }
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
    /// Raw PWM per side, clamped to `-MAX_DUTY..MAX_DUTY` (±4095) by the
    /// motor driver. Positive = forward, negative = reverse. The WebUI
    /// joystick saturates at ±2000 (see `DUTY` in `bin/web/index.html`);
    /// CLI / test callers should treat 2000 as "full" and scale from there.
    SetMotors {
        left: i32,
        right: i32,
    },
    Stop,
    SetPan(f32),
    SetTilt(f32),
    /// Play a short randomized synchronized LED + buzzer emote.
    Emote,
    SetSensors(SensorConfig),
}

// ── Request bodies ────────────────────────────────────────────────────────────

/// Body of `POST /api/drive`. `left` / `right` are raw PWM, see
/// [`Command::SetMotors`] for the scale (±4095 hard cap, ±2000 = "full"
/// by WebUI convention).
#[derive(Deserialize)]
pub struct DriveBody {
    pub left: i32,
    pub right: i32,
}

#[derive(Deserialize)]
pub struct AngleBody {
    pub angle: f32,
}

// ── IMU sample (debug endpoint) ──────────────────────────────────────────────

/// Wire format for `GET /api/imu/sample`: the latest BMI160 reading
/// alongside the *current* host clock and the latest camera frame's
/// capture time, so a caller can directly read off how far in the past
/// each event was without doing any clock-domain conversion themselves.
///
/// `t_*_ago_ms` are all positive; they are `now - t_event` in
/// milliseconds. The interesting one for sync verification is
/// `frame_to_sample_ms = t_frame_capture_ago_ms - t_sample_ago_ms`,
/// which is the host-monotonic gap between the latest camera frame and
/// the latest IMU sample. With a 200 Hz IMU and a 10–30 Hz camera, this
/// should be roughly uniform in `[0, 5 ms]`.
#[derive(Clone, Serialize)]
pub struct ImuSampleView {
    pub gyro_dps: [f32; 3],
    pub accel_mps2: [f32; 3],
    /// Chip-internal 24-bit counter ticks (39.0625 µs each); a stalled
    /// sample stream shows up as this not advancing across requests.
    pub sensortime: u32,
    /// Achieved sample rate EWMA in Hz.
    pub rate_hz: f32,
    /// Milliseconds between the IMU sample and "now" at request time.
    pub t_sample_ago_ms: f32,
    /// Milliseconds between the latest camera frame arrival and "now".
    /// `None` if no camera frame has been captured yet.
    pub t_frame_capture_ago_ms: Option<f32>,
    /// Signed gap between the two: positive means the frame is older
    /// than the IMU sample (typical, since the IMU is 200 Hz vs the
    /// camera's ~10 Hz). Same as `t_frame_capture_ago_ms - t_sample_ago_ms`.
    pub frame_to_sample_ms: Option<f32>,
}
