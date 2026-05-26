//! Wire contract shared between the web server and the robot supervisor.
//!
//! One source of truth for the JSON the UI depends on and the command
//! protocol the hardware loop consumes.

use serde::{Deserialize, Serialize};

use crate::hal::bno055::CalibStatus;

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
    /// IMU yaw (chip-frame, degrees, wrapped to `(-180, 180]`). `None`
    /// when the BNO055 wasn't reachable at boot OR is still in its
    /// fusion warm-up (`calib.gyr < 1`) — same gate as `/api/imu/sample`
    /// 503. Pitch and roll are exposed on `/api/imu/sample` for the
    /// debug card; only yaw lives on the SSE snapshot.
    pub imu_yaw_deg: Option<f32>,
    /// Linear acceleration (gravity removed by the chip) in m/s², chip
    /// body frame. Stays near zero on a still chassis; spikes signal
    /// physical impulse (bump, kick, pickup) and drive the label
    /// worker's spike-rejection counter.
    pub imu_linear_accel_mps2: Option<[f32; 3]>,
    /// Achieved polling rate (EWMA). Should sit ~100 Hz (BNO055 fusion
    /// rate); sagging means I²C contention.
    pub imu_rate_hz: Option<f32>,
    /// Per-subsystem calibration status (`sys/gyr/acc/mag`, each in
    /// `0..=3`). `mag` stays at zero in IMUPLUS by design. Drives the
    /// WebUI's "fusion ready" badge.
    pub imu_calib: Option<CalibStatus>,
    /// Signed ms gap between latest camera frame arrival and latest IMU
    /// sample, on the same monotonic clock: `frame_ago_ms − sample_ago_ms`.
    /// Positive = frame is older. A steady drift here would break the
    /// SLAM rotation hint — canary surfaced in the WebUI.
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
            imu_yaw_deg: None,
            imu_linear_accel_mps2: None,
            imu_rate_hz: None,
            imu_calib: None,
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

/// Wire format for `GET /api/imu/sample`: the latest BNO055 fusion
/// snapshot alongside the *current* host clock and the latest camera
/// frame's capture time, so a caller can directly read off how far in
/// the past each event was without doing any clock-domain conversion
/// themselves.
///
/// The orientation is the chip's IMUPLUS-fused absolute orientation in
/// the chip body frame; it's drift-compensated by the chip's fusion
/// engine, so the same chassis pose produces the same quaternion
/// across long-horizon observations (modulo the chip's auto-zero pass,
/// which happens during the first stillness after boot).
#[derive(Clone, Serialize)]
pub struct ImuSampleView {
    /// Unit quaternion as `[w, x, y, z]`. The canonical orientation
    /// representation; yaw/pitch/roll below are derived from it for
    /// human-readable WebUI display.
    pub orientation_quat: [f32; 4],
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub roll_deg: f32,
    /// Gravity-removed linear acceleration, chip body frame, m/s².
    pub linear_accel_mps2: [f32; 3],
    /// Per-subsystem calibration status (each `0..=3`).
    pub calib: CalibStatus,
    /// Achieved fusion polling rate EWMA in Hz.
    pub rate_hz: f32,
    /// Monotonic per-sample counter incremented by the polling thread.
    /// A stalled sample stream shows as this not advancing across
    /// requests — cheapest "is the IMU thread alive" check we have.
    pub sample_index: u32,
    /// Milliseconds between the IMU sample and "now" at request time.
    pub t_sample_ago_ms: f32,
    /// Milliseconds between the latest camera frame arrival and "now".
    /// `None` if no camera frame has been captured yet.
    pub t_frame_capture_ago_ms: Option<f32>,
    /// Signed gap between the two: positive means the frame is older
    /// than the IMU sample. With a 100 Hz IMU and ~10 Hz camera this
    /// should sit in roughly `[0, 10 ms]`.
    pub frame_to_sample_ms: Option<f32>,
}
