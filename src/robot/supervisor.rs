//! Robot supervisor: the control loop that owns [`Car`] and mediates
//! between the web layer and the hardware.
//!
//! It drains a [`Command`] channel, polls sensors into a shared
//! [`SensorSnapshot`], and enforces the teleop safety behaviours
//! (forward-collision lock with hysteresis, time-to-collision estimate,
//! dead-man stop).
//!
//! The safety math is factored into pure helpers so it can be unit-tested
//! without any hardware.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use log::{info, warn};
use tokio::sync::mpsc;

use crate::api::{Command, SensorConfig, SensorSnapshot, battery_pct};
use crate::robot::car::Car;
use crate::robot::emote;

// Stop and lock out forward commands when closer than this.
const COLLISION_STOP_CM: f32 = 30.0;
// Hysteresis: unlock only after obstacle retreats past this.
const COLLISION_CLEAR_CM: f32 = 38.0;
const POLL_PERIOD: Duration = Duration::from_millis(80);
const DEAD_MAN_TIMEOUT: Duration = Duration::from_millis(500);
// Battery-voltage low-pass: the raw ADC reading swings several percent
// per poll from motor-load sag + ADC noise. At the ~12 Hz poll this
// gives a ~1.5 s time constant — enough to settle the reported value.
const BATT_EWMA_ALPHA: f32 = 0.05;

// ── Pure safety helpers ───────────────────────────────────────────────────────

/// Time-to-collision estimate (seconds) from two successive ultrasonic
/// readings `dt` apart. `None` unless approaching faster than 2 cm/s.
fn time_to_collision(d_prev: f32, d_now: f32, dt: f32) -> Option<f32> {
    if dt <= 0.0 {
        return None;
    }
    let closing = (d_prev - d_now) / dt; // cm/s, positive = approaching
    if closing > 2.0 {
        Some(d_now / closing)
    } else {
        None
    }
}

/// Outcome of one collision-stop evaluation.
struct CollisionDecision {
    /// Whether to hard-stop the motors this tick.
    stop: bool,
    /// The obstacle lock state to carry forward.
    lock: bool,
}

/// Forward-collision state machine with hysteresis: stop + lock when an
/// obstacle is inside the stop threshold while driving forward; clear the
/// lock only once it retreats past the (larger) clear threshold.
fn collision_step(going_forward: bool, us_cm: Option<f32>, lock: bool) -> CollisionDecision {
    let mut stop = false;
    let mut lock = lock;
    if going_forward && us_cm.is_some_and(|d| d < COLLISION_STOP_CM) {
        stop = true;
        lock = true;
    }
    if lock && us_cm.is_some_and(|d| d > COLLISION_CLEAR_CM) {
        lock = false;
    }
    CollisionDecision { stop, lock }
}

// ── Control loop ──────────────────────────────────────────────────────────────

/// Run the supervisor loop forever. Intended to own a dedicated thread.
pub fn run(mut cmd_rx: mpsc::Receiver<Command>, sensors: Arc<RwLock<SensorSnapshot>>) {
    let mut car = Car::new().expect("Car init failed");
    let mut config = SensorConfig::default();
    let mut last_drive = Instant::now();
    let mut is_driving = false;
    // Motor state for direction detection
    let mut motor_left: i32 = 0;
    let mut motor_right: i32 = 0;
    // Prevents re-applying forward commands while obstacle is in the way
    let mut obstacle_lock = false;
    // Previous US reading for TTC estimation
    let mut prev_us: Option<(f32, Instant)> = None;
    // Low-pass-filtered battery voltage + the last percentage logged.
    // The poll runs ~12×/s and the raw reading is noisy, so we filter
    // the voltage and log only when the integer percentage changes.
    let mut batt_v_ewma: Option<f32> = None;
    let mut last_batt_pct: Option<u8> = None;
    // Last commanded bracket angles, mirrored into the snapshot so the
    // web UI can keep its camera joystick in sync.
    let mut cur_pan: f32 = 90.0;
    let mut cur_tilt: f32 = 90.0;

    loop {
        // ── Command queue ──────────────────────────────────────────────────────
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                Command::Stop => {
                    info!("hw: stop command");
                    obstacle_lock = false; // explicit stop always clears lock
                    motor_left = 0;
                    motor_right = 0;
                    if let Err(e) = car.stop() {
                        warn!("hw: stop error: {e}");
                    }
                    is_driving = false;
                }
                Command::SetMotors { left, right } => {
                    // Any backward component unlocks the obstacle stop
                    if left < 0 || right < 0 {
                        obstacle_lock = false;
                    }
                    let going_forward = left > 0 && right > 0;
                    if obstacle_lock && going_forward {
                        // silently ignore: obstacle still in the way.
                        // Do NOT update motor_left/right so going_forward stays
                        // false in the sensor poll and the warning doesn't loop.
                    } else {
                        motor_left = left;
                        motor_right = right;
                        if let Err(e) = car.motors().drive(left, right) {
                            warn!("hw: drive({left},{right}) error: {e}");
                        }
                        last_drive = Instant::now();
                        is_driving = left != 0 || right != 0;
                    }
                }
                Command::SetPan(a) => {
                    car.pan_tilt().set_pan(a).ok();
                    cur_pan = a;
                }
                Command::SetTilt(a) => {
                    car.pan_tilt().set_tilt(a).ok();
                    cur_tilt = a;
                }
                Command::Emote => {
                    emote::play_emote(&mut car);
                }
                Command::SetSensors(cfg) => {
                    config = cfg;
                }
            }
        }

        // ── Normal sensor poll ────────────────────────────────────────────────
        // Low-pass the noisy battery reading; skip failed reads so a
        // transient zero can't yank the filtered average down.
        if let Ok(v) = car.battery_v() {
            batt_v_ewma = Some(match batt_v_ewma {
                Some(prev) => prev + BATT_EWMA_ALPHA * (v - prev),
                None => v,
            });
        }
        let battery_v = batt_v_ewma.unwrap_or(0.0);
        let (light_left, light_right) = if config.light {
            car.light()
                .map(|(l, r)| (Some(l), Some(r)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };
        let ir = if config.ir {
            let (l, c, r) = car.ir.read_all();
            Some([l, c, r])
        } else {
            None
        };
        let us_cm = if config.us {
            car.us().distance_cm()
        } else {
            None
        };

        // ── TTC estimation ─────────────────────────────────────────────────────
        let now_t = Instant::now();
        let ttc_s = if let Some(d_now) = us_cm {
            let result = prev_us.and_then(|(d_prev, t_prev)| {
                let dt = now_t.duration_since(t_prev).as_secs_f32();
                time_to_collision(d_prev, d_now, dt)
            });
            prev_us = Some((d_now, now_t));
            result
        } else {
            prev_us = None;
            None
        };

        // ── Collision stop ────────────────────────────────────────────────────
        let going_forward = motor_left > 0 && motor_right > 0;
        let decision = collision_step(going_forward, us_cm, obstacle_lock);
        if decision.stop {
            warn!("hw: collision stop — obstacle at {:.0}cm", us_cm.unwrap());
            car.stop().ok();
            motor_left = 0;
            motor_right = 0;
            is_driving = false;
        }
        obstacle_lock = decision.lock;

        // The bracket is kept forward by the web UI's spring-back camera
        // joystick (re-centers pan/tilt on release), so the supervisor no
        // longer estimates forward travel to auto-center it here.

        let battery_pct = battery_pct(battery_v);
        // Battery drains slowly; log only when the integer percentage
        // moves so the journal stays readable (still catches every
        // meaningful change). The live value is always in the snapshot.
        if last_batt_pct != Some(battery_pct) {
            info!("bat: {battery_v:.3} V  {battery_pct}%");
            last_batt_pct = Some(battery_pct);
        }
        *sensors.write().unwrap() = SensorSnapshot {
            battery_v,
            battery_pct,
            light_left,
            light_right,
            ir,
            us_cm,
            ttc_s,
            pan: cur_pan,
            tilt: cur_tilt,
            camera_fps: 0.0,         // filled by SSE handler
            exposure_us: 0,          // filled by SSE handler
            gain: 0.0,               // filled by SSE handler
            brightness: 0.0,         // filled by SSE handler
            luma: 0,                 // filled by SSE handler
            imu_gyro_dps: None,      // filled by SSE handler
            imu_accel_mps2: None,    // filled by SSE handler
            imu_rate_hz: None,       // filled by SSE handler
            imu_frame_sync_ms: None, // filled by SSE handler
        };

        // Safety stop if motors have been spinning with no command for 500 ms
        if is_driving && last_drive.elapsed() > DEAD_MAN_TIMEOUT {
            car.stop().ok();
            is_driving = false;
        }

        thread::sleep(POLL_PERIOD);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttc_reports_when_closing() {
        // 100 → 90 cm in 0.1 s = 100 cm/s closing; ttc = 90/100 = 0.9 s.
        let ttc = time_to_collision(100.0, 90.0, 0.1).unwrap();
        assert!((ttc - 0.9).abs() < 1e-4);
    }

    #[test]
    fn ttc_none_when_not_closing() {
        assert!(time_to_collision(90.0, 90.5, 0.1).is_none()); // receding
        assert!(time_to_collision(90.0, 89.99, 0.1).is_none()); // < 2 cm/s
        assert!(time_to_collision(100.0, 50.0, 0.0).is_none()); // dt guard
    }

    #[test]
    fn collision_locks_on_close_obstacle_while_forward() {
        let d = collision_step(true, Some(20.0), false);
        assert!(d.stop);
        assert!(d.lock);
    }

    #[test]
    fn collision_does_not_trigger_when_not_forward() {
        let d = collision_step(false, Some(10.0), false);
        assert!(!d.stop);
        assert!(!d.lock);
    }

    #[test]
    fn collision_lock_holds_through_hysteresis_band() {
        // Locked; obstacle now at 35 cm — past stop (30) but not yet past
        // clear (38). Lock must persist.
        let d = collision_step(false, Some(35.0), true);
        assert!(!d.stop);
        assert!(d.lock);
        // Retreats past the clear threshold → unlock.
        let d = collision_step(false, Some(40.0), true);
        assert!(!d.lock);
    }

    #[test]
    fn collision_lock_persists_when_distance_unknown() {
        let d = collision_step(false, None, true);
        assert!(d.lock);
    }
}
