//! Robot supervisor: the control loop that owns [`Car`] and mediates
//! between the web layer and the hardware.
//!
//! It drains a [`Command`] channel, runs autonomous exploration when
//! asked, polls sensors into a shared [`SensorSnapshot`], and enforces
//! the teleop safety behaviours (forward-collision lock with hysteresis,
//! time-to-collision estimate, pan/tilt auto-center, dead-man stop).
//!
//! The safety math is factored into pure helpers so it can be unit-tested
//! without any hardware.

use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use log::{info, warn};
use tokio::sync::mpsc;

use crate::api::{Command, SensorConfig, SensorSnapshot, battery_pct};
use crate::robot::{car::Car, explore, map::Map};

// Stop and lock out forward commands when closer than this.
const COLLISION_STOP_CM: f32 = 30.0;
// Hysteresis: unlock only after obstacle retreats past this.
const COLLISION_CLEAR_CM: f32 = 38.0;
// Auto-center pan/tilt after this much sustained forward travel.
// Both wheels must exceed this duty (of 4095) to count as "driving
// forward". Kept below the teleop forward duty (DUTY = 2000 in the web
// UI) so ordinary forward driving actually accumulates travel — at the
// old value of 2000 the strict `>` made the common case never trigger.
const AUTOCENTER_MIN_DUTY: i32 = 1500;
const AUTOCENTER_TRAVEL_CM: f32 = 10.0;
const MAX_SPEED_CMS: f32 = 100.0;
const POLL_PERIOD: Duration = Duration::from_millis(80);
const DEAD_MAN_TIMEOUT: Duration = Duration::from_millis(500);

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

/// Estimated forward travel (cm) over `dt` seconds for the given motor
/// command, or `None` when not driving forward hard enough to count
/// (which also resets the auto-center accumulator). `dt` is the *measured*
/// loop period, not a constant: the supervisor loop also does blocking
/// sensor I/O, so the real period runs well over the nominal 80 ms sleep.
fn fwd_travel_step(motor_left: i32, motor_right: i32, dt: f32) -> Option<f32> {
    if motor_left > AUTOCENTER_MIN_DUTY && motor_right > AUTOCENTER_MIN_DUTY {
        let avg_fraction = (motor_left + motor_right) as f32 / 2.0 / 4095.0;
        Some(avg_fraction * MAX_SPEED_CMS * dt)
    } else {
        None
    }
}

// ── Control loop ──────────────────────────────────────────────────────────────

/// Run the supervisor loop forever. Intended to own a dedicated thread.
pub fn run(
    mut cmd_rx: mpsc::Receiver<Command>,
    sensors: Arc<RwLock<SensorSnapshot>>,
    map: Arc<RwLock<Map>>,
    explore_stop: Arc<AtomicBool>,
) {
    let mut car = Car::new().expect("Car init failed");
    let mut config = SensorConfig::default();
    let mut last_drive = Instant::now();
    let mut is_driving = false;
    let mut explore_active = false;
    // Motor state for direction detection
    let mut motor_left: i32 = 0;
    let mut motor_right: i32 = 0;
    // Prevents re-applying forward commands while obstacle is in the way
    let mut obstacle_lock = false;
    // Previous US reading for TTC estimation
    let mut prev_us: Option<(f32, Instant)> = None;
    // Pan/tilt auto-center state
    let mut fwd_travel_est: f32 = 0.0;
    let mut pan_auto_centered = false;
    // Wall-clock of the last travel accumulation, for measured-dt
    // integration (real loop period varies with sensor I/O cost).
    let mut last_travel_t = Instant::now();
    // Last commanded bracket angles, mirrored into the snapshot so the
    // web UI can keep its camera joystick in sync (esp. when we
    // auto-center after forward driving).
    let mut cur_pan: f32 = 90.0;
    let mut cur_tilt: f32 = 90.0;

    loop {
        // ── Command queue ──────────────────────────────────────────────────────
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                Command::ExploreStart => {
                    info!("hw: exploration started");
                    explore_stop.store(false, Ordering::Relaxed);
                    explore_active = true;
                    is_driving = false;
                }
                Command::Stop => {
                    info!("hw: stop command — cancelling exploration");
                    explore_active = false;
                    explore_stop.store(true, Ordering::Relaxed);
                    obstacle_lock = false; // explicit stop always clears lock
                    motor_left = 0;
                    motor_right = 0;
                    if let Err(e) = car.stop() {
                        warn!("hw: stop error: {e}");
                    }
                    is_driving = false;
                }
                Command::SetMotors { left, right } => {
                    if explore_active {
                        info!("hw: manual drive — cancelling exploration");
                    }
                    explore_active = false;
                    explore_stop.store(true, Ordering::Relaxed);
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
                    pan_auto_centered = false;
                    fwd_travel_est = 0.0;
                }
                Command::SetTilt(a) => {
                    car.pan_tilt().set_tilt(a).ok();
                    cur_tilt = a;
                    pan_auto_centered = false;
                    fwd_travel_est = 0.0;
                }
                Command::SetLed { r, g, b } => {
                    car.leds.set_all(r, g, b);
                    car.leds.show().ok();
                }
                Command::LedOff => {
                    car.leds.clear().ok();
                }
                Command::Buzzer(on) => {
                    if on {
                        car.buzzer.on()
                    } else {
                        car.buzzer.off()
                    }
                }
                Command::SetSensors(cfg) => {
                    config = cfg;
                }
                Command::Scan => {
                    let noop = AtomicBool::new(false);
                    let rays = explore::do_scan(&mut car, &noop);
                    map.write().unwrap().integrate_scan(&rays);
                }
            }
        }

        // ── Exploration loop ───────────────────────────────────────────────────
        if explore_active {
            sensors.write().unwrap().explore_state = "scanning".into();
            let status = explore::tick(&mut car, &map, &explore_stop);
            info!("explore: tick → {status}");
            {
                let mut snap = sensors.write().unwrap();
                snap.explore_state = status.to_string();
                let v = car.battery_v().unwrap_or(snap.battery_v);
                snap.battery_v = v;
                snap.battery_pct = battery_pct(v);
            }
            if status == explore::Status::Complete || explore_stop.load(Ordering::Relaxed) {
                info!("explore: stopped (status={status})");
                explore_active = false;
                explore_stop.store(false, Ordering::Relaxed);
            }
            continue; // skip normal sensor poll + sleep
        }

        // ── Normal sensor poll ────────────────────────────────────────────────
        let battery_v = car.battery_v().unwrap_or(0.0);
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

        // Auto-center pan when sustained forward motion detected.
        // Ensures US sensor faces forward for collision detection.
        let travel_dt = now_t.duration_since(last_travel_t).as_secs_f32();
        last_travel_t = now_t;
        if let Some(inc) = fwd_travel_step(motor_left, motor_right, travel_dt) {
            fwd_travel_est += inc;
            if fwd_travel_est > AUTOCENTER_TRAVEL_CM && !pan_auto_centered {
                let mut pt = car.pan_tilt();
                pt.set_pan(90.0).ok();
                pt.set_tilt(90.0).ok();
                cur_pan = 90.0;
                cur_tilt = 90.0;
                pan_auto_centered = true;
                info!(
                    "hw: auto-centered pan+tilt after ~{:.0}cm forward travel",
                    fwd_travel_est
                );
            }
        } else {
            fwd_travel_est = 0.0;
            pan_auto_centered = false;
        }

        let battery_pct = battery_pct(battery_v);
        info!("bat: {battery_v:.3} V  {battery_pct}%");
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
            explore_state: "idle".into(),
            camera_fps: 0.0, // filled by SSE handler
            exposure_us: 0,  // filled by SSE handler
            gain: 0.0,       // filled by SSE handler
            brightness: 0.0, // filled by SSE handler
            luma: 0,         // filled by SSE handler
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

    #[test]
    fn fwd_travel_only_counts_strong_forward() {
        let dt = 0.080;
        assert!(fwd_travel_step(0, 0, dt).is_none());
        assert!(fwd_travel_step(1000, 1000, dt).is_none()); // below threshold
        assert!(fwd_travel_step(3000, 1000, dt).is_none()); // one wheel too slow
        // The web UI drives forward at exactly DUTY = 2000; that must
        // accumulate travel (regression guard for the auto-center bug).
        assert!(fwd_travel_step(2000, 2000, dt).is_some());
        let inc = fwd_travel_step(4095, 4095, dt).unwrap();
        assert!((inc - MAX_SPEED_CMS * dt).abs() < 1e-3);
    }
}
