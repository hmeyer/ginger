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
use std::time::{Duration, Instant, SystemTime};

use log::{info, warn};
use tokio::sync::mpsc;

use crate::api::{Command, SensorConfig, SensorSnapshot, battery_pct};
use crate::robot::car::Car;

// Stop and lock out forward commands when closer than this.
const COLLISION_STOP_CM: f32 = 30.0;
// Hysteresis: unlock only after obstacle retreats past this.
const COLLISION_CLEAR_CM: f32 = 38.0;
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

// ── Buzzer & LED "personality" ────────────────────────────────────────────────

/// Tiny dependency-free xorshift PRNG, seeded from the wall clock. Good
/// enough to make the buzzer/LED show feel different every press.
struct Rng(u64);
impl Rng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Rng(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
    /// Uniform float in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// HSV → RGB. `h` in degrees, `s`/`v` in `[0, 1]`.
fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let q = |f: f32| ((f + m) * 255.0).round() as u8;
    (q(r), q(g), q(b))
}

/// A pitch near the phrase's (drifting) centre, jittered within the band.
fn band_freq(rng: &mut Rng, centre: f32, lo: i32, hi: i32) -> i32 {
    let j = (rng.unit() - 0.5) * (hi - lo) as f32 * 0.6;
    (centre + j).clamp(lo as f32, hi as f32) as i32
}

/// Play a short R2D2-ish warble with a randomly chosen *mood*. Each mood
/// biases the pitch band, tempo and overall rise/fall contour so
/// successive presses sound like different little exclamations rather
/// than the same random noise. Pitch is bit-banged on the buzzer pin
/// (see [`crate::hal::buzzer`]); blocks the supervisor loop < ~1.3 s.
fn play_buzzer_tune(car: &mut Car) {
    let mut rng = Rng::new();

    // (lo_hz, hi_hz, tempo%, contour −1/0/+1, segment kinds)
    // kinds: 0 gliss · 1 warble · 2 sustained · 3 blips · 4 tight trill
    let moods: [(i32, i32, u64, i32, &[u8]); 5] = [
        (900, 3000, 70, 1, &[3, 3, 0, 4]),  // excited: high, fast, rising
        (450, 1800, 115, 1, &[3, 2, 0]),    // curious: mid, questioning
        (150, 850, 155, -1, &[2, 0, 3]),    // grumpy: low, slow, falling
        (700, 2400, 60, 0, &[4, 4, 3]),     // alarmed: harsh fast trills
        (250, 2600, 100, 0, &[0, 1, 2, 3]), // chatty: wide, mixed
    ];
    let (lo, hi, tempo, contour, kinds) = moods[rng.range(0, 4) as usize];

    let segments = rng.range(4, 8);
    let mut centre = match contour {
        c if c > 0 => lo as f32,
        c if c < 0 => hi as f32,
        _ => (lo + hi) as f32 / 2.0,
    };
    let drift = contour as f32 * (hi - lo) as f32 / segments as f32;
    let dur = |rng: &mut Rng, a: u64, b: u64| (rng.range(a, b) * tempo / 100).max(6);

    for _ in 0..segments {
        match kinds[rng.range(0, kinds.len() as u64 - 1) as usize] {
            // gliss: smooth pitch sweep
            0 => {
                let (f0, f1) = (
                    band_freq(&mut rng, centre, lo, hi),
                    band_freq(&mut rng, centre, lo, hi),
                );
                let steps = 12;
                let total = dur(&mut rng, 90, 200);
                for s in 0..steps {
                    let fr = f0 + (f1 - f0) * s as i32 / steps as i32;
                    car.buzzer.tone(fr.max(60) as u32, (total / steps).max(4));
                }
            }
            // warble: alternate two pitches
            1 => {
                let (a, b) = (
                    band_freq(&mut rng, centre, lo, hi),
                    band_freq(&mut rng, centre, lo, hi),
                );
                for i in 0..rng.range(4, 8) {
                    car.buzzer.tone(
                        if i.is_multiple_of(2) { a } else { b }.max(60) as u32,
                        dur(&mut rng, 14, 28),
                    );
                }
            }
            // one sustained chirp
            2 => car.buzzer.tone(
                band_freq(&mut rng, centre, lo, hi).max(60) as u32,
                dur(&mut rng, 80, 170),
            ),
            // tight trill on two close pitches
            4 => {
                let base = band_freq(&mut rng, centre, lo, hi);
                let b = base + rng.range(120, 380) as i32;
                for i in 0..rng.range(6, 12) {
                    car.buzzer.tone(
                        if i.is_multiple_of(2) { base } else { b }.max(60) as u32,
                        dur(&mut rng, 9, 18),
                    );
                }
            }
            // short blips with little gaps
            _ => {
                for _ in 0..rng.range(1, 3) {
                    car.buzzer.tone(
                        band_freq(&mut rng, centre, lo, hi).max(60) as u32,
                        dur(&mut rng, 20, 70),
                    );
                    car.buzzer.tone(0, dur(&mut rng, 15, 40));
                }
            }
        }
        centre = (centre + drift).clamp(lo as f32, hi as f32);
        // A short pause between most segments.
        if rng.range(0, 2) != 0 {
            car.buzzer.tone(0, dur(&mut rng, 25, 110));
        }
    }
    car.buzzer.off();
}

fn led_frame(car: &mut Car, ms: u64) {
    car.leds.show().ok();
    thread::sleep(Duration::from_millis(ms));
}

/// Occasionally blink the strip dark for a short random beat, so the
/// animations breathe instead of running at a constant pace.
fn led_pause(car: &mut Car, rng: &mut Rng) {
    if rng.range(0, 7) == 0 {
        car.leds.set_all(0, 0, 0);
        car.leds.show().ok();
        thread::sleep(Duration::from_millis(rng.range(70, 200)));
    }
}

/// A colour palette: `at(t, v)` maps a phase `t` to an RGB at value `v`.
/// Picking one per show is what makes runs look different beyond "the
/// same full-rainbow again" — single-hue, duo, warm, cool, etc.
#[derive(Clone, Copy)]
struct Pal {
    h0: f32,
    span: f32,
    sat: f32,
}
impl Pal {
    fn at(&self, t: f32, v: f32) -> (u8, u8, u8) {
        hsv(self.h0 + t * self.span, self.sat, v.clamp(0.0, 1.0))
    }
}
fn pick_pal(rng: &mut Rng) -> Pal {
    let h0 = rng.unit() * 360.0;
    match rng.range(0, 5) {
        0 => Pal {
            h0,
            span: 360.0,
            sat: 1.0,
        }, // full rainbow
        1 => Pal {
            h0,
            span: 14.0,
            sat: 1.0,
        }, // single hue
        2 => Pal {
            h0,
            span: 70.0,
            sat: 1.0,
        }, // analogous
        3 => Pal {
            h0,
            span: 180.0,
            sat: 1.0,
        }, // complementary duo
        4 => Pal {
            h0: rng.unit() * 45.0,
            span: 55.0,
            sat: 1.0,
        }, // warm / fire
        _ => Pal {
            h0: 180.0 + rng.unit() * 100.0,
            span: 80.0,
            sat: 0.85,
        }, // cool / ice
    }
}

/// Play a short, randomized per-pixel LED animation, then clear. A
/// palette, speed and brightness are chosen once per press (so the same
/// motion looks different each time); length, pacing and mid-show pauses
/// are randomized too. Blocks the supervisor loop briefly (~1.5–3 s).
fn play_led_show(car: &mut Car) {
    use crate::hal::led::LED_COUNT as N;
    let mut rng = Rng::new();
    let pal = pick_pal(&mut rng);
    let speed = rng.range(55, 150); // % of base pace
    let bright = if rng.range(0, 3) == 0 { 0.5 } else { 1.0 };
    let scale = |base: u64| (base * speed / 100).max(8);

    match rng.range(0, 4) {
        // Comet looping around the ring with a fading tail.
        0 => {
            let mut ph = rng.unit();
            for step in 0..(N as u64 * rng.range(2, 5)) {
                car.leds.set_all(0, 0, 0);
                let head = (step as usize) % N;
                for t in 0..4 {
                    let idx = (head + N - t) % N;
                    let v = (1.0 - t as f32 * 0.28).max(0.0) * bright;
                    let (r, g, b) = pal.at(ph, v);
                    car.leds.set(idx, r, g, b);
                }
                ph += 0.05;
                led_frame(car, scale(45));
                led_pause(car, &mut rng);
            }
        }
        // Loopy snake: a solid body slithering around the ring.
        1 => {
            let len = rng.range(2, 4) as usize;
            let mut ph = rng.unit();
            for step in 0..(N as u64 * rng.range(2, 5)) {
                car.leds.set_all(0, 0, 0);
                let head = (step as usize) % N;
                for k in 0..len {
                    let idx = (head + N - k) % N;
                    let v = (1.0 - k as f32 / len as f32 * 0.55) * bright;
                    let (r, g, b) = pal.at(ph + k as f32 * 0.06, v);
                    car.leds.set(idx, r, g, b);
                }
                ph += 0.04;
                led_frame(car, scale(52));
                led_pause(car, &mut rng);
            }
        }
        // Sparkles: random pixels pop and decay.
        2 => {
            let mut px = [(0.0f32, 0.0f32); N]; // (phase, value)
            for _ in 0..rng.range(32, 60) {
                for _ in 0..rng.range(1, 2) {
                    let i = rng.range(0, N as u64 - 1) as usize;
                    px[i] = (rng.unit(), 1.0);
                }
                for (i, (p, v)) in px.iter_mut().enumerate() {
                    *v *= 0.80;
                    let (r, g, b) = pal.at(*p, *v * bright);
                    car.leds.set(i, r, g, b);
                }
                led_frame(car, scale(45));
                led_pause(car, &mut rng);
            }
        }
        // Travelling brightness wave (pulses even on a single-hue palette).
        3 => {
            let mut off = 0.0f32;
            for _ in 0..rng.range(30, 55) {
                for i in 0..N {
                    let t = i as f32 / N as f32;
                    let wave = 0.5 + 0.5 * (off + i as f32 * 0.9).sin();
                    let v = (0.25 + 0.75 * wave) * bright;
                    let (r, g, b) = pal.at(t, v);
                    car.leds.set(i, r, g, b);
                }
                off += 0.45;
                led_frame(car, scale(45));
            }
        }
        // Theatre chase: every third pixel, colour drifting.
        _ => {
            let mut ph = rng.unit();
            for fr in 0..rng.range(18, 36) {
                let (r, g, b) = pal.at(ph, bright);
                for i in 0..N {
                    if (i + fr as usize).is_multiple_of(3) {
                        car.leds.set(i, r, g, b);
                    } else {
                        car.leds.set(i, 0, 0, 0);
                    }
                }
                ph += 0.05;
                led_frame(car, scale(70));
                led_pause(car, &mut rng);
            }
        }
    }
    car.leds.clear().ok();
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
                Command::LedShow => {
                    play_led_show(&mut car);
                }
                Command::BuzzerTune => {
                    play_buzzer_tune(&mut car);
                }
                Command::SetSensors(cfg) => {
                    config = cfg;
                }
            }
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

        // The bracket is kept forward by the web UI's spring-back camera
        // joystick (re-centers pan/tilt on release), so the supervisor no
        // longer estimates forward travel to auto-center it here.

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
}
