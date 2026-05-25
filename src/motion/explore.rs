//! Stage 4 of `PLAN.md`: greedy exploration controller driven by a
//! swept-ultrasonic local scan.
//!
//! The behaviour is dead-simple by design — Stage 4 is the "does the
//! robot move usefully without operator input?" milestone, not a
//! global planner:
//!
//! ```text
//!   Idle ─(POST /api/motion/explore?on=1)→ Scanning
//!   Scanning → BoxedIn       (no usable heading found)
//!   Scanning → Turning(θ*)   (heading picked from polar scan)
//!   Turning(θ*) → Driving(d) (gyro confirmed chassis aligned)
//!   Driving(d) → Scanning    (travelled d, or ultrasonic too close,
//!                             or pose distance covered)
//!   any → Idle               (POST /api/motion/explore?on=0)
//! ```
//!
//! ## Swept scan
//!
//! `PAN_FIRST` → `PAN_LAST` in `PAN_STEP_DEG` increments. Each step:
//!   1. `Command::SetPan(angle)` via the existing supervisor channel.
//!   2. Sleep `SERVO_SETTLE_MS` for the bracket to land.
//!   3. Snapshot `sensors.us_cm`.
//!
//! The scan blocks the controller thread for ~3 s per waypoint (15
//! readings × 250 ms). Acceptable: scanning is a once-per-waypoint
//! cost, not a continuous overhead. Chassis stays still throughout
//! (no motor command issued).
//!
//! ## Best-heading pick
//!
//! For each angle bin, look at the minimum distance in a `±20°`
//! window centred on the bin. Pick the bin with the largest such
//! min. Tiebreak toward `90°` (current chassis heading) so the
//! controller doesn't thrash between near-equal options.
//!
//! ## Driving
//!
//! `Turning(θ*)` sets `omega_target = +sign(Δθ) × W_TURN_RAD_S`,
//! issues motion drive, watches pose. When `|pose.theta − θ_target|`
//! shrinks below `TURN_TOLERANCE_RAD`, stop and proceed to Driving.
//!
//! `Driving(d)` sets `v_target = V_FORWARD_M_S`. Stop when any of:
//!   - chassis has moved `d` metres (from pose deltas),
//!   - ultrasonic falls below `STOP_DIST_CM`,
//!   - operator hits `POST /api/motion/explore?on=0`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use log::{info, warn};
use serde::Serialize;
use tokio::sync::mpsc;

use crate::api::{Command, SensorSnapshot};
use crate::motion::{ModelInput, MotionTarget, MotorModel, PoseState};

// ── Sweep parameters ──────────────────────────────────────────────────────────

const PAN_FIRST_DEG: f32 = 15.0;
const PAN_LAST_DEG: f32 = 165.0;
const PAN_STEP_DEG: f32 = 10.0;
const SERVO_SETTLE_MS: u64 = 250;
const PAN_CENTER_DEG: f32 = 90.0;

// ── Best-heading window ───────────────────────────────────────────────────────

const HEADING_WINDOW_HALF_DEG: f32 = 20.0;
/// Minimum max-min distance we'll consider "drivable". Below this the
/// chassis is hemmed in; emit `BoxedIn`.
const BOXED_IN_CM: f32 = 50.0;

// ── Driving parameters ───────────────────────────────────────────────────────

const V_FORWARD_M_S: f32 = 0.25;
const W_TURN_RAD_S: f32 = 0.6;
/// Aim for ≤ ~3° heading error before we commit to driving.
const TURN_TOLERANCE_RAD: f32 = 0.05;
const TURN_TIMEOUT_S: f32 = 5.0;
/// Stop a driving leg if ultrasonic shows anything this close. Above
/// the supervisor's 30 cm hard stop, with some margin for braking.
const STOP_DIST_CM: f32 = 40.0;
/// Max distance per driving leg before mandatory re-scan.
const MAX_LEG_M: f32 = 1.0;
const DRIVE_TIMEOUT_S: f32 = 8.0;

/// Controller tick (kept generous — the supervisor / pose threads do
/// the real work between ticks).
const TICK_MS: u64 = 100;

// ── Public types ──────────────────────────────────────────────────────────────

/// One polar scan reading.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ScanRay {
    /// Pan-servo angle for this reading, in degrees. `0` = chassis-left,
    /// `90` = forward, `180` = chassis-right.
    pub pan_deg: f32,
    /// Distance in cm. `None` when the ultrasonic returned no reading
    /// (out of nominal range or sensor error).
    pub distance_cm: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PolarScan {
    pub rays: Vec<ScanRay>,
    /// Index into `rays` of the chosen best heading.
    pub chosen_idx: usize,
    /// Min distance (cm) within the ±20° window of the chosen bin —
    /// the metric the heading was picked on.
    pub chosen_clearance_cm: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum ExplorePhase {
    Idle,
    Scanning,
    Turning { target_theta_rad: f32 },
    Driving { distance_remaining_m: f32 },
    BoxedIn,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExploreStatus {
    pub on: bool,
    pub phase: ExplorePhase,
    /// Latest scan, if one has been taken since startup.
    pub last_scan: Option<PolarScan>,
    /// Total scans completed since the controller started running.
    pub scans_completed: u32,
}

// ── Controller handle ────────────────────────────────────────────────────────

/// The handle stored in `AppState`. The controller worker thread holds
/// clones of the same `Arc`s and updates them in place; this struct
/// is the read-side and the on/off switch.
pub struct ExploreHandle {
    on: Arc<AtomicBool>,
    status: Arc<RwLock<ExploreStatus>>,
}

impl ExploreHandle {
    pub fn new() -> Self {
        Self {
            on: Arc::new(AtomicBool::new(false)),
            status: Arc::new(RwLock::new(ExploreStatus {
                on: false,
                phase: ExplorePhase::Idle,
                last_scan: None,
                scans_completed: 0,
            })),
        }
    }

    pub fn set_on(&self, on: bool) {
        self.on.store(on, Ordering::SeqCst);
        self.status.write().unwrap().on = on;
    }

    pub fn status(&self) -> ExploreStatus {
        self.status.read().unwrap().clone()
    }
}

impl Default for ExploreHandle {
    fn default() -> Self {
        Self::new()
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

struct ExploreWorker {
    sensors: Arc<RwLock<SensorSnapshot>>,
    motor_model: Arc<RwLock<MotorModel>>,
    motion_target: Arc<RwLock<MotionTarget>>,
    pose: Arc<RwLock<PoseState>>,
    cmd_tx: mpsc::Sender<Command>,
    on: Arc<AtomicBool>,
    status: Arc<RwLock<ExploreStatus>>,
}

impl ExploreWorker {
    fn run(&mut self) {
        loop {
            if !self.on.load(Ordering::SeqCst) {
                // Idle: keep the motors stopped so a leftover command
                // doesn't keep the chassis drifting.
                self.send_motion(0.0, 0.0);
                self.set_phase(ExplorePhase::Idle);
                thread::sleep(Duration::from_millis(TICK_MS));
                continue;
            }
            // One pass through the state machine. Each branch is
            // synchronous (and may block for seconds during a scan)
            // — we don't tick at any particular rate while exploring.
            self.set_phase(ExplorePhase::Scanning);
            let scan = self.do_scan();
            self.status.write().unwrap().scans_completed += 1;

            if scan.chosen_clearance_cm < BOXED_IN_CM {
                warn!(
                    "explore: boxed in (clearance {:.0} cm < {:.0}) — stopping",
                    scan.chosen_clearance_cm, BOXED_IN_CM
                );
                self.send_motion(0.0, 0.0);
                self.set_phase(ExplorePhase::BoxedIn);
                self.on.store(false, Ordering::SeqCst);
                self.status.write().unwrap().on = false;
                continue;
            }
            let target_theta = self.target_theta_from_scan(&scan);
            self.status.write().unwrap().last_scan = Some(scan);

            if !self.on.load(Ordering::SeqCst) {
                continue;
            }
            self.do_turn(target_theta);
            if !self.on.load(Ordering::SeqCst) {
                continue;
            }
            self.do_drive();
        }
    }

    fn set_phase(&self, p: ExplorePhase) {
        self.status.write().unwrap().phase = p;
    }

    /// Sweep the pan servo from `PAN_FIRST_DEG` to `PAN_LAST_DEG` in
    /// `PAN_STEP_DEG` increments, sampling ultrasonic at each angle.
    /// Restores pan to `PAN_CENTER_DEG` on the way out.
    fn do_scan(&mut self) -> PolarScan {
        let mut rays: Vec<ScanRay> = Vec::new();
        let mut a = PAN_FIRST_DEG;
        while a <= PAN_LAST_DEG + 0.01 {
            let _ = self.cmd_tx.blocking_send(Command::SetPan(a));
            thread::sleep(Duration::from_millis(SERVO_SETTLE_MS));
            let d = self.sensors.read().unwrap().us_cm;
            rays.push(ScanRay {
                pan_deg: a,
                distance_cm: d,
            });
            a += PAN_STEP_DEG;
        }
        // Park the bracket forward again so manual driving + the next
        // scan starts from a known centre.
        let _ = self.cmd_tx.blocking_send(Command::SetPan(PAN_CENTER_DEG));

        let (chosen_idx, chosen_clearance_cm) = pick_best_heading(&rays);
        PolarScan {
            rays,
            chosen_idx,
            chosen_clearance_cm,
        }
    }

    /// Map the chosen pan angle (servo degrees) to a chassis-frame
    /// yaw target (radians) and add it to the current `pose.theta`.
    /// Pan-90° = chassis-forward. Pan < 90° = chassis-left (CCW = +ω
    /// in our convention).
    fn target_theta_from_scan(&self, scan: &PolarScan) -> f32 {
        let chosen_pan = scan.rays[scan.chosen_idx].pan_deg;
        let yaw_offset = (PAN_CENTER_DEG - chosen_pan).to_radians();
        let theta_now = self.pose.read().unwrap().theta;
        wrap_pi(theta_now + yaw_offset)
    }

    fn do_turn(&mut self, target_theta: f32) {
        let start = std::time::Instant::now();
        loop {
            if !self.on.load(Ordering::SeqCst) {
                self.send_motion(0.0, 0.0);
                return;
            }
            let theta_now = self.pose.read().unwrap().theta;
            let err = wrap_pi(target_theta - theta_now);
            self.set_phase(ExplorePhase::Turning {
                target_theta_rad: target_theta,
            });
            if err.abs() < TURN_TOLERANCE_RAD {
                self.send_motion(0.0, 0.0);
                return;
            }
            if start.elapsed().as_secs_f32() > TURN_TIMEOUT_S {
                warn!(
                    "explore: turn timeout (target {:.2} rad, current {:.2} rad)",
                    target_theta, theta_now
                );
                self.send_motion(0.0, 0.0);
                return;
            }
            let omega = err.signum() * W_TURN_RAD_S;
            self.send_motion(0.0, omega);
            thread::sleep(Duration::from_millis(TICK_MS));
        }
    }

    fn do_drive(&mut self) {
        let start = std::time::Instant::now();
        let (x0, y0) = {
            let p = self.pose.read().unwrap();
            (p.x, p.y)
        };
        loop {
            if !self.on.load(Ordering::SeqCst) {
                self.send_motion(0.0, 0.0);
                return;
            }
            let (x, y) = {
                let p = self.pose.read().unwrap();
                (p.x, p.y)
            };
            let travelled = ((x - x0).powi(2) + (y - y0).powi(2)).sqrt();
            let remaining = (MAX_LEG_M - travelled).max(0.0);
            self.set_phase(ExplorePhase::Driving {
                distance_remaining_m: remaining,
            });
            if remaining <= 0.05 {
                self.send_motion(0.0, 0.0);
                return;
            }
            let us = self.sensors.read().unwrap().us_cm;
            if let Some(d) = us
                && d < STOP_DIST_CM
            {
                info!(
                    "explore: stopping leg — obstacle at {:.0} cm < {:.0}",
                    d, STOP_DIST_CM
                );
                self.send_motion(0.0, 0.0);
                return;
            }
            if start.elapsed().as_secs_f32() > DRIVE_TIMEOUT_S {
                warn!("explore: drive leg timeout");
                self.send_motion(0.0, 0.0);
                return;
            }
            self.send_motion(V_FORWARD_M_S, 0.0);
            thread::sleep(Duration::from_millis(TICK_MS));
        }
    }

    /// Translate desired motion through the motor model and issue the
    /// resulting PWM via the supervisor's command channel. Mirrors
    /// `motion_drive` in `src/server.rs` but takes the cmd channel
    /// directly instead of routing through HTTP.
    fn send_motion(&self, v_target: f32, omega_target: f32) {
        let (pwm_l_prev, pwm_r_prev, v_prev, omega_prev) = {
            let p = self.pose.read().unwrap();
            let t = self.motion_target.read().unwrap();
            (t.pwm_l, t.pwm_r, p.v_us.unwrap_or(p.v_cmd), p.omega_gyro)
        };
        let battery_v = self.sensors.read().unwrap().battery_v;
        let input = ModelInput {
            pwm_l_prev,
            pwm_r_prev,
            v_prev,
            omega_prev,
            battery_v,
            v_target,
            omega_target,
        };
        let pwm = self.motor_model.read().unwrap().predict(input);
        *self.motion_target.write().unwrap() = MotionTarget {
            v_target,
            omega_target,
            pwm_l: pwm.pwm_l,
            pwm_r: pwm.pwm_r,
        };
        let _ = self.cmd_tx.blocking_send(Command::SetMotors {
            left: pwm.pwm_l,
            right: pwm.pwm_r,
        });
    }
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

fn wrap_pi(a: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let pi = std::f32::consts::PI;
    let mut x = a % two_pi;
    if x > pi {
        x -= two_pi;
    } else if x <= -pi {
        x += two_pi;
    }
    x
}

/// For each `rays[i]`, compute the min `distance_cm` in the
/// `±HEADING_WINDOW_HALF_DEG` window centred on `rays[i].pan_deg`.
/// Treat `None` readings as infinitely far (the worst-case missing
/// data is a false-positive "free" — the supervisor's collision stop
/// catches surprises). Pick the bin with the largest such min;
/// tiebreak toward `PAN_CENTER_DEG`.
fn pick_best_heading(rays: &[ScanRay]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_min: f32 = -1.0;
    let mut best_centerness: f32 = f32::INFINITY;
    for (i, r) in rays.iter().enumerate() {
        let mut min_dist = f32::INFINITY;
        for r2 in rays {
            if (r2.pan_deg - r.pan_deg).abs() <= HEADING_WINDOW_HALF_DEG {
                let d = r2.distance_cm.unwrap_or(f32::INFINITY);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        let centerness = (r.pan_deg - PAN_CENTER_DEG).abs();
        if min_dist > best_min || (min_dist == best_min && centerness < best_centerness) {
            best_idx = i;
            best_min = min_dist;
            best_centerness = centerness;
        }
    }
    let chosen_clearance = if best_min.is_finite() { best_min } else { 0.0 };
    (best_idx, chosen_clearance)
}

// ── Spawning ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn spawn(
    sensors: Arc<RwLock<SensorSnapshot>>,
    motor_model: Arc<RwLock<MotorModel>>,
    motion_target: Arc<RwLock<MotionTarget>>,
    pose: Arc<RwLock<PoseState>>,
    cmd_tx: mpsc::Sender<Command>,
    handle: &ExploreHandle,
) {
    let on = handle.on.clone();
    let status = handle.status.clone();
    thread::Builder::new()
        .name("motion-explore".into())
        .spawn(move || {
            let mut worker = ExploreWorker {
                sensors,
                motor_model,
                motion_target,
                pose,
                cmd_tx,
                on,
                status,
            };
            worker.run();
        })
        .map(|_| ())
        .unwrap_or_else(|e| warn!("motion-explore: could not spawn thread: {e}"));
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(pan_deg: f32, distance_cm: f32) -> ScanRay {
        ScanRay {
            pan_deg,
            distance_cm: Some(distance_cm),
        }
    }

    #[test]
    fn pick_best_heading_prefers_clearest_window() {
        // Three sectors: left 50 cm, center 150 cm, right 60 cm.
        // ±20° windows should pick the centre 90° bin.
        let mut rays = vec![];
        for p in (15..=165).step_by(10).map(|i| i as f32) {
            let d = if p < 50.0 {
                50.0
            } else if p > 130.0 {
                60.0
            } else {
                150.0
            };
            rays.push(ray(p, d));
        }
        let (idx, clearance) = pick_best_heading(&rays);
        assert!(
            (rays[idx].pan_deg - 90.0).abs() < 20.0,
            "expected near-centre, got pan={}",
            rays[idx].pan_deg
        );
        assert!(clearance >= 60.0);
    }

    #[test]
    fn pick_best_heading_tiebreaks_toward_centre() {
        // Symmetric: all bins clear at 150 cm — should pick the bin
        // nearest the chassis centre. With PAN_STEP_DEG = 10 and
        // FIRST = 15, the closest sample to 90° is 85° (also 95°,
        // both equidistant); the algorithm keeps the first one it
        // visits with the best centerness.
        let mut rays = vec![];
        for p in (15..=165).step_by(10).map(|i| i as f32) {
            rays.push(ray(p, 150.0));
        }
        let (idx, _) = pick_best_heading(&rays);
        let picked = rays[idx].pan_deg;
        assert!(
            (picked - 85.0).abs() < 0.1 || (picked - 95.0).abs() < 0.1,
            "expected ~85° or ~95° (nearest to centre), got {picked}"
        );
    }

    #[test]
    fn boxed_in_when_no_window_clears() {
        // Everything below 50 cm everywhere — the worker would flag
        // BoxedIn. We test the metric (max-min) here.
        let mut rays = vec![];
        for p in (15..=165).step_by(10).map(|i| i as f32) {
            rays.push(ray(p, 30.0));
        }
        let (_, clearance) = pick_best_heading(&rays);
        assert!(clearance < BOXED_IN_CM);
    }

    #[test]
    fn wrap_pi_keeps_range() {
        for a in [
            0.0,
            1.0,
            std::f32::consts::PI - 0.01,
            -std::f32::consts::PI + 0.01,
        ] {
            let w = wrap_pi(a);
            assert!(
                w > -std::f32::consts::PI && w <= std::f32::consts::PI,
                "{a} → {w} out of range"
            );
        }
        // 3π should wrap to π
        let w = wrap_pi(3.0 * std::f32::consts::PI);
        assert!((w - std::f32::consts::PI).abs() < 1e-4 || (w + std::f32::consts::PI).abs() < 1e-4);
    }
}
