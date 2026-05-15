//! Autonomous frontier-based exploration.
//!
//! Runs entirely on the hardware thread (which owns Car).
//! An AtomicBool stop-flag lets HTTP handlers abort between steps.
//!
//! Coordinate system: x = east (right), y = south (down in image).
//! Robot heading: 0° = north, clockwise positive.
//! Pan servo: 90° = forward, 0° = full-left, 180° = full-right.

use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use log::{info, warn};
use serde::Serialize;

use crate::robot::{
    car::Car,
    map::{CELL_CM, H, MAX_RANGE_CM, Map, ScanRay, W},
};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const DRIVE_DUTY: i32 = 1500; // ~37% power — reliable start, safe speed
pub const STEP_MS: u64 = 400; // one forward step duration
pub const STEP_CM: f32 = 12.0; // estimated cm per step (calibrate if needed)
pub const TURN_MS: u64 = 280; // one turn pulse duration
pub const TURN_DEG: f32 = 18.0; // estimated degrees per turn pulse
pub const SAFE_CM: f32 = 40.0; // min forward clearance before stepping
pub const ALIGN_DEG: f32 = 25.0; // heading tolerance before stepping
pub const LOW_BAT_V: f32 = 6.0; // stop exploring below this voltage (2S LiPo hard cutoff)

// Minimum squared distance (cells²) a frontier must be from the robot.
// Prevents picking the robot's own cell (which is always marked free and
// has unscanned neighbours behind it).
const MIN_FRONTIER_DIST2: i64 = 9; // 3 cells = 30 cm

pub const SCAN_ANGLES: &[f32] = &[30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0];
const SETTLE_MS: u64 = 300; // servo settle time between scan steps

// ── Status ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Idle,
    Scanning,
    Turning,
    Moving,
    Stuck,
    Complete,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Idle => "idle",
            Self::Scanning => "scanning",
            Self::Turning => "turning",
            Self::Moving => "moving",
            Self::Stuck => "stuck",
            Self::Complete => "complete",
        };
        f.write_str(s)
    }
}

// ── Scan ──────────────────────────────────────────────────────────────────────

/// Sweep pan servo across SCAN_ANGLES and collect ultrasonic readings.
/// Returns to centre when done. Aborts early if `stop` is set.
pub fn do_scan(car: &mut Car, stop: &AtomicBool) -> Vec<ScanRay> {
    info!("scan: sweeping {} angles", SCAN_ANGLES.len());
    let mut rays = Vec::with_capacity(SCAN_ANGLES.len());
    for &pan in SCAN_ANGLES {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        car.pan_tilt().set_pan(pan).ok();
        thread::sleep(Duration::from_millis(SETTLE_MS));
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let raw = car.us().distance_cm().unwrap_or(MAX_RANGE_CM + 1.0);
        let capped = raw >= MAX_RANGE_CM;
        let dist = raw.min(MAX_RANGE_CM);
        info!(
            "  pan={:.0}°  dist={:.1} cm{}",
            pan,
            dist,
            if capped { " (capped)" } else { "" }
        );
        rays.push(ScanRay {
            pan_deg: pan,
            dist_cm: dist,
            capped,
        });
    }
    car.pan_tilt().set_pan(90.0).ok();
    rays
}

// ── Safety helpers ────────────────────────────────────────────────────────────

/// True if all rays within ±30° of forward are beyond SAFE_CM.
pub fn is_forward_safe(rays: &[ScanRay]) -> bool {
    rays.iter()
        .filter(|r| (r.pan_deg - 90.0).abs() <= 30.0)
        .all(|r| r.dist_cm >= SAFE_CM)
}

/// True if the nearest obstacle in the forward arc is to the left of centre.
fn obstacle_is_left(rays: &[ScanRay]) -> bool {
    let left = rays
        .iter()
        .filter(|r| r.pan_deg < 90.0)
        .map(|r| r.dist_cm)
        .fold(f32::MAX, f32::min);
    let right = rays
        .iter()
        .filter(|r| r.pan_deg > 90.0)
        .map(|r| r.dist_cm)
        .fold(f32::MAX, f32::min);
    left < right
}

// ── Map helpers ───────────────────────────────────────────────────────────────

/// Nearest frontier cell: a free cell (1–127) adjacent to an unknown cell (0).
pub fn find_nearest_frontier(map: &Map) -> Option<(usize, usize)> {
    let rx = map.robot_gx as i64;
    let ry = map.robot_gy as i64;
    let mut best: Option<(usize, usize, i64)> = None;

    for y in 1..(H - 1) {
        for x in 1..(W - 1) {
            let cell = map.cells[y * W + x];
            if cell == 0 || cell > 127 {
                continue;
            }

            let has_unknown = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .iter()
                .any(|&(nx, ny)| map.cells[ny * W + nx] == 0);
            if !has_unknown {
                continue;
            }

            let dist2 = (x as i64 - rx).pow(2) + (y as i64 - ry).pow(2);
            if dist2 < MIN_FRONTIER_DIST2 {
                continue;
            } // skip robot's own neighbourhood
            if best.is_none_or(|(_, _, d)| dist2 < d) {
                best = Some((x, y, dist2));
            }
        }
    }
    best.map(|(x, y, _)| (x, y))
}

/// Desired heading (°, 0=north CW) from robot position to grid cell (fx, fy).
pub fn heading_to_cell(map: &Map, fx: usize, fy: usize) -> f32 {
    let dx = (fx as f32) - map.robot_gx;
    let dy = -((fy as f32) - map.robot_gy); // flip: north = +y in map = -dy image
    dx.atan2(dy).to_degrees()
}

/// Smallest signed rotation from `from` to `to` (°, CW positive).
pub fn angle_diff(from: f32, to: f32) -> f32 {
    let mut d = (to - from).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    d
}

// ── Motion primitives ─────────────────────────────────────────────────────────

fn step_forward(car: &mut Car, map: &Arc<RwLock<Map>>, stop: &AtomicBool) {
    info!("move: forward duty={} for {}ms", DRIVE_DUTY, STEP_MS);
    if let Err(e) = car.motors().drive(DRIVE_DUTY, DRIVE_DUTY) {
        warn!("move: drive() error: {e}");
    }
    let deadline = Instant::now() + Duration::from_millis(STEP_MS);
    while Instant::now() < deadline {
        if stop.load(Ordering::Relaxed) {
            info!("move: stop flag — cutting step short");
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    if let Err(e) = car.stop() {
        warn!("move: stop() error: {e}");
    }
    if stop.load(Ordering::Relaxed) {
        return;
    }

    let mut m = map.write().unwrap();
    let h_rad = m.robot_heading.to_radians();
    m.robot_gx += h_rad.sin() * STEP_CM / CELL_CM;
    m.robot_gy += -h_rad.cos() * STEP_CM / CELL_CM;
    m.robot_gx = m.robot_gx.clamp(0.0, (W - 1) as f32);
    m.robot_gy = m.robot_gy.clamp(0.0, (H - 1) as f32);
    info!(
        "move: pose now ({:.1}, {:.1}) heading={:.1}°",
        m.robot_gx, m.robot_gy, m.robot_heading
    );
}

fn turn_pulse(car: &mut Car, map: &Arc<RwLock<Map>>, clockwise: bool, stop: &AtomicBool) {
    let dir = if clockwise { "CW" } else { "CCW" };
    info!("turn: {dir} duty={} for {}ms", DRIVE_DUTY, TURN_MS);
    let (l, r) = if clockwise {
        (DRIVE_DUTY, -DRIVE_DUTY)
    } else {
        (-DRIVE_DUTY, DRIVE_DUTY)
    };
    if let Err(e) = car.motors().drive(l, r) {
        warn!("turn: drive() error: {e}");
    }
    let deadline = Instant::now() + Duration::from_millis(TURN_MS);
    while Instant::now() < deadline {
        if stop.load(Ordering::Relaxed) {
            info!("turn: stop flag — cutting turn short");
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    if let Err(e) = car.stop() {
        warn!("turn: stop() error: {e}");
    }
    if stop.load(Ordering::Relaxed) {
        return;
    }

    let delta = if clockwise { TURN_DEG } else { -TURN_DEG };
    let mut m = map.write().unwrap();
    m.robot_heading = (m.robot_heading + delta).rem_euclid(360.0);
    info!("turn: heading now {:.1}°", m.robot_heading);
}

// ── Main exploration tick ─────────────────────────────────────────────────────

/// Execute one exploration step: scan → decide → act.
/// `stop` is checked between every sub-operation.
pub fn tick(car: &mut Car, map: &Arc<RwLock<Map>>, stop: &AtomicBool) -> Status {
    if stop.load(Ordering::Relaxed) {
        return Status::Idle;
    }

    // Battery check
    match car.battery_v() {
        Ok(v) if v < LOW_BAT_V => {
            warn!("explore: low battery {v:.2} V — stopping");
            return Status::Complete;
        }
        Ok(v) => info!("explore: battery {v:.2} V"),
        Err(e) => warn!("explore: battery read error: {e}"),
    }

    // 1. Scan
    let rays = do_scan(car, stop);
    if stop.load(Ordering::Relaxed) || rays.is_empty() {
        return Status::Idle;
    }
    map.write().unwrap().integrate_scan(&rays);

    // 2. Find nearest frontier
    let (fx, fy) = {
        let m = map.read().unwrap();
        match find_nearest_frontier(&m) {
            None => {
                info!("explore: no frontiers — exploration complete");
                return Status::Complete;
            }
            Some(f) => f,
        }
    };

    // 3. Compute heading error toward frontier
    let (diff, current_heading, target_heading) = {
        let m = map.read().unwrap();
        let target = heading_to_cell(&m, fx, fy);
        let diff = angle_diff(m.robot_heading, target);
        (diff, m.robot_heading, target)
    };
    info!(
        "explore: frontier at cell ({fx},{fy}), heading={current_heading:.1}°, target={target_heading:.1}°, diff={diff:.1}°"
    );

    // 4. Turn to align if needed
    if diff.abs() > ALIGN_DEG {
        if stop.load(Ordering::Relaxed) {
            return Status::Idle;
        }
        info!(
            "explore: turning {} by ~{TURN_DEG}°",
            if diff > 0.0 { "CW" } else { "CCW" }
        );
        turn_pulse(car, map, diff > 0.0, stop);
        return Status::Turning;
    }

    // 5. Step forward if safe, otherwise turn away from obstacle
    let safe = is_forward_safe(&rays);
    info!("explore: forward safe={safe}");
    if safe {
        if stop.load(Ordering::Relaxed) {
            return Status::Idle;
        }
        step_forward(car, map, stop);
        Status::Moving
    } else {
        let clockwise = !obstacle_is_left(&rays);
        warn!(
            "explore: blocked — turning {} to escape",
            if clockwise { "CW" } else { "CCW" }
        );
        for _ in 0..2 {
            if stop.load(Ordering::Relaxed) {
                return Status::Idle;
            }
            turn_pulse(car, map, clockwise, stop);
        }
        Status::Stuck
    }
}
