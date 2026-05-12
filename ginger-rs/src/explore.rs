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
    time::Duration,
};

use serde::Serialize;

use crate::{
    car::Car,
    map::{Map, ScanRay, W, H, CELL_CM, MAX_RANGE_CM},
};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const DRIVE_DUTY: i32 = 800;   // ~40% power — slow and safe
pub const STEP_MS:    u64 = 400;   // one forward step duration
pub const STEP_CM:    f32 = 12.0;  // estimated cm per step (calibrate if needed)
pub const TURN_MS:    u64 = 280;   // one turn pulse duration
pub const TURN_DEG:   f32 = 18.0;  // estimated degrees per turn pulse
pub const SAFE_CM:    f32 = 40.0;  // min forward clearance before stepping
pub const ALIGN_DEG:  f32 = 25.0;  // heading tolerance before stepping
pub const LOW_BAT_V:  f32 = 6.5;   // stop exploring below this voltage

pub const SCAN_ANGLES: &[f32] = &[30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0];
const SETTLE_MS: u64 = 300;        // servo settle time between scan steps

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
            Self::Idle     => "idle",
            Self::Scanning => "scanning",
            Self::Turning  => "turning",
            Self::Moving   => "moving",
            Self::Stuck    => "stuck",
            Self::Complete => "complete",
        };
        f.write_str(s)
    }
}

// ── Scan ──────────────────────────────────────────────────────────────────────

/// Sweep pan servo across SCAN_ANGLES and collect ultrasonic readings.
/// Returns to centre when done. Aborts early if `stop` is set.
pub fn do_scan(car: &mut Car, stop: &AtomicBool) -> Vec<ScanRay> {
    let mut rays = Vec::with_capacity(SCAN_ANGLES.len());
    for &pan in SCAN_ANGLES {
        if stop.load(Ordering::Relaxed) { break; }
        car.pan_tilt().set_pan(pan).ok();
        thread::sleep(Duration::from_millis(SETTLE_MS));
        if stop.load(Ordering::Relaxed) { break; }
        let raw    = car.us().distance_cm().unwrap_or(MAX_RANGE_CM + 1.0);
        let capped = raw >= MAX_RANGE_CM;
        rays.push(ScanRay { pan_deg: pan, dist_cm: raw.min(MAX_RANGE_CM), capped });
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
    let left  = rays.iter().filter(|r| r.pan_deg < 90.0).map(|r| r.dist_cm).fold(f32::MAX, f32::min);
    let right = rays.iter().filter(|r| r.pan_deg > 90.0).map(|r| r.dist_cm).fold(f32::MAX, f32::min);
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
            if cell == 0 || cell > 127 { continue; }

            let has_unknown = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .iter()
                .any(|&(nx, ny)| map.cells[ny * W + nx] == 0);
            if !has_unknown { continue; }

            let dist2 = (x as i64 - rx).pow(2) + (y as i64 - ry).pow(2);
            if best.map_or(true, |(_, _, d)| dist2 < d) {
                best = Some((x, y, dist2));
            }
        }
    }
    best.map(|(x, y, _)| (x, y))
}

/// Desired heading (°, 0=north CW) from robot position to grid cell (fx, fy).
pub fn heading_to_cell(map: &Map, fx: usize, fy: usize) -> f32 {
    let dx =  (fx as f32) - map.robot_gx;
    let dy = -((fy as f32) - map.robot_gy); // flip: north = +y in map = -dy image
    dx.atan2(dy).to_degrees()
}

/// Smallest signed rotation from `from` to `to` (°, CW positive).
pub fn angle_diff(from: f32, to: f32) -> f32 {
    let mut d = (to - from).rem_euclid(360.0);
    if d > 180.0 { d -= 360.0; }
    d
}

// ── Motion primitives ─────────────────────────────────────────────────────────

fn step_forward(car: &mut Car, map: &Arc<RwLock<Map>>) {
    car.motors().drive(DRIVE_DUTY, DRIVE_DUTY).ok();
    thread::sleep(Duration::from_millis(STEP_MS));
    car.stop().ok();

    let mut m = map.write().unwrap();
    let h_rad = m.robot_heading.to_radians();
    m.robot_gx +=  h_rad.sin() * STEP_CM / CELL_CM;
    m.robot_gy += -h_rad.cos() * STEP_CM / CELL_CM; // y-down: north = -y
    m.robot_gx = m.robot_gx.clamp(0.0, (W - 1) as f32);
    m.robot_gy = m.robot_gy.clamp(0.0, (H - 1) as f32);
}

fn turn_pulse(car: &mut Car, map: &Arc<RwLock<Map>>, clockwise: bool) {
    let (l, r) = if clockwise {
        (DRIVE_DUTY, -DRIVE_DUTY)
    } else {
        (-DRIVE_DUTY, DRIVE_DUTY)
    };
    car.motors().drive(l, r).ok();
    thread::sleep(Duration::from_millis(TURN_MS));
    car.stop().ok();

    let delta = if clockwise { TURN_DEG } else { -TURN_DEG };
    let mut m = map.write().unwrap();
    m.robot_heading = (m.robot_heading + delta).rem_euclid(360.0);
}

// ── Main exploration tick ─────────────────────────────────────────────────────

/// Execute one exploration step: scan → decide → act.
/// `stop` is checked between every sub-operation.
pub fn tick(car: &mut Car, map: &Arc<RwLock<Map>>, stop: &AtomicBool) -> Status {
    if stop.load(Ordering::Relaxed) { return Status::Idle; }

    // Battery check
    if let Ok(v) = car.battery_v() {
        if v < LOW_BAT_V { return Status::Complete; }
    }

    // 1. Scan
    let rays = do_scan(car, stop);
    if stop.load(Ordering::Relaxed) || rays.is_empty() { return Status::Idle; }
    map.write().unwrap().integrate_scan(&rays);

    // 2. Find nearest frontier
    let (fx, fy) = {
        let m = map.read().unwrap();
        match find_nearest_frontier(&m) {
            None    => return Status::Complete,
            Some(f) => f,
        }
    };

    // 3. Compute heading error toward frontier
    let diff = {
        let m      = map.read().unwrap();
        let target = heading_to_cell(&m, fx, fy);
        angle_diff(m.robot_heading, target)
    };

    // 4. Turn to align if needed
    if diff.abs() > ALIGN_DEG {
        if stop.load(Ordering::Relaxed) { return Status::Idle; }
        turn_pulse(car, map, diff > 0.0);
        return Status::Turning;
    }

    // 5. Step forward if safe, otherwise turn away from obstacle
    if is_forward_safe(&rays) {
        if stop.load(Ordering::Relaxed) { return Status::Idle; }
        step_forward(car, map);
        Status::Moving
    } else {
        let clockwise = !obstacle_is_left(&rays); // turn away from obstacle
        for _ in 0..2 {
            if stop.load(Ordering::Relaxed) { return Status::Idle; }
            turn_pulse(car, map, clockwise);
        }
        Status::Stuck
    }
}
