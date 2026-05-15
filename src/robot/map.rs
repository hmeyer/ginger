//! 2-D occupancy grid.
//!
//! Coordinate system: x = east (right), y = south (down).
//! Robot heading: 0° = north, clockwise positive.
//! Pan servo: 90° = forward relative to robot, 0° = full-left, 180° = full-right.

use serde::Serialize;

pub mod render;

pub const W: usize = 320; // cells — 32 m at 10 cm/cell
pub const H: usize = 320;
pub const CELL_CM: f32 = 10.0;
pub const MAX_RANGE_CM: f32 = 150.0;

// Cell encoding: 0 = unknown, 1–127 = free (confidence), 128–255 = occupied
pub const UNKNOWN: u8 = 0;

#[derive(Clone)]
pub struct Map {
    pub cells: Vec<u8>,
    pub robot_gx: f32, // grid coords (floats for sub-cell precision)
    pub robot_gy: f32,
    pub robot_heading: f32, // degrees, 0=north, clockwise
}

#[derive(Serialize, Clone)]
pub struct MapMeta {
    pub width: usize,
    pub height: usize,
    pub cell_cm: f32,
    pub robot_gx: f32,
    pub robot_gy: f32,
    pub robot_heading: f32,
}

/// One reading from the ultrasonic sensor during a sweep.
#[derive(Clone, Debug)]
pub struct ScanRay {
    pub pan_deg: f32, // servo angle (90 = forward)
    pub dist_cm: f32, // measured distance, already capped at MAX_RANGE_CM
    pub capped: bool, // true if we hit the cap (no real obstacle detected)
}

impl Default for Map {
    fn default() -> Self {
        Self::new()
    }
}

impl Map {
    pub fn new() -> Self {
        Self {
            cells: vec![UNKNOWN; W * H],
            robot_gx: (W / 2) as f32,
            robot_gy: (H / 2) as f32,
            robot_heading: 0.0,
        }
    }

    pub fn meta(&self) -> MapMeta {
        MapMeta {
            width: W,
            height: H,
            cell_cm: CELL_CM,
            robot_gx: self.robot_gx,
            robot_gy: self.robot_gy,
            robot_heading: self.robot_heading,
        }
    }

    /// Integrate a full sweep of rays into the grid.
    pub fn integrate_scan(&mut self, rays: &[ScanRay]) {
        let rx = self.robot_gx;
        let ry = self.robot_gy;
        let heading = self.robot_heading;
        for ray in rays {
            self.cast_ray(rx, ry, heading, ray);
        }
    }

    fn cast_ray(&mut self, from_gx: f32, from_gy: f32, heading_deg: f32, ray: &ScanRay) {
        // Absolute angle in map coords (sin/cos in image-y-down system)
        let abs_deg = heading_deg + (ray.pan_deg - 90.0);
        let abs_rad = abs_deg.to_radians();
        let dx = abs_rad.sin(); //  east component
        let dy = -abs_rad.cos(); // -north = south component (y-down)

        let cells_along = (ray.dist_cm / CELL_CM).ceil() as usize;

        // Mark all cells along the ray as free
        for i in 0..cells_along {
            let t = i as f32 * CELL_CM / ray.dist_cm;
            let gx = (from_gx + dx * ray.dist_cm / CELL_CM * t).round() as i32;
            let gy = (from_gy + dy * ray.dist_cm / CELL_CM * t).round() as i32;
            if let Some(cell) = self.cell_mut(gx, gy) {
                // Increment free confidence, clamped to 127
                if *cell < 128 {
                    *cell = (*cell).saturating_add(8).min(127);
                }
            }
        }

        // Mark endpoint as occupied only if it's a real obstacle (not capped)
        if !ray.capped {
            let gx = (from_gx + dx * ray.dist_cm / CELL_CM).round() as i32;
            let gy = (from_gy + dy * ray.dist_cm / CELL_CM).round() as i32;
            if let Some(cell) = self.cell_mut(gx, gy) {
                *cell = (*cell).max(128).saturating_add(16);
            }
        }
    }

    fn cell_mut(&mut self, gx: i32, gy: i32) -> Option<&mut u8> {
        if gx < 0 || gy < 0 || gx >= W as i32 || gy >= H as i32 {
            return None;
        }
        Some(&mut self.cells[gy as usize * W + gx as usize])
    }
}
