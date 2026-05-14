//! 2-D occupancy grid.
//!
//! Coordinate system: x = east (right), y = south (down).
//! Robot heading: 0° = north, clockwise positive.
//! Pan servo: 90° = forward relative to robot, 0° = full-left, 180° = full-right.

use image::{DynamicImage, ImageBuffer, Rgb};
use serde::Serialize;

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

    /// Render occupancy grid as a PNG image (1 px per cell).
    pub fn render_png(&self) -> Vec<u8> {
        let mut img = ImageBuffer::<Rgb<u8>, _>::new(W as u32, H as u32);

        for y in 0..H {
            for x in 0..W {
                let cell = self.cells[y * W + x];
                let rgb = match cell {
                    0 => [14u8, 14, 28], // unknown — very dark
                    1..=127 => {
                        let t = cell as f32 / 127.0;
                        [
                            (30.0 + t * 10.0) as u8,
                            (55.0 + t * 80.0) as u8,
                            (25.0 + t * 10.0) as u8,
                        ] // dark→bright green
                    }
                    _ => [180u8, 48, 38], // occupied — red
                };
                img.put_pixel(x as u32, y as u32, Rgb(rgb));
            }
        }

        // Robot body: 3-cell radius circle in blue
        let rx = self.robot_gx as i32;
        let ry = self.robot_gy as i32;
        for dy in -3i32..=3 {
            for dx in -3i32..=3 {
                if dx * dx + dy * dy <= 9 {
                    let px = (rx + dx).clamp(0, W as i32 - 1) as u32;
                    let py = (ry + dy).clamp(0, H as i32 - 1) as u32;
                    img.put_pixel(px, py, Rgb([68, 170, 255]));
                }
            }
        }
        // Heading indicator: white dot 5 cells ahead
        let hr = self.robot_heading.to_radians();
        let nx = (rx + (hr.sin() * 5.0) as i32).clamp(0, W as i32 - 1) as u32;
        let ny = (ry + (-hr.cos() * 5.0) as i32).clamp(0, H as i32 - 1) as u32;
        img.put_pixel(nx, ny, Rgb([255, 255, 255]));

        let mut buf = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    /// ASCII export for LLM consumption. Shows explored bounding box + robot.
    pub fn to_ascii(&self) -> String {
        // Find bounding box of non-unknown cells
        let mut min_x = W;
        let mut max_x = 0;
        let mut min_y = H;
        let mut max_y = 0;
        for y in 0..H {
            for x in 0..W {
                if self.cells[y * W + x] != UNKNOWN {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        if max_x < min_x {
            return "(empty map)".to_string();
        }
        // Pad by 2
        let x0 = min_x.saturating_sub(2);
        let y0 = min_y.saturating_sub(2);
        let x1 = (max_x + 2).min(W - 1);
        let y1 = (max_y + 2).min(H - 1);

        let rx = self.robot_gx.round() as usize;
        let ry = self.robot_gy.round() as usize;

        let mut out = String::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let ch = if x == rx && y == ry {
                    '@'
                } else {
                    match self.cells[y * W + x] {
                        0 => '?',
                        1..=127 => '.',
                        _ => '#',
                    }
                };
                out.push(ch);
            }
            out.push('\n');
        }
        out
    }
}
