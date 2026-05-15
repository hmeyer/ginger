//! Presentation for [`Map`]: PNG and ASCII renderings.
//!
//! Kept separate from the occupancy-grid model so the domain type does not
//! depend on the `image` crate.

use image::{DynamicImage, ImageBuffer, Rgb};

use super::{H, Map, UNKNOWN, W};

impl Map {
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
