//! FAST-9 corner detector (Rosten & Drummond) with SAD scoring and
//! non-maximal suppression.
//!
//! Detection is the dominant cost of the SLAM frontend, so the per-row
//! scan is parallelized with `rayon` across the Pi 4's four cores. The
//! inner test is plain scalar today; it is isolated in [`corner_score`]
//! so an aarch64 NEON fast-path can drop in without touching callers.

use rayon::prelude::*;

use super::image::GrayImage;

/// A detected corner in some image's pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Corner {
    pub x: u16,
    pub y: u16,
    pub score: u16,
}

/// Bresenham circle of radius 3 (16 px), clockwise from the top.
const CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];
const BORDER: usize = 3; // circle radius
const ARC: usize = 9; // FAST-9: ≥9 contiguous ring pixels

/// If `(x, y)` is a FAST-9 corner at threshold `t`, return its score
/// (sum of |ring − centre| over the 16 ring pixels), else `None`.
#[inline]
fn corner_score(img: &GrayImage, x: usize, y: usize, t: i32) -> Option<u16> {
    let w = img.width;
    let ip = img.data[y * w + x] as i32;
    let hi = ip + t;
    let lo = ip - t;

    // Cheap reject on the 4 compass pixels (idx 0,4,8,12): any 9-arc
    // covers ≥2 of them, so <2 bright and <2 dark ⇒ not a corner.
    let mut cb = 0;
    let mut cd = 0;
    for k in [0usize, 4, 8, 12] {
        let (dx, dy) = CIRCLE[k];
        let p = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32;
        if p >= hi {
            cb += 1;
        } else if p <= lo {
            cd += 1;
        }
    }
    if cb < 2 && cd < 2 {
        return None;
    }

    // Classify the full ring and find the longest circular run.
    let mut cls = [0i8; 16];
    for (i, &(dx, dy)) in CIRCLE.iter().enumerate() {
        let p = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32;
        cls[i] = if p >= hi {
            1
        } else if p <= lo {
            -1
        } else {
            0
        };
    }
    if !has_arc(&cls, 1) && !has_arc(&cls, -1) {
        return None;
    }

    // Score: SAD of ring vs centre (monotone-ish; good for NMS ranking).
    let mut s = 0i32;
    for &(dx, dy) in &CIRCLE {
        let p = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32;
        s += (p - ip).abs();
    }
    Some(s.min(u16::MAX as i32) as u16)
}

/// Is there a circular run of ≥[`ARC`] pixels all equal to `want`?
#[inline]
fn has_arc(cls: &[i8; 16], want: i8) -> bool {
    let mut run = 0usize;
    // Scan 16+ARC-1 to cover wrap-around runs.
    for i in 0..16 + ARC - 1 {
        if cls[i % 16] == want {
            run += 1;
            if run >= ARC {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Detect FAST-9 corners over the whole image at threshold `t`.
/// Rows are scanned in parallel.
pub fn detect(img: &GrayImage, t: u8) -> Vec<Corner> {
    let (w, h) = (img.width, img.height);
    if w <= 2 * BORDER || h <= 2 * BORDER {
        return Vec::new();
    }
    let t = t as i32;
    (BORDER..h - BORDER)
        .into_par_iter()
        .flat_map_iter(move |y| {
            (BORDER..w - BORDER).filter_map(move |x| {
                corner_score(img, x, y, t).map(|score| Corner {
                    x: x as u16,
                    y: y as u16,
                    score,
                })
            })
        })
        .collect()
}

/// 3×3 non-maximal suppression: keep a corner only if its score is
/// strictly greater than every 8-neighbour corner's score.
pub fn non_max_suppress(corners: &[Corner], w: usize, h: usize) -> Vec<Corner> {
    if corners.is_empty() {
        return Vec::new();
    }
    let mut score = vec![0u16; w * h];
    for c in corners {
        score[c.y as usize * w + c.x as usize] = c.score;
    }
    corners
        .iter()
        .copied()
        .filter(|c| {
            let (cx, cy, s) = (c.x as usize, c.y as usize, c.score);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    if score[ny as usize * w + nx as usize] > s {
                        return false;
                    }
                }
            }
            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Image with a bright filled disc on a dark field — its rim is a
    /// ring of corners; the flat interior/background are not.
    fn disc(w: usize, h: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        let (cx, cy, r2) = (
            w as f32 / 2.0,
            h as f32 / 2.0,
            (w.min(h) as f32 / 4.0).powi(2),
        );
        for y in 0..h {
            for x in 0..w {
                let d = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                img.data[y * w + x] = if d < r2 { 240 } else { 15 };
            }
        }
        img
    }

    #[test]
    fn flat_image_has_no_corners() {
        let mut img = GrayImage::new(40, 40);
        img.data.iter_mut().for_each(|p| *p = 100);
        assert!(detect(&img, 20).is_empty());
    }

    #[test]
    fn detects_corners_on_a_disc_rim() {
        let img = disc(60, 60);
        let corners = detect(&img, 30);
        assert!(
            corners.len() > 8,
            "expected many rim corners, got {}",
            corners.len()
        );
        // All near the rim radius (~15 px from centre), none at centre.
        let (cx, cy) = (30.0f32, 30.0f32);
        for c in &corners {
            let d = ((c.x as f32 - cx).powi(2) + (c.y as f32 - cy).powi(2)).sqrt();
            assert!(d > 6.0, "corner too close to flat centre: {c:?}");
        }
    }

    #[test]
    fn nms_thins_clusters() {
        let img = disc(60, 60);
        let raw = detect(&img, 30);
        let kept = non_max_suppress(&raw, img.width, img.height);
        assert!(!kept.is_empty());
        assert!(kept.len() <= raw.len());
    }

    #[test]
    fn higher_threshold_detects_fewer() {
        let img = disc(80, 80);
        let lo = detect(&img, 20).len();
        let hi = detect(&img, 80).len();
        assert!(hi <= lo, "stricter threshold should not add corners");
    }
}
