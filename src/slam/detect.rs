//! Image → oriented-BRIEF features: the per-frame detection pipeline
//! (FAST over a scale pyramid, grid-spread cap, then orientation +
//! steered BRIEF on the survivors), the small level-0 wire types it
//! produces ([`FeaturePoint`] / [`Match`]), and the per-stage timing
//! breakdown ([`StageMs`]).

use std::time::{Duration, Instant};

use serde::Serialize;

use super::brief;
use super::fast;
use super::image::{GrayImage, build_pyramid};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Pyramid depth and scale (ORB-SLAM defaults).
pub(crate) const N_LEVELS: usize = 8;
const SCALE_FACTOR: f32 = 1.2;
/// FAST intensity threshold (0–255).
const FAST_THRESHOLD: u8 = 20;
/// Grid cell (level-0 px) and max features per cell — spreads features
/// so they don't all clump on the strongest texture.
const CELL_PX: u16 = 24;
const PER_CELL: usize = 6;
/// Hard cap on features kept/streamed (payload + readability bound).
const MAX_FEATURES: usize = 500;

// ── Shared snapshot (WebUI) ───────────────────────────────────────────────────

/// One detected feature, in **level-0 pixel coordinates**.
#[derive(Clone, Copy, Serialize)]
pub struct FeaturePoint {
    pub x: u16,
    pub y: u16,
    pub level: u8,
    pub score: u16,
    /// Intensity-centroid orientation (radians) used to steer BRIEF.
    pub angle: f32,
}

/// A descriptor-verified correspondence from the previous frame
/// (`x0, y0`) to this frame (`x1, y1`), in level-0 pixel coordinates.
#[derive(Clone, Copy, Serialize)]
pub struct Match {
    pub x0: u16,
    pub y0: u16,
    pub x1: u16,
    pub y1: u16,
}

/// Per-stage wall time for the last frame (ms), EWMA-smoothed. The fields
/// sum to roughly [`super::SlamSnapshot::detect_ms`]; the split shows where
/// the frontend budget goes. `orient`/`describe` run on the *pre-cap* corner
/// set, so they scale with scene texture (`n_total`), not `n_kept`.
#[derive(Clone, Copy, Default, Serialize)]
pub struct StageMs {
    /// YUYV → grayscale luma extraction.
    pub gray: f32,
    /// Scale-pyramid build (bilinear resize × levels).
    pub pyramid: f32,
    /// FAST detect + non-max suppression, summed over levels.
    pub fast: f32,
    /// BRIEF input box blur, summed over levels.
    pub blur: f32,
    /// Intensity-centroid orientation, summed over all pre-cap corners.
    pub orient: f32,
    /// Steered 256-bit BRIEF, summed over all pre-cap corners.
    pub describe: f32,
    /// Brute-force mutual-NN matching against the previous frame.
    pub matching: f32,
}

impl StageMs {
    /// Fold a fresh sample in with the same 0.2/0.8 EWMA as `detect_ms`,
    /// seeding on the first sample.
    pub(crate) fn ewma(&mut self, s: &StageMs) {
        let f = |p: &mut f32, x: f32| *p = if *p == 0.0 { x } else { 0.2 * x + 0.8 * *p };
        f(&mut self.gray, s.gray);
        f(&mut self.pyramid, s.pyramid);
        f(&mut self.fast, s.fast);
        f(&mut self.blur, s.blur);
        f(&mut self.orient, s.orient);
        f(&mut self.describe, s.describe);
        f(&mut self.matching, s.matching);
    }
}

#[inline]
pub(crate) fn ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

// ── Detection pipeline ────────────────────────────────────────────────────────

/// One pre-cap corner: level-0 coordinates (for scoring + grid spread)
/// plus the level and level-local coordinates needed to orient and
/// describe it *after* the cap, so the expensive BRIEF work only runs on
/// the features that survive selection.
struct Cand {
    x: u16,
    y: u16,
    level: u8,
    score: u16,
    lx: i32,
    ly: i32,
}

/// Run FAST over the pyramid, select (strongest-first, grid-spread,
/// capped) keypoints, then orient + describe **only the survivors**, and
/// return them in level-0 coordinates with aligned descriptors plus the
/// pre-cap total.
///
/// Selection uses FAST score only, so deferring orientation/BRIEF until
/// after the cap is output-identical to describing every corner — it just
/// skips the work for the ~90% that the cap discards. Orientation and the
/// descriptor need the patch fully inside the *level* image, so corners
/// within [`brief::BORDER`] of a level edge are dropped during scan.
///
/// The per-corner `Instant` reads below cost ~tens of ns each (vDSO
/// `clock_gettime`); across the kept set that is well under a millisecond
/// and roughly equal between `orient` and `describe`, so the split stays
/// trustworthy for a measurement build.
pub fn detect_features(
    gray: &GrayImage,
) -> (Vec<FeaturePoint>, Vec<brief::Descriptor>, u32, StageMs) {
    let (w, h) = (gray.width, gray.height);
    let mut st = StageMs::default();

    let t = Instant::now();
    let pyramid = build_pyramid(gray.clone(), N_LEVELS, SCALE_FACTOR);
    st.pyramid = ms(t.elapsed());

    let (mut fast_d, mut blur_d) = (Duration::ZERO, Duration::ZERO);

    // Pass 1: detect + blur per level, collect lightweight candidates.
    let mut all: Vec<Cand> = Vec::new();
    let mut blurred: Vec<GrayImage> = Vec::with_capacity(pyramid.len());
    for (lvl, level) in pyramid.iter().enumerate() {
        let (lw, lh) = (level.image.width, level.image.height);
        let t = Instant::now();
        let raw = fast::detect(&level.image, FAST_THRESHOLD);
        let kept = fast::non_max_suppress(&raw, lw, lh);
        fast_d += t.elapsed();
        // BRIEF samples single pixels of a smoothed image (one blur/level).
        let t = Instant::now();
        blurred.push(level.image.box_blur(2));
        blur_d += t.elapsed();
        for c in kept {
            let (lx, ly) = (c.x as i32, c.y as i32);
            if !brief::in_bounds(lw, lh, lx, ly) {
                continue;
            }
            // Map this level's coords back onto level 0.
            let x = ((c.x as f32 * level.scale).round() as usize).min(w.saturating_sub(1));
            let y = ((c.y as f32 * level.scale).round() as usize).min(h.saturating_sub(1));
            all.push(Cand {
                x: x as u16,
                y: y as u16,
                level: lvl as u8,
                score: c.score,
                lx,
                ly,
            });
        }
    }
    st.fast = ms(fast_d);
    st.blur = ms(blur_d);
    let n_total = all.len() as u32;

    // Strongest first, then grid-spread so a cell can't hoard the cap.
    all.sort_unstable_by_key(|c| std::cmp::Reverse(c.score));
    let cells_x = w as u16 / CELL_PX + 1;
    let mut cell_count = vec![0u8; (cells_x as usize) * (h / CELL_PX as usize + 1)];
    let cap = MAX_FEATURES.min(all.len());
    let mut points = Vec::with_capacity(cap);
    let mut descs = Vec::with_capacity(cap);

    // Pass 2: orient + describe only the kept survivors.
    let (mut orient_d, mut desc_d) = (Duration::ZERO, Duration::ZERO);
    for c in all {
        if points.len() >= MAX_FEATURES {
            break;
        }
        let ci = (c.y / CELL_PX) as usize * cells_x as usize + (c.x / CELL_PX) as usize;
        if (cell_count[ci] as usize) < PER_CELL {
            cell_count[ci] += 1;
            let lvl_img = &pyramid[c.level as usize].image;
            let t = Instant::now();
            let angle = brief::orientation(lvl_img, c.lx, c.ly);
            orient_d += t.elapsed();
            let t = Instant::now();
            let desc = brief::describe(&blurred[c.level as usize], c.lx, c.ly, angle);
            desc_d += t.elapsed();
            points.push(FeaturePoint {
                x: c.x,
                y: c.y,
                level: c.level,
                score: c.score,
                angle,
            });
            descs.push(desc);
        }
    }
    st.orient = ms(orient_d);
    st.describe = ms(desc_d);
    (points, descs, n_total, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_caps_on_a_textured_image() {
        // Checkerboard → lots of corners everywhere.
        let (w, h) = (160, 120);
        let mut g = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = if (x / 8 + y / 8) % 2 == 0 { 230 } else { 20 };
            }
        }
        let (pts, descs, n_total, _) = detect_features(&g);
        assert!(n_total > 0);
        assert!(!pts.is_empty());
        assert!(pts.len() <= MAX_FEATURES);
        // Descriptors stay aligned 1:1 with points.
        assert_eq!(pts.len(), descs.len());
        // Strongest-first ordering preserved out of the spread pass.
        for pair in pts.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
        // Coordinates in-bounds; orientation is a real angle.
        assert!(pts.iter().all(|p| (p.x as usize) < w && (p.y as usize) < h));
        assert!(pts.iter().all(|p| p.angle.is_finite()));
    }

    #[test]
    fn blank_image_yields_nothing() {
        let g = GrayImage::new(120, 90);
        let (pts, descs, n_total, _) = detect_features(&g);
        assert_eq!(n_total, 0);
        assert!(pts.is_empty());
        assert!(descs.is_empty());
    }
}
