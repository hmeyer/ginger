//! Visual SLAM frontend (ORB-SLAM-style), built up in milestones.
//!
//! **M1 (current): FAST + oriented BRIEF.** A dedicated thread consumes
//! the camera independently of the H.264/WebRTC path, builds a grayscale
//! scale pyramid, runs FAST-9 per level with non-maximal suppression and
//! a grid-spread cap, computes an intensity-centroid orientation and a
//! steered 256-bit BRIEF descriptor per keypoint, brute-force matches
//! against the previous frame, and publishes a compact [`SlamSnapshot`]
//! (points + match lines) for the WebUI to draw live. Later milestones
//! add two-view init, local-map tracking, local BA and loop closing.

pub mod brief;
pub mod fast;
pub mod image;

use std::sync::{Arc, RwLock};
use std::time::Instant;

use log::info;
use serde::Serialize;

use crate::camera::Camera;
use image::{GrayImage, build_pyramid, gray_from_yuyv};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Pyramid depth and scale (ORB-SLAM defaults).
const N_LEVELS: usize = 8;
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

/// Latest frontend result, polled by `GET /api/slam/stream`.
#[derive(Clone, Serialize)]
pub struct SlamSnapshot {
    pub width: u16,
    pub height: u16,
    /// Total corners after NMS, before the grid-spread cap.
    pub n_total: u32,
    /// Frontend time for the last frame (ms), EWMA-smoothed.
    pub detect_ms: f32,
    /// Frontend throughput (frames/s), EWMA-smoothed.
    pub fps: f32,
    /// Strongest-first, capped at [`MAX_FEATURES`].
    pub points: Vec<FeaturePoint>,
    /// BRIEF correspondences to the previous frame's features.
    pub matches: Vec<Match>,
}

impl SlamSnapshot {
    pub fn initial() -> Self {
        Self {
            width: 0,
            height: 0,
            n_total: 0,
            detect_ms: 0.0,
            fps: 0.0,
            points: Vec::new(),
            matches: Vec::new(),
        }
    }
}

// ── Detection pipeline ────────────────────────────────────────────────────────

/// Run FAST over the pyramid, describe each keypoint with oriented
/// BRIEF, and return (strongest-first, grid-spread, capped) features in
/// level-0 coordinates with aligned descriptors, plus the pre-cap total.
///
/// Orientation and the descriptor need the patch fully inside the *level*
/// image, so corners within [`brief::BORDER`] of a level edge are
/// dropped here (before the level coords are discarded).
pub fn detect_features(gray: &GrayImage) -> (Vec<FeaturePoint>, Vec<brief::Descriptor>, u32) {
    let (w, h) = (gray.width, gray.height);
    let pyramid = build_pyramid(gray.clone(), N_LEVELS, SCALE_FACTOR);

    let mut all: Vec<(FeaturePoint, brief::Descriptor)> = Vec::new();
    for (lvl, level) in pyramid.iter().enumerate() {
        let (lw, lh) = (level.image.width, level.image.height);
        let raw = fast::detect(&level.image, FAST_THRESHOLD);
        let kept = fast::non_max_suppress(&raw, lw, lh);
        // BRIEF samples single pixels of a smoothed image (one blur/level).
        let blurred = level.image.box_blur(2);
        for c in kept {
            let (lx, ly) = (c.x as i32, c.y as i32);
            if !brief::in_bounds(lw, lh, lx, ly) {
                continue;
            }
            let angle = brief::orientation(&level.image, lx, ly);
            let desc = brief::describe(&blurred, lx, ly, angle);
            // Map this level's coords back onto level 0.
            let x = ((c.x as f32 * level.scale).round() as usize).min(w.saturating_sub(1));
            let y = ((c.y as f32 * level.scale).round() as usize).min(h.saturating_sub(1));
            all.push((
                FeaturePoint {
                    x: x as u16,
                    y: y as u16,
                    level: lvl as u8,
                    score: c.score,
                    angle,
                },
                desc,
            ));
        }
    }
    let n_total = all.len() as u32;

    // Strongest first, then grid-spread so a cell can't hoard the cap.
    all.sort_unstable_by_key(|p| std::cmp::Reverse(p.0.score));
    let cells_x = w as u16 / CELL_PX + 1;
    let mut cell_count = vec![0u8; (cells_x as usize) * (h / CELL_PX as usize + 1)];
    let cap = MAX_FEATURES.min(all.len());
    let mut points = Vec::with_capacity(cap);
    let mut descs = Vec::with_capacity(cap);
    for (p, d) in all {
        if points.len() >= MAX_FEATURES {
            break;
        }
        let ci = (p.y / CELL_PX) as usize * cells_x as usize + (p.x / CELL_PX) as usize;
        if (cell_count[ci] as usize) < PER_CELL {
            cell_count[ci] += 1;
            points.push(p);
            descs.push(d);
        }
    }
    (points, descs, n_total)
}

// ── Frontend thread ───────────────────────────────────────────────────────────

/// Own a dedicated thread: pull frames (independently of the video
/// encoder), detect features, and publish into `snapshot`.
pub fn run(camera: Arc<Camera>, snapshot: Arc<RwLock<SlamSnapshot>>) {
    let mut detect_ms = 0.0f32;
    let mut fps = 0.0f32;
    let mut last = Instant::now();
    let mut prev: Option<(Vec<FeaturePoint>, Vec<brief::Descriptor>)> = None;
    info!("slam: frontend started (FAST + oriented BRIEF, {N_LEVELS} levels)");

    loop {
        let frame = camera.wait_frame();
        let t0 = Instant::now();
        let gray = gray_from_yuyv(&frame);
        let (points, descs, n_total) = detect_features(&gray);

        // Brute-force match against the previous frame's descriptors.
        let matches = match &prev {
            Some((pp, pd)) => brief::match_descriptors(pd, &descs)
                .into_iter()
                .map(|(i, j)| {
                    let a = pp[i as usize];
                    let b = points[j as usize];
                    Match {
                        x0: a.x,
                        y0: a.y,
                        x1: b.x,
                        y1: b.y,
                    }
                })
                .collect(),
            None => Vec::new(),
        };
        prev = Some((points.clone(), descs));

        let elapsed = t0.elapsed().as_secs_f32() * 1000.0;

        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f32();
        last = now;
        // EWMA so the WebUI readout is stable.
        detect_ms = if detect_ms == 0.0 {
            elapsed
        } else {
            0.2 * elapsed + 0.8 * detect_ms
        };
        if dt > 0.0 {
            let inst = 1.0 / dt;
            fps = if fps == 0.0 {
                inst
            } else {
                0.2 * inst + 0.8 * fps
            };
        }

        if let Ok(mut s) = snapshot.write() {
            *s = SlamSnapshot {
                width: gray.width as u16,
                height: gray.height as u16,
                n_total,
                detect_ms,
                fps,
                points,
                matches,
            };
        }
    }
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
        let (pts, descs, n_total) = detect_features(&g);
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
        let (pts, descs, n_total) = detect_features(&g);
        assert_eq!(n_total, 0);
        assert!(pts.is_empty());
        assert!(descs.is_empty());
    }
}
