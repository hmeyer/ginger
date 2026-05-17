//! Visual SLAM frontend (ORB-SLAM-style), built up in milestones.
//!
//! **M0 (current): FAST features.** A dedicated thread consumes the
//! camera independently of the H.264/WebRTC path, builds a grayscale
//! scale pyramid, runs FAST-9 per level with non-maximal suppression and
//! a grid-spread cap, and publishes a compact [`SlamSnapshot`] for the
//! WebUI to draw live. Later milestones add the ORB descriptor, two-view
//! init, local-map tracking, local BA and loop closing.

pub mod fast;
pub mod image;

use std::sync::{Arc, RwLock};
use std::time::Instant;

use log::info;
use serde::Serialize;

use crate::camera::Camera;
use crate::marker;
use image::{GrayImage, build_pyramid};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Pyramid depth and scale (ORB-SLAM defaults).
const N_LEVELS: usize = 8;
const SCALE_FACTOR: f32 = 1.2;
/// FAST intensity threshold (0–255).
const FAST_THRESHOLD: u8 = 20;
/// Extra guard band (px) around the marker ROI to mask from detection.
const ROI_GUARD: usize = 2;
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
}

/// Latest frontend result, polled by `GET /api/slam/stream`.
#[derive(Clone, Serialize)]
pub struct SlamSnapshot {
    pub width: u16,
    pub height: u16,
    /// Camera frame id these features were detected on (matches the
    /// luma marker burned into the video — the client uses it to draw
    /// features over the exact frame they came from).
    pub frame_id: u64,
    /// Total corners after NMS, before the grid-spread cap.
    pub n_total: u32,
    /// Frontend time for the last frame (ms), EWMA-smoothed.
    pub detect_ms: f32,
    /// Frontend throughput (frames/s), EWMA-smoothed.
    pub fps: f32,
    /// Strongest-first, capped at [`MAX_FEATURES`].
    pub points: Vec<FeaturePoint>,
}

impl SlamSnapshot {
    pub fn initial() -> Self {
        Self {
            width: 0,
            height: 0,
            frame_id: 0,
            n_total: 0,
            detect_ms: 0.0,
            fps: 0.0,
            points: Vec::new(),
        }
    }
}

// ── Detection pipeline ────────────────────────────────────────────────────────

/// Run FAST over the pyramid and return (strongest-first, grid-spread,
/// capped) features in level-0 coordinates, plus the pre-cap NMS total.
pub fn detect_features(gray: &GrayImage) -> (Vec<FeaturePoint>, u32) {
    let (w, h) = (gray.width, gray.height);
    // Mask the bottom-right marker ROI (+ guard) so the frame-id code
    // never produces features. ROI is flush to the corner, so a point
    // is excluded iff it's past the (guard-expanded) top-left of it.
    let roi_origin = if w >= marker::ROI_W && h >= marker::ROI_H {
        let (rx, ry, _, _) = marker::roi_rect(w, h);
        Some((rx.saturating_sub(ROI_GUARD), ry.saturating_sub(ROI_GUARD)))
    } else {
        None
    };
    let in_roi = |x: usize, y: usize| roi_origin.is_some_and(|(ex, ey)| x >= ex && y >= ey);

    let pyramid = build_pyramid(gray.clone(), N_LEVELS, SCALE_FACTOR);

    let mut all: Vec<FeaturePoint> = Vec::new();
    for (lvl, level) in pyramid.iter().enumerate() {
        let raw = fast::detect(&level.image, FAST_THRESHOLD);
        let kept = fast::non_max_suppress(&raw, level.image.width, level.image.height);
        for c in kept {
            // Map this level's coords back onto level 0.
            let x = ((c.x as f32 * level.scale).round() as usize).min(w.saturating_sub(1));
            let y = ((c.y as f32 * level.scale).round() as usize).min(h.saturating_sub(1));
            if in_roi(x, y) {
                continue;
            }
            all.push(FeaturePoint {
                x: x as u16,
                y: y as u16,
                level: lvl as u8,
                score: c.score,
            });
        }
    }
    let n_total = all.len() as u32;

    // Strongest first, then grid-spread so a cell can't hoard the cap.
    all.sort_unstable_by_key(|p| std::cmp::Reverse(p.score));
    let cells_x = w as u16 / CELL_PX + 1;
    let mut cell_count = vec![0u8; (cells_x as usize) * (h / CELL_PX as usize + 1)];
    let mut out = Vec::with_capacity(MAX_FEATURES.min(all.len()));
    for p in all {
        if out.len() >= MAX_FEATURES {
            break;
        }
        let ci = (p.y / CELL_PX) as usize * cells_x as usize + (p.x / CELL_PX) as usize;
        if (cell_count[ci] as usize) < PER_CELL {
            cell_count[ci] += 1;
            out.push(p);
        }
    }
    (out, n_total)
}

/// Detect features on `gray` and assemble a snapshot tagged with
/// `frame_id` (propagated verbatim — never re-stamped). Timing fields
/// are left at 0 for the caller to fill with smoothed values.
pub fn build_snapshot(gray: &GrayImage, frame_id: u64) -> SlamSnapshot {
    let (points, n_total) = detect_features(gray);
    SlamSnapshot {
        width: gray.width as u16,
        height: gray.height as u16,
        frame_id,
        n_total,
        detect_ms: 0.0,
        fps: 0.0,
        points,
    }
}

// ── Frontend thread ───────────────────────────────────────────────────────────

/// Own a dedicated thread: pull frames (independently of the video
/// encoder), detect features, and publish into `snapshot`.
pub fn run(camera: Arc<Camera>, snapshot: Arc<RwLock<SlamSnapshot>>) {
    let mut detect_ms = 0.0f32;
    let mut fps = 0.0f32;
    let mut last = Instant::now();
    info!("slam: frontend started (FAST, {N_LEVELS} levels)");

    loop {
        let frame = camera.wait_frame();
        let t0 = Instant::now();
        let gray = GrayImage::from_yuyv(&frame);
        let mut snap = build_snapshot(&gray, frame.id);
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

        snap.detect_ms = detect_ms;
        snap.fps = fps;
        if let Ok(mut s) = snapshot.write() {
            *s = snap;
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
        let (pts, n_total) = detect_features(&g);
        assert!(n_total > 0);
        assert!(!pts.is_empty());
        assert!(pts.len() <= MAX_FEATURES);
        // Strongest-first ordering preserved out of the spread pass.
        for pair in pts.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
        // Coordinates stay in-bounds.
        assert!(pts.iter().all(|p| (p.x as usize) < w && (p.y as usize) < h));
    }

    #[test]
    fn blank_image_yields_nothing() {
        let g = GrayImage::new(120, 90);
        let (pts, n_total) = detect_features(&g);
        assert_eq!(n_total, 0);
        assert!(pts.is_empty());
    }

    #[test]
    fn frame_id_propagates_verbatim() {
        let g = GrayImage::new(160, 120);
        for fid in [0u64, 1, 255, 4242, u64::MAX] {
            assert_eq!(build_snapshot(&g, fid).frame_id, fid);
        }
    }

    #[test]
    fn no_features_inside_marker_roi() {
        // Full-frame checkerboard so corners would otherwise land in
        // the bottom-right marker ROI too.
        let (w, h) = (640usize, 480usize);
        let mut g = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                g.data[y * w + x] = if (x / 6 + y / 6) % 2 == 0 { 235 } else { 16 };
            }
        }
        let (rx, ry, _, _) = marker::roi_rect(w, h);
        let (ex, ey) = (rx - ROI_GUARD, ry - ROI_GUARD);
        let snap = build_snapshot(&g, 7);
        assert!(!snap.points.is_empty());
        assert!(
            snap.points
                .iter()
                .all(|p| (p.x as usize) < ex || (p.y as usize) < ey),
            "a feature landed inside the masked marker ROI"
        );
    }
}
