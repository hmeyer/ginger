//! Visual SLAM frontend (ORB-SLAM-style), built up in milestones.
//!
//! **M3 (current): two-view monocular initialization.** A dedicated
//! thread consumes the camera independently of the H.264/WebRTC path,
//! builds a grayscale scale pyramid, runs FAST-9 per level (NMS +
//! grid-spread cap), computes intensity-centroid orientation + a steered
//! 256-bit BRIEF descriptor, and matches frames. Against an anchor
//! frame it accumulates parallax and, once well-conditioned, runs the
//! calibrated [`ginger_slam_core::twoview`] initializer (essential /
//! homography → relative pose + triangulated points), publishing a
//! [`SlamSnapshot`] (live overlay) and a [`MapSnapshot`] (top-down map).
//! Later milestones add local-map tracking (M4), local BA + keyframes
//! (M5) and loop closing (M6).

pub mod brief;
pub mod fast;
pub mod image;

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use ginger_slam_core::intrinsics::Intrinsics;
use log::{info, warn};
use serde::Serialize;

use ginger_slam_core::camera::CameraModel;
use ginger_slam_core::tracking::{self, Observation};
use ginger_slam_core::twoview::{self, InitOptions};
use nalgebra::{Isometry3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};

use crate::camera::Camera;
use image::{GrayImage, build_pyramid, gray_from_yuyv};

// Two-view initialization gates: enough matches against the anchor
// frame, and enough median pixel parallax (fraction of image width) so
// the geometry is well-conditioned. Below the floor the anchor is stale
// (scene changed) and is reset to the current frame.
const INIT_MIN_MATCHES: usize = 80;
const INIT_MIN_DISP_FRAC: f32 = 0.04;
const ANCHOR_RESET_MATCHES: usize = 25;

// Tracking gates: min map-point matches to attempt a pose solve, and
// min reprojection inliers for the refined pose to be trusted.
const TRACK_MIN_MATCHES: usize = 15;
const TRACK_MIN_INLIERS: usize = 10;

/// Where an operator-supplied calibration is read from, if present.
/// Absent → the rev 1.3 FOV-derived prior (`verified = false`).
const INTRINSICS_PATH: &str = "slam.toml";

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

/// Per-stage wall time for the last frame (ms), EWMA-smoothed. The fields
/// sum to roughly [`SlamSnapshot::detect_ms`]; the split shows where the
/// frontend budget goes. `orient`/`describe` run on the *pre-cap* corner
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
    fn ewma(&mut self, s: &StageMs) {
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
fn ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

/// Active camera intrinsics, surfaced to the WebUI so the calibration
/// state (and the `UNVERIFIED` rev 1.3 prior) is visible, not buried in
/// logs. This is M2's only user-facing output — M2 ships no SLAM yet.
#[derive(Clone, Serialize)]
pub struct IntrinsicsView {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// Horizontal FOV (deg) implied by `fx`/width.
    pub fov_deg: f32,
    /// `false` for the derived prior; `true` only after a real
    /// calibration. The HUD renders this as an `UNVERIFIED` badge.
    pub verified: bool,
    pub model: String,
}

impl IntrinsicsView {
    fn of(i: &Intrinsics) -> Self {
        Self {
            fx: i.fx as f32,
            fy: i.fy as f32,
            cx: i.cx as f32,
            cy: i.cy as f32,
            fov_deg: i.hfov_deg() as f32,
            verified: i.verified,
            model: i.model.clone(),
        }
    }

    fn uninitialized() -> Self {
        Self {
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            fov_deg: 0.0,
            verified: false,
            model: "(awaiting first frame)".into(),
        }
    }
}

/// Resolve intrinsics for a given stream resolution: an operator
/// `slam.toml` if present and parseable, else the rev 1.3 FOV prior.
fn resolve_intrinsics(width: u32, height: u32) -> Intrinsics {
    match std::fs::read_to_string(INTRINSICS_PATH) {
        Ok(s) => match Intrinsics::from_toml_str(&s) {
            Ok(i) => {
                info!(
                    "slam: loaded intrinsics from {INTRINSICS_PATH} \
                     (model={:?} verified={})",
                    i.model, i.verified
                );
                i
            }
            Err(e) => {
                warn!(
                    "slam: {INTRINSICS_PATH} unparseable ({e}); \
                     falling back to rev 1.3 prior"
                );
                Intrinsics::rev1_3_prior(width, height)
            }
        },
        Err(_) => Intrinsics::rev1_3_prior(width, height),
    }
}

/// Latest frontend result, polled by `GET /api/slam/stream`.
#[derive(Clone, Serialize)]
pub struct SlamSnapshot {
    pub width: u16,
    pub height: u16,
    /// Total corners after NMS, before the grid-spread cap.
    pub n_total: u32,
    /// Corners kept after the grid-spread cap (`points.len()`). The
    /// `n_total - n_kept` gap is orientation/describe work that was
    /// computed and then discarded.
    pub n_kept: u32,
    /// Frontend time for the last frame (ms), EWMA-smoothed.
    pub detect_ms: f32,
    /// Per-stage breakdown of `detect_ms` (ms, EWMA).
    pub stages: StageMs,
    /// Frontend throughput (frames/s), EWMA-smoothed.
    pub fps: f32,
    /// Strongest-first, capped at [`MAX_FEATURES`].
    pub points: Vec<FeaturePoint>,
    /// BRIEF correspondences to the previous frame's features.
    pub matches: Vec<Match>,
    /// Active camera intrinsics + calibration state (M2 surface).
    pub intrinsics: IntrinsicsView,
}

impl SlamSnapshot {
    pub fn initial() -> Self {
        Self {
            width: 0,
            height: 0,
            n_total: 0,
            n_kept: 0,
            detect_ms: 0.0,
            stages: StageMs::default(),
            fps: 0.0,
            points: Vec::new(),
            matches: Vec::new(),
            intrinsics: IntrinsicsView::uninitialized(),
        }
    }
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

// ── Initial map (two-view) ────────────────────────────────────────────────────

/// The monocular bootstrap result, polled by `GET /api/slam/map` and
/// drawn as a top-down (x, z) plot. Scale is arbitrary (monocular), so
/// the WebUI auto-fits the bounding box.
#[derive(Clone, Serialize)]
pub struct MapSnapshot {
    /// Human-readable init state for the WebUI.
    pub status: String,
    /// `"essential"`, `"homography"`, or empty before init.
    pub model: String,
    /// Triangulated map points, top-down `(x, z)`, world (view-1) frame.
    pub points: Vec<[f32; 2]>,
    /// Camera-centre trajectory, top-down `(x, z)`, oldest → newest.
    pub cameras: Vec<[f32; 2]>,
    pub n_points: u32,
    /// Homography-vs-essential selection ratio of the init model.
    pub r_h: f32,
    /// True once tracking is live (post-initialization).
    pub tracking: bool,
    /// Map points matched+inlier in the latest tracked frame.
    pub n_tracked: u32,
}

impl MapSnapshot {
    pub fn initial() -> Self {
        Self {
            status: "waiting for an anchor frame".into(),
            model: String::new(),
            points: Vec::new(),
            cameras: Vec::new(),
            n_points: 0,
            r_h: 0.0,
            tracking: false,
            n_tracked: 0,
        }
    }
}

/// Live map + pose history once two-view init has succeeded. Points are
/// in the world (view-1) frame, paired with the init-frame descriptor
/// used to re-find them; `poses` are `T_cw` (world → camera), oldest
/// first (`poses[0]` = identity = view-1).
struct MapState {
    pts: Vec<Vector3<f64>>,
    desc: Vec<brief::Descriptor>,
    poses: Vec<Isometry3<f64>>,
    model: String,
    r_h: f32,
}

/// Trajectory payload cap (memory + JSON bound); keep the newest.
const MAX_TRAJECTORY: usize = 800;

impl MapState {
    /// Camera centre of a `T_cw` pose in world coords: `-Rᵀt`.
    fn centre(t: &Isometry3<f64>) -> [f32; 2] {
        let c = t.inverse().translation.vector;
        [c.x as f32, c.z as f32]
    }

    fn publish(&self, map: &Arc<RwLock<MapSnapshot>>, status: String, n_tracked: u32) {
        let points: Vec<[f32; 2]> = self
            .pts
            .iter()
            .filter(|p| p.x.is_finite() && p.z.is_finite())
            .map(|p| [p.x as f32, p.z as f32])
            .collect();
        let n = self.poses.len();
        let start = n.saturating_sub(MAX_TRAJECTORY);
        let cameras: Vec<[f32; 2]> = self.poses[start..].iter().map(Self::centre).collect();
        if let Ok(mut m) = map.write() {
            *m = MapSnapshot {
                status,
                model: self.model.clone(),
                n_points: points.len() as u32,
                points,
                cameras,
                r_h: self.r_h,
                tracking: true,
                n_tracked,
            };
        }
    }
}

/// Pixel → calibrated/normalized image point (undistort, then `K⁻¹`).
fn calib_norm(cam: &CameraModel, x: u16, y: u16) -> Vector2<f64> {
    let ud = cam.undistort_point(&Vector2::new(x as f64, y as f64));
    Vector2::new((ud.x - cam.cx) / cam.fx, (ud.y - cam.cy) / cam.fy)
}

// ── Frontend thread ───────────────────────────────────────────────────────────

/// Own a dedicated thread: pull frames (independently of the video
/// encoder), detect features, and publish into `snapshot`.
pub fn run(
    camera: Arc<Camera>,
    snapshot: Arc<RwLock<SlamSnapshot>>,
    map: Arc<RwLock<MapSnapshot>>,
) {
    let mut detect_ms = 0.0f32;
    let mut fps = 0.0f32;
    let mut stages = StageMs::default();
    let mut frame_n: u64 = 0;
    let mut last = Instant::now();
    let mut prev: Option<(Vec<FeaturePoint>, Vec<brief::Descriptor>)> = None;
    // Two-view bootstrap: an anchor frame to accumulate parallax
    // against, until `state` holds a map; then M4 tracking takes over.
    let mut anchor: Option<(Vec<FeaturePoint>, Vec<brief::Descriptor>)> = None;
    let mut state: Option<MapState> = None;
    // Resolved once from the first frame's resolution (the libcamera
    // ViewFinder mode is full-FOV, so the FOV-derived prior is
    // resolution-agnostic — no need to query libcamera properties).
    let mut intrinsics: Option<Intrinsics> = None;
    let mut iview = IntrinsicsView::uninitialized();
    info!("slam: frontend started (FAST + oriented BRIEF, {N_LEVELS} levels)");

    loop {
        let frame = camera.wait_frame();
        let t0 = Instant::now();
        let tg = Instant::now();
        let gray = gray_from_yuyv(&frame);
        let gray_ms = ms(tg.elapsed());

        if intrinsics.is_none() {
            let i = resolve_intrinsics(gray.width as u32, gray.height as u32);
            if i.verified {
                info!(
                    "slam: camera intrinsics verified (model={:?}, fov≈{:.1}°)",
                    i.model,
                    i.hfov_deg()
                );
            } else {
                warn!(
                    "slam: camera intrinsics are an UNVERIFIED prior \
                     (model={:?}, fov≈{:.1}°) — run a calibration and write {INTRINSICS_PATH}",
                    i.model,
                    i.hfov_deg()
                );
            }
            iview = IntrinsicsView::of(&i);
            intrinsics = Some(i);
        }

        let (points, descs, n_total, mut stage) = detect_features(&gray);
        stage.gray = gray_ms;
        let n_kept = points.len() as u32;

        // Brute-force match against the previous frame's descriptors.
        let tm = Instant::now();
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
        stage.matching = ms(tm.elapsed());

        // Pre-init: accumulate parallax vs an anchor, run two-view init.
        // Post-init: project the map into each frame and refine the pose
        // (constant-velocity prediction + motion-only BA).
        if let Some(intr) = intrinsics.as_ref() {
            let cam = intr.to_camera_model();
            match state.as_mut() {
                None => {
                    let mut want_anchor = false;
                    if let Some((apts, adesc)) = anchor.as_ref() {
                        let mm = brief::match_descriptors(adesc, &descs);
                        if mm.len() >= INIT_MIN_MATCHES {
                            let mut disp: Vec<f32> = mm
                                .iter()
                                .map(|&(ia, ib)| {
                                    let a = apts[ia as usize];
                                    let b = points[ib as usize];
                                    let dx = a.x as f32 - b.x as f32;
                                    let dy = a.y as f32 - b.y as f32;
                                    (dx * dx + dy * dy).sqrt()
                                })
                                .collect();
                            disp.sort_by(|x, y| x.partial_cmp(y).unwrap());
                            let med = disp[disp.len() / 2];
                            let min_disp = gray.width as f32 * INIT_MIN_DISP_FRAC;
                            if med >= min_disp {
                                let corrs: Vec<twoview::Corr> = mm
                                    .iter()
                                    .map(|&(ia, ib)| {
                                        let a = apts[ia as usize];
                                        let b = points[ib as usize];
                                        (calib_norm(&cam, a.x, a.y), calib_norm(&cam, b.x, b.y))
                                    })
                                    .collect();
                                if let Some(tv) =
                                    twoview::initialize(&corrs, InitOptions::default())
                                {
                                    // cam1 = world origin; cam2 = Tcw = [R|t].
                                    let pose2 = Isometry3::from_parts(
                                        Translation3::from(tv.t),
                                        UnitQuaternion::from_rotation_matrix(
                                            &Rotation3::from_matrix_unchecked(tv.r),
                                        ),
                                    );
                                    // Align triangulated points with the
                                    // init-frame descriptor that re-finds them.
                                    let (mut pts, mut desc) = (Vec::new(), Vec::new());
                                    let mut pi = 0usize;
                                    for (k, &(_ia, ib)) in mm.iter().enumerate() {
                                        if tv.inliers[k] {
                                            if let Some(&x) = tv.points.get(pi)
                                                && x.z > 0.0
                                                && x.x.is_finite()
                                                && x.z.is_finite()
                                            {
                                                pts.push(x);
                                                desc.push(descs[ib as usize]);
                                            }
                                            pi += 1;
                                        }
                                    }
                                    let model = match tv.model {
                                        twoview::Model::Essential => "essential",
                                        twoview::Model::Homography => "homography",
                                    };
                                    let n = pts.len();
                                    let ms = MapState {
                                        pts,
                                        desc,
                                        poses: vec![Isometry3::identity(), pose2],
                                        model: model.into(),
                                        r_h: tv.r_h as f32,
                                    };
                                    ms.publish(
                                        &map,
                                        format!("initialized: {n} pts via {model}"),
                                        n as u32,
                                    );
                                    info!(
                                        "slam: two-view init OK — {n} pts, model={model}, \
                                         R_H={:.2}, matches={}",
                                        tv.r_h,
                                        mm.len()
                                    );
                                    state = Some(ms);
                                }
                            } else if let Ok(mut m) = map.write() {
                                m.status = format!(
                                    "parallax {med:.0}/{min_disp:.0}px · {} matches",
                                    mm.len()
                                );
                            }
                        } else if mm.len() < ANCHOR_RESET_MATCHES {
                            want_anchor = true;
                        }
                    } else {
                        want_anchor = true;
                    }
                    if want_anchor && state.is_none() {
                        anchor = Some((points.clone(), descs.clone()));
                        if let Ok(mut m) = map.write() {
                            m.status = "anchor set — translate sideways for parallax".into();
                        }
                    }
                }
                Some(st) => {
                    let mm = brief::match_descriptors(&st.desc, &descs);
                    if mm.len() >= TRACK_MIN_MATCHES {
                        let obs: Vec<Observation> = mm
                            .iter()
                            .map(|&(im, ic)| {
                                let kp = points[ic as usize];
                                Observation {
                                    point: st.pts[im as usize],
                                    obs: calib_norm(&cam, kp.x, kp.y),
                                }
                            })
                            .collect();
                        let np = st.poses.len();
                        let predict = if np >= 2 {
                            tracking::constant_velocity(&st.poses[np - 2], &st.poses[np - 1])
                        } else {
                            st.poses[np - 1]
                        };
                        let huber = 2.0 / intr.fx;
                        let thr = 5.0 / intr.fx;
                        match tracking::track_pose(&obs, &predict, huber, thr) {
                            Some(rep) if rep.converged && rep.n_inliers >= TRACK_MIN_INLIERS => {
                                st.poses.push(rep.pose);
                                if st.poses.len() > MAX_TRAJECTORY * 2 {
                                    let drop = st.poses.len() - MAX_TRAJECTORY;
                                    st.poses.drain(0..drop);
                                }
                                let status =
                                    format!("tracking: {}/{} inliers", rep.n_inliers, obs.len());
                                st.publish(&map, status, rep.n_inliers as u32);
                            }
                            Some(rep) => {
                                if let Ok(mut m) = map.write() {
                                    m.status = format!(
                                        "tracking weak: {}/{} inliers",
                                        rep.n_inliers,
                                        obs.len()
                                    );
                                }
                            }
                            None => {
                                if let Ok(mut m) = map.write() {
                                    m.status = "tracking lost (solve failed)".into();
                                }
                            }
                        }
                    } else if let Ok(mut m) = map.write() {
                        m.status = format!("tracking lost: only {} map matches", mm.len());
                    }
                }
            }
        }

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
        stages.ewma(&stage);

        // Throttled so the journal stays readable (~3 s at 10 fps); the
        // WebUI HUD has the live per-frame view.
        frame_n += 1;
        if frame_n.is_multiple_of(32) {
            info!(
                "slam stages(ms): gray {:.1} pyr {:.1} fast {:.1} blur {:.1} \
                 orient {:.1} desc {:.1} match {:.1} | total {:.0} | \
                 corners {n_kept}/{n_total} | {fps:.1} fps",
                stages.gray,
                stages.pyramid,
                stages.fast,
                stages.blur,
                stages.orient,
                stages.describe,
                stages.matching,
                detect_ms,
            );
        }

        if let Ok(mut s) = snapshot.write() {
            *s = SlamSnapshot {
                width: gray.width as u16,
                height: gray.height as u16,
                n_total,
                n_kept,
                detect_ms,
                stages,
                fps,
                points,
                matches,
                intrinsics: iview.clone(),
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
