//! State surfaced to the WebUI and the top-down map publisher: the
//! active-intrinsics view ([`IntrinsicsView`]), the live overlay
//! snapshot ([`SlamSnapshot`]), the bootstrap/tracking map snapshot
//! ([`MapSnapshot`]) and the helpers that build it from the shared
//! [`Map`] (`publish_map`, `centre`, `calib_norm`).

use log::{info, warn};
use serde::Serialize;

use ginger_slam_core::camera::CameraModel;
use ginger_slam_core::intrinsics::Intrinsics;
use ginger_slam_core::map::Map;
use nalgebra::{Isometry3, Vector2};

use super::detect::{FeaturePoint, Match, StageMs};

/// Where an operator-supplied calibration is read from, if present.
/// Absent → the rev 1.3 FOV-derived prior (`verified = false`).
pub(crate) const INTRINSICS_PATH: &str = "slam.toml";

/// Active camera intrinsics, surfaced to the WebUI so the calibration
/// state (and the `UNVERIFIED` rev 1.3 prior) is visible, not buried in
/// logs.
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
    pub(crate) fn of(i: &Intrinsics) -> Self {
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

    pub(crate) fn uninitialized() -> Self {
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
pub(crate) fn resolve_intrinsics(width: u32, height: u32) -> Intrinsics {
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
    /// Strongest-first, capped at `MAX_FEATURES`.
    pub points: Vec<FeaturePoint>,
    /// BRIEF correspondences to the previous frame's features.
    pub matches: Vec<Match>,
    /// Active camera intrinsics + calibration state.
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
    /// Per-frame camera-centre trajectory, top-down `(x, z)`, oldest →
    /// newest (smooth path).
    pub cameras: Vec<[f32; 2]>,
    /// Keyframe camera centres, top-down `(x, z)` (the BA-refined
    /// subset the local map is built on).
    pub keyframes: Vec<[f32; 2]>,
    pub n_points: u32,
    /// Homography-vs-essential selection ratio of the init model.
    pub r_h: f32,
    /// True once tracking is live (post-initialization).
    pub tracking: bool,
    /// Map points matched+inlier in the latest tracked frame.
    pub n_tracked: u32,
    /// Alive keyframe count (debug HUD).
    pub n_keyframes: u32,
    /// Loop closures applied so far (debug HUD).
    pub loop_closures: u64,
    /// BoW vocabulary is trained — relocalization can run (debug HUD).
    pub bow_ready: bool,
    /// BoW vocabulary word count, 0 until trained (debug HUD).
    pub bow_words: u32,
    // ── Two-view bootstrap diagnostics ────────────────────────────────────
    // Populated only while the frontend is in `Stage::Bootstrapping`
    // (when `tracking == false`); zeros once tracking is live. Captured
    // here rather than logged so a debug client can poll `/api/slam/map`
    // at high rate and watch the curves while jogging the car around.
    /// Matches between the current frame and the active anchor (0 if no
    /// anchor yet).
    pub boot_matches: u32,
    /// Median pixel disparity of those matches (0 if no anchor / no
    /// matches yet).
    pub boot_median_disp_px: f32,
    /// Parallax threshold the bootstrap is gated on
    /// (`width * INIT_MIN_DISP_FRAC`).
    pub boot_min_disp_px: f32,
    /// Frames the current anchor has been alive for (resets to 0 each
    /// time the anchor is replaced).
    pub boot_anchor_age: u32,
    /// Cumulative anchor resets since the process started — a quick
    /// signal for "the anchor keeps getting thrown away before
    /// parallax can grow".
    pub boot_anchor_resets: u32,
}

impl MapSnapshot {
    pub fn initial() -> Self {
        Self {
            status: "waiting for an anchor frame".into(),
            model: String::new(),
            points: Vec::new(),
            cameras: Vec::new(),
            keyframes: Vec::new(),
            n_points: 0,
            r_h: 0.0,
            tracking: false,
            n_tracked: 0,
            n_keyframes: 0,
            loop_closures: 0,
            bow_ready: false,
            bow_words: 0,
            boot_matches: 0,
            boot_median_disp_px: 0.0,
            boot_min_disp_px: 0.0,
            boot_anchor_age: 0,
            boot_anchor_resets: 0,
        }
    }
}

/// Trajectory payload cap (memory + JSON bound); keep the newest.
pub(crate) const MAX_TRAJECTORY: usize = 800;

/// Camera centre of a `T_cw` pose in world coords: `-Rᵀt`, top-down.
fn centre(t: &Isometry3<f64>) -> [f32; 2] {
    let c = t.inverse().translation.vector;
    [c.x as f32, c.z as f32]
}

/// Build a [`MapSnapshot`] from the shared map + the per-frame
/// trajectory: every alive map point, every alive keyframe centre, and
/// the (capped) per-frame camera path.
pub(crate) fn publish_map(
    world: &Map,
    traj: &[Isometry3<f64>],
    model: &str,
    r_h: f32,
    status: String,
    n_tracked: u32,
) -> MapSnapshot {
    let points: Vec<[f32; 2]> = world
        .alive_points()
        .filter(|p| p.pos.x.is_finite() && p.pos.z.is_finite())
        .map(|p| [p.pos.x as f32, p.pos.z as f32])
        .collect();
    let keyframes: Vec<[f32; 2]> = world.alive_keyframes().map(|k| centre(&k.pose)).collect();
    let n = traj.len();
    let start = n.saturating_sub(MAX_TRAJECTORY);
    let cameras: Vec<[f32; 2]> = traj[start..].iter().map(centre).collect();
    let n_keyframes = keyframes.len() as u32;
    MapSnapshot {
        status,
        model: model.into(),
        n_points: points.len() as u32,
        points,
        cameras,
        keyframes,
        r_h,
        tracking: true,
        n_tracked,
        n_keyframes,
        // Owned by `Frontend::on_frame` — refreshed every frame, incl.
        // while Lost/Bootstrapping when this builder is not called.
        loop_closures: 0,
        bow_ready: false,
        bow_words: 0,
        // Bootstrap diagnostics are only meaningful pre-init; zero here.
        boot_matches: 0,
        boot_median_disp_px: 0.0,
        boot_min_disp_px: 0.0,
        boot_anchor_age: 0,
        boot_anchor_resets: 0,
    }
}

/// Pixel → calibrated/normalized image point (undistort, then `K⁻¹`).
pub(crate) fn calib_norm(cam: &CameraModel, x: u16, y: u16) -> Vector2<f64> {
    let ud = cam.undistort_point(&Vector2::new(x as f64, y as f64));
    Vector2::new((ud.x - cam.cx) / cam.fx, (ud.y - cam.cy) / cam.fy)
}
