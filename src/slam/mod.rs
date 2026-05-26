//! Visual SLAM frontend (ORB-SLAM-style).
//!
//! A dedicated thread consumes the camera independently of the
//! H.264/WebRTC path, builds a grayscale pyramid, runs FAST-9 per level
//! (NMS + grid-spread cap), computes intensity-centroid orientation + a
//! steered 256-bit BRIEF descriptor, and feeds the [`Frontend`] state
//! machine:
//! accumulate parallax vs an anchor → two-view init
//! ([`ginger_slam_core::twoview`]) → per-frame tracking
//! (constant-velocity + motion-only BA, [`ginger_slam_core::tracking`])
//! against a keyframe map. Healthy frames become keyframes
//! ([`ginger_slam_core::map`]); a decoupled [`mapper::LocalMapper`]
//! thread triangulates new points ([`ginger_slam_core::triangulation`]),
//! runs block-sparse Schur local BA ([`ginger_slam_core::local_ba`]),
//! and BoW-detects loops ([`place`] / [`ginger_slam_core::bow`]) →
//! `Sim3` verify + Essential-graph pose-graph correction
//! ([`ginger_slam_core::sim3`]). A lost track relocalizes via BoW +
//! PnP-RANSAC ([`ginger_slam_core::pnp`]). Publishes a [`SlamSnapshot`]
//! (live overlay) and a [`MapSnapshot`] (top-down trajectory + keyframes
//! + map, snapping on loop closure).
//!
//! Module layout:
//! - [`detect`] — image → oriented-BRIEF features + per-stage timing.
//! - [`snapshot`] — WebUI-surfaced state + top-down map publishing.
//! - [`frontend`] — the [`Frontend`] state machine ([`Frontend::on_frame`]).
//! - [`run`] (here) — the frontend thread that drives them.

pub mod brief;
pub mod fast;
pub mod image;
pub mod mapper;
pub mod place;

mod detect;
mod frontend;
mod snapshot;

pub use detect::{FeaturePoint, Match, StageMs, detect_features};
pub use frontend::{FrameOut, Frontend};
pub use snapshot::{IntrinsicsView, MapSnapshot, SlamSnapshot};

use std::sync::{Arc, RwLock};
use std::time::Instant;

use log::{info, warn};
use nalgebra::{Quaternion, UnitQuaternion};

use crate::camera::Camera;
use crate::imu::{Imu, ImuSample};
use detect::{N_LEVELS, ms};
use image::gray_from_yuyv;

/// Compute ΔR = `q_curr * q_prev⁻¹` from two BNO055 fusion samples,
/// promoted to `f64` for the frontend's downstream math.
///
/// Returns `None` if the two samples are the same fusion frame (chip
/// hasn't refreshed since the previous camera frame — predict gains
/// nothing). Sign and frame conventions follow the chip: the orientation
/// is in the BNO055 body frame relative to its boot pose; we default to
/// `R_camera_imu = I` (camera and chip Z-axes both ≈ chassis-vertical).
/// If validation shows the predict rotates the wrong axis, apply a
/// fixed `ROT_CHIP_TO_CAMERA` rotation either here or in the driver.
fn delta_rotation(prev: &ImuSample, curr: &ImuSample) -> Option<UnitQuaternion<f64>> {
    if prev.sample_index == curr.sample_index {
        return None;
    }
    let to_f64 = |q: UnitQuaternion<f32>| {
        let q = q.into_inner();
        UnitQuaternion::new_unchecked(Quaternion::new(
            q.w as f64, q.i as f64, q.j as f64, q.k as f64,
        ))
    };
    Some(to_f64(curr.orientation) * to_f64(prev.orientation).inverse())
}

/// Camera-frame ΔR predicted from the BNO055's fusion engine over
/// `(t_prev_capture, t_curr_capture]`. Pulls the chip's best orientation
/// estimate at-or-before each camera capture instant and returns their
/// quaternion delta. Returns `None` if either endpoint has no orientation
/// (chip still warming up, or the camera frame is older than the IMU
/// ring window).
fn rotation_between(
    imu: &Imu,
    t_prev_capture: Instant,
    t_curr_capture: Instant,
) -> Option<UnitQuaternion<f64>> {
    let prev = imu.latest_before(t_prev_capture)?;
    let curr = imu.latest_before(t_curr_capture)?;
    delta_rotation(&prev, &curr)
}

// ── Frontend thread ───────────────────────────────────────────────────────────

/// Own a dedicated thread: pull frames (independently of the video
/// encoder), run the [`Frontend`] pipeline, and publish snapshots.
///
/// `imu` enables the gyro-pre-integrated rotation predict (PLAN.md
/// Stage 4). Pass `None` for vision-only behaviour; the kill-switch
/// `GINGER_IMU_PREDICT=0` env var also forces this path at runtime
/// even when an `Imu` was opened.
pub fn run(
    camera: Arc<Camera>,
    imu: Option<Arc<Imu>>,
    snapshot: Arc<RwLock<SlamSnapshot>>,
    map: Arc<RwLock<MapSnapshot>>,
) {
    let mut fe = Frontend::new();
    // Decouple local mapping (triangulation + heavy Schur local BA)
    // onto its own thread/core per the Pi-4 strategy; tracking
    // here just inserts keyframes over the channel and keeps frame rate.
    if let Some(lm) = fe.take_local_mapper() {
        std::thread::Builder::new()
            .name("slam-mapper".into())
            .spawn(move || lm.run_loop())
            .expect("spawn slam-mapper thread");
    }
    let mut detect_ms = 0.0f32;
    let mut fps = 0.0f32;
    let mut stages = StageMs::default();
    let mut frame_n: u64 = 0;
    let mut last = Instant::now();
    // Kill switch: if the IMU is wired up but you want to A/B test
    // against vision-only (e.g. a session-5 validation run), set
    // `GINGER_IMU_PREDICT=0` in the unit env. Read once at thread start
    // — flipping it at runtime requires a restart, which keeps the
    // session boundary clean for comparison.
    let imu_predict_enabled =
        imu.is_some() && std::env::var("GINGER_IMU_PREDICT").map_or(true, |v| v != "0");
    let imu_for_predict = if imu_predict_enabled {
        imu.as_ref().map(Arc::clone)
    } else {
        if imu.is_some() {
            warn!("slam: GINGER_IMU_PREDICT=0 — IMU present but predict is vision-only");
        }
        None
    };
    let mut prev_capture: Option<Instant> = None;
    info!(
        "slam: frontend started (FAST + oriented BRIEF, {N_LEVELS} levels, \
         imu-predict={})",
        if imu_predict_enabled { "on" } else { "off" }
    );

    loop {
        let frame = camera.wait_frame();
        let t0 = Instant::now();
        let tg = Instant::now();
        let gray = gray_from_yuyv(&frame);
        let gray_ms = ms(tg.elapsed());

        let (points, descs, n_total, mut stage) = detect_features(&gray);
        stage.gray = gray_ms;
        let n_kept = points.len() as u32;

        // IMU pre-integration over `(prev_capture, frame.t_capture]`.
        // First frame has no previous; the integrator also returns None
        // if fewer than two samples landed in the interval (chip
        // starting up, polling stall). On `None` `on_frame` falls back
        // to CV — same code path as `GINGER_IMU_PREDICT=0`.
        // IMU pre-integration replaced by the BNO055's fusion: read
        // the chip's orientation at-or-before each camera capture and
        // take the quaternion delta. No software integration, no bias
        // tracking — the chip's IMUPLUS engine does both internally.
        let rotation_hint = match (&imu_for_predict, prev_capture) {
            (Some(im), Some(t_prev)) => rotation_between(im, t_prev, frame.t_capture),
            _ => None,
        };
        prev_capture = Some(frame.t_capture);

        let out = fe.on_frame(&points, &descs, gray.width, gray.height, rotation_hint);
        stage.matching = out.match_ms;

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
                matches: out.matches,
                intrinsics: out.intrinsics,
            };
        }
        if let Ok(mut m) = map.write() {
            *m = out.map;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use std::time::Duration;

    fn sample(orientation: UnitQuaternion<f32>, index: u32) -> ImuSample {
        ImuSample {
            orientation,
            linear_accel: [0.0; 3],
            t_read: Instant::now() + Duration::from_millis(index as u64 * 10),
            sample_index: index,
        }
    }

    /// Two orientation snapshots — identity and a 90° rotation about
    /// chip-Z — should produce a delta whose yaw component is +π/2.
    /// This is the BNO055 fusion replacement for the old gyro-integration
    /// path: no integration, no bias, just `q_curr * q_prev⁻¹`.
    #[test]
    fn delta_rotation_recovers_90deg_yaw_between_snapshots() {
        let q_prev = UnitQuaternion::identity();
        let q_curr =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_2);
        let dr = delta_rotation(&sample(q_prev, 0), &sample(q_curr, 1))
            .expect("distinct sample indices return Some");
        let (_, _, yaw) = dr.euler_angles();
        assert!(
            (yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "yaw = {yaw} (expected π/2)"
        );
        // Axis should be near +Z.
        let (axis, _) = dr.axis_angle().expect("non-identity rotation has an axis");
        assert!(axis.z > 0.999, "expected +Z axis, got {axis:?}");
    }

    /// Identical orientations from two distinct fusion frames produce
    /// an (approximately) identity ΔR — no rotation between them.
    #[test]
    fn delta_rotation_identity_when_orientations_match() {
        let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);
        let dr = delta_rotation(&sample(q, 0), &sample(q, 1)).unwrap();
        assert!(
            dr.angle() < 1e-6,
            "expected identity, got angle {}",
            dr.angle()
        );
    }

    /// Same fusion-frame index returns None so the frontend falls back
    /// to CV — this is the "chip hasn't refreshed" / polling stall guard.
    #[test]
    fn same_sample_index_returns_none() {
        let q = UnitQuaternion::identity();
        assert!(delta_rotation(&sample(q, 7), &sample(q, 7)).is_none());
    }
}
