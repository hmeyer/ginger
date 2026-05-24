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
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::camera::Camera;
use crate::imu::{Imu, ImuSample};
use detect::{N_LEVELS, ms};
use ginger_slam_core::lie::so3_exp;
use image::gray_from_yuyv;

/// Camera-frame ΔR predicted from a stream of IMU samples bracketing
/// `(t_prev_capture, t_curr_capture]`, with the learned gyro bias
/// subtracted per sample. Returns `None` if there aren't enough samples
/// in the interval to be meaningful (chip just booted, the polling
/// thread stalled, the camera frame is older than the ring window).
///
/// **Extrinsic.** We default to `R_camera_imu = I` — see PLAN.md Stage 3
/// for the gravity-vector argument (chip is Z-up, gyro_z = yaw, which is
/// the dominant rotational DOF for an indoor robot). If validation shows
/// the predict rotates the wrong axis, override here.
///
/// The integration is forward-Euler in SO(3): for each sample i we apply
/// `ΔR_i = exp([(ω_i - bias) · dt_i]_×)` and compose right-to-left
/// (`R_total = ΔR_n · ... · ΔR_1`), so the result is the rotation that
/// takes the *body frame at t_prev_capture* to the *body frame at
/// t_curr_capture*. `dt_i` is the gap from sample i to the next sample
/// (or to `t_curr_capture` for the last sample). The mean-rate trapezoid
/// would be marginally more accurate over a 33 ms interval, but at the
/// chip's noise floor it doesn't matter and the trapezoid version
/// requires sample i+1, which complicates the boundary.
fn gyro_pre_integrate(
    samples: &[ImuSample],
    bias_dps: [f32; 3],
    t_curr_capture: Instant,
) -> Option<UnitQuaternion<f64>> {
    if samples.len() < 2 {
        return None;
    }
    let to_rad = std::f64::consts::PI / 180.0;
    let mut r = Matrix3::<f64>::identity();
    for i in 0..samples.len() {
        let s = samples[i];
        let next_t = if i + 1 < samples.len() {
            samples[i + 1].t_read
        } else {
            t_curr_capture
        };
        let dt = next_t.saturating_duration_since(s.t_read).as_secs_f64();
        if dt <= 0.0 {
            continue;
        }
        let g = s.raw.gyro_dps();
        let omega = Vector3::new(
            (g[0] - bias_dps[0]) as f64 * to_rad,
            (g[1] - bias_dps[1]) as f64 * to_rad,
            (g[2] - bias_dps[2]) as f64 * to_rad,
        );
        let dr = so3_exp(&(omega * dt));
        r = dr * r;
    }
    Some(UnitQuaternion::from_matrix(&r))
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
        let rotation_hint = match (&imu_for_predict, prev_capture) {
            (Some(im), Some(t_prev)) => {
                let bias = im.gyro_bias_dps();
                let samples = im.recent_since(t_prev);
                gyro_pre_integrate(&samples, bias, frame.t_capture)
            }
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
    use crate::hal::bmi160::RawSample;
    use std::time::Duration;

    /// Build a synthetic stream of N IMU samples spanning `duration`,
    /// each carrying a constant raw gyro `[gx,gy,gz]` in LSB. The
    /// timestamps are evenly spaced; the test consumer treats them
    /// exactly like a real stream from the polling thread.
    fn synth_stream(n: usize, duration: Duration, gyro_raw: [i16; 3]) -> Vec<ImuSample> {
        let t0 = Instant::now();
        let step = duration / (n as u32);
        (0..n)
            .map(|i| ImuSample {
                raw: RawSample {
                    gyro: gyro_raw,
                    accel: [0, 0, 0],
                },
                t_read: t0 + step * (i as u32),
                sensortime: i as u32,
            })
            .collect()
    }

    /// A constant 90 dps stream integrated for 1 s should yield a
    /// rotation of ~90° about the same axis as the gyro vector. Using
    /// 90 LSB → 90 · 500/32768 ≈ 1.37 dps, but stay calibration-clean
    /// by picking 5898 LSB → 90 dps exactly.
    #[test]
    fn integrates_90deg_spin_about_z_to_pi_over_two() {
        // 90 dps × (500/32768) = LSB → 90 / (500/32768) ≈ 5898.24
        let raw = (90.0 / (500.0 / 32768.0)) as i16;
        let dur = Duration::from_millis(1000);
        // Span the samples slightly past the "current" capture so the
        // last `dt` is exact: synth stream ends at t0+dur−step; we pass
        // t_curr_capture = t0+dur so the last gap closes the interval.
        let samples = synth_stream(200, dur, [0, 0, raw]);
        // Pick t_curr_capture deterministically from the synthetic stream.
        let t_curr = samples.last().unwrap().t_read + (dur / 200);
        let dr = gyro_pre_integrate(&samples, [0.0; 3], t_curr)
            .expect("integrator returns Some on a populated stream");
        // Expect rotation about +Z by π/2 rad.
        let aa = dr.axis_angle().expect("non-identity rotation has an axis");
        let (axis, angle) = aa;
        // Axis should be near [0,0,1] (sign-checked).
        assert!(
            (axis.z - 1.0).abs() < 1e-3,
            "expected +Z axis, got {axis:?}"
        );
        // Angle should be ~π/2 with ~1% tolerance (forward-Euler).
        let expected = std::f64::consts::FRAC_PI_2;
        let err = (angle - expected).abs();
        assert!(
            err < expected * 0.01,
            "expected {expected} rad, got {angle} (err {err})"
        );
    }

    /// Bias subtraction: the same constant raw stream integrated with
    /// that same constant as bias should yield identity (no rotation).
    #[test]
    fn bias_subtraction_zeros_a_pure_bias_stream() {
        // Derive `bias` from the LSB stream so quantization doesn't
        // smuggle in a residual (raw stored as i16 truncates).
        let raw_dps = 1.5_f32;
        let raw_lsb = (raw_dps / (500.0 / 32768.0)) as i16;
        let actual_dps = raw_lsb as f32 * (500.0 / 32768.0);
        let dur = Duration::from_millis(1000);
        let samples = synth_stream(200, dur, [raw_lsb, raw_lsb, raw_lsb]);
        let t_curr = samples.last().unwrap().t_read + (dur / 200);
        let dr = gyro_pre_integrate(&samples, [actual_dps; 3], t_curr)
            .expect("Some on a populated stream");
        let angle = dr.angle();
        assert!(
            angle < 1e-6,
            "exactly-cancelled stream should integrate to identity; got angle {angle}"
        );
    }

    /// Sparse stream (< 2 samples) returns None so the frontend falls
    /// back to CV — this is the "first frame after boot" / "polling
    /// stalled" guard the predict policy relies on.
    #[test]
    fn empty_or_singleton_stream_returns_none() {
        let t0 = Instant::now();
        assert!(gyro_pre_integrate(&[], [0.0; 3], t0).is_none());
        let one = ImuSample {
            raw: RawSample {
                gyro: [1000, 0, 0],
                accel: [0, 0, 0],
            },
            t_read: t0,
            sensortime: 0,
        };
        assert!(gyro_pre_integrate(&[one], [0.0; 3], t0 + Duration::from_millis(33)).is_none());
    }
}
