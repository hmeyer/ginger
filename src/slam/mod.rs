//! Visual SLAM frontend (ORB-SLAM-style), built up in milestones.
//!
//! **M6 (current): relocalization + loop closing.** A dedicated thread
//! consumes the camera independently of the H.264/WebRTC path, builds a
//! grayscale pyramid, runs FAST-9 per level (NMS + grid-spread cap),
//! computes intensity-centroid orientation + a steered 256-bit BRIEF
//! descriptor, and feeds the [`Frontend`] state machine:
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

use log::info;

use crate::camera::Camera;
use detect::{N_LEVELS, ms};
use image::gray_from_yuyv;

// ── Frontend thread ───────────────────────────────────────────────────────────

/// Own a dedicated thread: pull frames (independently of the video
/// encoder), run the [`Frontend`] pipeline, and publish snapshots.
pub fn run(
    camera: Arc<Camera>,
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
    info!("slam: frontend started (FAST + oriented BRIEF, {N_LEVELS} levels)");

    loop {
        let frame = camera.wait_frame();
        let t0 = Instant::now();
        let tg = Instant::now();
        let gray = gray_from_yuyv(&frame);
        let gray_ms = ms(tg.elapsed());

        let (points, descs, n_total, mut stage) = detect_features(&gray);
        stage.gray = gray_ms;
        let n_kept = points.len() as u32;

        let out = fe.on_frame(&points, &descs, gray.width, gray.height);
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
