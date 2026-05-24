//! The camera-free SLAM pipeline state machine: intrinsics resolution,
//! frame-to-frame overlay matching, two-view initialization, live
//! tracking, and BoW+PnP relocalization. Pulled out of the camera/server
//! loop so the whole pipeline is exercised by deterministic headless
//! tests via [`Frontend::on_frame`] — no camera, server, or sleeping.

use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use log::{info, warn};

use ginger_slam_core::camera::CameraModel;
use ginger_slam_core::intrinsics::Intrinsics;
use ginger_slam_core::map::Map;
use ginger_slam_core::pnp::{self, PnpOptions};
use ginger_slam_core::tracking::{self, Observation};
use ginger_slam_core::twoview::{self, InitOptions};
use nalgebra::{Isometry3, Rotation3, Translation3, UnitQuaternion, Vector3};

use super::brief;
use super::detect::{FeaturePoint, Match, ms};
use super::mapper::{KeyframeJob, LocalMapper};
use super::place::PlaceDb;
use super::snapshot::{
    INTRINSICS_PATH, IntrinsicsView, MAX_TRAJECTORY, MapSnapshot, calib_norm, publish_map,
    resolve_intrinsics,
};

// Two-view initialization gates: enough matches against the anchor
// frame, and enough median pixel parallax (fraction of image width) so
// the geometry is well-conditioned. Below the floor the anchor is stale
// (scene changed) and is reset to the current frame.
//
// Values tuned 2026-05-23 from a live forward-drive bootstrap trace
// (see `/tmp/slam-trace.csv` in the SLAM debug session). Two issues
// the prior values (0.04 / 25) made unfixable on this hardware:
//
// * **Parallax gate**: a 1-second forward pulse (~25 cm) at 35% duty
//   produces a median disparity of only ~20–26 px against a same-
//   session anchor. The old 0.04 × 800 = 32 px gate was unreachable in
//   any single pulse, and longer pulses run into the second issue
//   below before the gate can be crossed. 0.025 (20 px) is well within
//   the well-conditioned range the literature uses for monocular init
//   and is reached reliably in the trace.
// * **Anchor dead zone**: as the car drives forward, descriptor
//   matches against the anchor decay from ~325 down through ~80; once
//   below INIT_MIN_MATCHES the parallax branch can no longer run, but
//   with the floor at 25 the anchor stayed alive for seconds in a
//   useless mid-range — *exactly* during the motion that was supposed
//   to trigger init. Raising the floor to half of INIT_MIN_MATCHES
//   forces a fresh anchor as soon as the match quality dips out of
//   the parallax-measurable range, so the next pulse re-anchors and
//   accumulates parallax against a frame that still correlates well.
const INIT_MIN_MATCHES: usize = 80;
const INIT_MIN_DISP_FRAC: f32 = 0.025;
const ANCHOR_RESET_MATCHES: usize = 40;

// Tracking gates: min map-point matches to attempt a pose solve, and
// min reprojection inliers for the refined pose to be trusted.
// `TRACK_MIN_INLIERS` was 10; lowered to 6 (2026-05-23) because live
// post-reloc cycles consistently landed at 6–9 inliers and were being
// thrown out, even though the pose was usable. A continuously usable
// 6-inlier pose beats a perfectly-conditioned-but-never-running 10.
const TRACK_MIN_MATCHES: usize = 15;
const TRACK_MIN_INLIERS: usize = 6;

/// Consecutive frames a weak tracking solve can be tolerated before
/// declaring `Stage::Lost`. The constant-velocity prediction can be
/// briefly wrong on motion onsets (e.g. the right-turn-after-forward
/// transition that triggered the failure in the 2026-05-23 trace),
/// producing a single frame of 5/37 inliers before the next refine
/// settles. Riding through 1–2 such frames with the predicted pose
/// keeps the map alive instead of destroying it on a transient.
const SOFT_LOST_MAX: usize = 3;

/// Relocalization gates. Kept conservative — a false relocalization
/// corrupts the trajectory, so favour staying lost.
const RELOC_MAX_CAND: usize = 5;
const RELOC_COVIS: usize = 5;
const RELOC_MAX_PTS: usize = 1500;
const RELOC_MIN_INLIERS: usize = 15;

/// A lost track that has not relocalized within this many frames gives
/// up: the map is discarded and a fresh session is re-bootstrapped. This
/// is also the *only* exit when the track was lost before the BoW
/// vocabulary self-trained (fewer than `place::N_VOCAB_KF` keyframes
/// existed) — without a vocabulary every relocalization query is empty,
/// so staying `Lost` would hang in "relocalizing…" forever.
///
/// Bumped 2026-05-23 from 30 to 90 (≈12 s at 7.5 fps): with the soft-
/// lost grace period added, the system stays in `Tracking` longer on
/// transient failures, so when we *do* reach `Lost` it tends to be a
/// real "drove past the mapped area" scenario where the user needs a
/// few seconds to swing back into view. 30 frames (~4 s) was too tight
/// for that, and prematurely-destroyed maps were exactly the failure
/// the user reported.
const RELOC_MAX_FRAMES: usize = 90;

/// Extended timeout when BoW *is* ready: a substantial mapped session
/// is worth a short wait, so give the user ~20 s at 7.5 fps to swing
/// the camera back into mapped scenery before re-bootstrapping. The
/// previous 60 s value felt like "stuck" during normal driving — by
/// 20 s the user has either driven back to mapped scenery (reloc
/// recovers) or they've moved on enough that a fresh bootstrap from
/// the current view is the right call.
const RELOC_MAX_FRAMES_BOW: usize = 150;

/// Per-frame output of [`Frontend::on_frame`]: overlay match lines, the
/// current intrinsics view + map snapshot, and the overlay-match wall
/// time (ms) for the HUD `matching` stage.
pub struct FrameOut {
    pub matches: Vec<Match>,
    pub intrinsics: IntrinsicsView,
    pub map: MapSnapshot,
    pub match_ms: f32,
}

/// Detected features for one frame: level-0 keypoints + aligned BRIEF.
type FrameFeatures = (Vec<FeaturePoint>, Vec<brief::Descriptor>);

/// Live tracking state once two-view init has succeeded. The map itself
/// (keyframes / points / covisibility) lives in the shared
/// [`Frontend::world`]; this holds only the tracking-thread bookkeeping:
/// the per-frame `T_cw` trajectory (oldest first; `[0]` = view-1
/// origin), the reference keyframe + frames since it, and the init
/// model badge.
#[derive(Clone)]
struct TrackState {
    trajectory: Vec<Isometry3<f64>>,
    ref_kf: u32,
    frames_since_kf: usize,
    model: String,
    r_h: f32,
}

/// Explicit pipeline state (replaces an implicit `Option` trio so
/// illegal combinations are unrepresentable and transitions are
/// obvious).
enum Stage {
    /// Pre-init: accumulating parallax against an optional anchor frame.
    Bootstrapping { anchor: Option<FrameFeatures> },
    /// Post-init: live tracking against the shared keyframe map.
    Tracking(TrackState),
    /// Track lost: per frame, BoW-query the place DB and try to recover
    /// the pose by PnP-RANSAC against candidate keyframes' map points;
    /// on success resume [`Stage::Tracking`] with the saved
    /// trajectory/map context. `since` counts frames spent lost.
    Lost { since: usize, track: TrackState },
}

pub struct Frontend {
    prev: Option<FrameFeatures>,
    stage: Stage,
    intrinsics: Option<Intrinsics>,
    iview: IntrinsicsView,
    map: MapSnapshot,
    /// Shared keyframe map: written by tracking (keyframe inserts) and
    /// the [`LocalMapper`] (new points + local BA); read by tracking
    /// each frame to re-find points and to publish.
    world: Arc<Mutex<Map>>,
    /// Keyframe handoff to the local mapper.
    jobs: Sender<KeyframeJob>,
    /// The mapper, until [`Frontend::take_local_mapper`] moves it to its
    /// own thread (`run`); kept here so tests can pump it synchronously.
    mapper: Option<LocalMapper>,
    /// Shared BoW place-recognition index: the mapper fills it as
    /// keyframes are inserted; relocalization / loop detection query
    /// it. Held here so the tracking side can reach it.
    place: Arc<Mutex<PlaceDb>>,
    /// Loop closures applied by the mapper; surfaced in the HUD status
    /// and observable for tests.
    loops: Arc<std::sync::atomic::AtomicU64>,
    /// Last `loops` value folded into the status line.
    last_loops: u64,
    /// Last reason tracking went `Stage::Lost`. Persisted across the
    /// Lost/relocalizing window (which otherwise overwrites the
    /// transient loss message in `map.status` within one frame), so a
    /// polling debug client can read why we lost without racing the
    /// frame loop.
    last_lost_reason: String,
    /// Cumulative tracking losses since process start (debug HUD).
    n_lost: u32,
    /// Consecutive frames the tracking solve has been weak (inliers
    /// below `TRACK_MIN_INLIERS`). Reset on a healthy solve; promotes
    /// to `Stage::Lost` once it exceeds `SOFT_LOST_MAX`.
    consecutive_track_fails: usize,
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend {
    pub fn new() -> Self {
        let world = Arc::new(Mutex::new(Map::new()));
        let place = Arc::new(Mutex::new(PlaceDb::new()));
        let loops = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (tx, rx) = channel();
        Self {
            prev: None,
            stage: Stage::Bootstrapping { anchor: None },
            intrinsics: None,
            iview: IntrinsicsView::uninitialized(),
            map: MapSnapshot::initial(),
            mapper: Some(LocalMapper::new(
                world.clone(),
                rx,
                place.clone(),
                loops.clone(),
            )),
            world,
            jobs: tx,
            place,
            loops,
            last_loops: 0,
            last_lost_reason: String::new(),
            n_lost: 0,
            consecutive_track_fails: 0,
        }
    }

    /// Loop closures applied so far — test/observability hook.
    pub fn loop_closures(&self) -> u64 {
        self.loops.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Move the local mapper out to run on its own thread (production).
    /// After this, [`Frontend::pump_local_mapper`] is a no-op — the
    /// mapper consumes keyframes off the channel instead.
    pub fn take_local_mapper(&mut self) -> Option<LocalMapper> {
        self.mapper.take()
    }

    /// Synchronously drain + process queued keyframes on the calling
    /// thread (tests / inline). Returns the number processed; no-op once
    /// the mapper has been taken for its own thread.
    pub fn pump_local_mapper(&mut self) -> usize {
        self.mapper.as_mut().map_or(0, LocalMapper::process_pending)
    }

    /// `(alive keyframes, alive map points)` — test/observability hook.
    pub fn map_stats(&self) -> (usize, usize) {
        let m = self.world.lock().unwrap();
        (m.n_keyframes(), m.n_points())
    }

    /// `(vocab ready, keyframes indexed in the BoW DB)` — test hook.
    pub fn place_stats(&self) -> (bool, usize) {
        let p = self.place.lock().unwrap();
        (p.is_ready(), p.len())
    }

    /// Place-recognition query against the BoW DB: best
    /// `(keyframe id, score)` matches for these descriptors.
    /// Test/relocalization hook (no skip filter).
    pub fn place_query(&self, descs: &[brief::Descriptor], max: usize) -> Vec<(u32, f64)> {
        self.place.lock().unwrap().query(descs, max, |_| false)
    }

    /// Process one frame's detected features (level-0 pixel coords with
    /// aligned descriptors). Pure given its inputs + prior state.
    ///
    /// `rotation_hint` is the camera-frame `ΔR` since the previous
    /// frame: an external estimate (typically gyro-pre-integrated from
    /// the IMU) of how much the camera rotated between `on_frame` calls.
    /// When supplied during `Stage::Tracking`, it replaces the rotational
    /// part of the constant-velocity predict — far more accurate for
    /// fast spins / motion onsets, which is the failure mode session 2
    /// reproduced. Pass `None` for vision-only behaviour (the default
    /// when the IMU isn't present, or when `GINGER_IMU_PREDICT=0`).
    pub fn on_frame(
        &mut self,
        points: &[FeaturePoint],
        descs: &[brief::Descriptor],
        width: usize,
        height: usize,
        rotation_hint: Option<UnitQuaternion<f64>>,
    ) -> FrameOut {
        if self.intrinsics.is_none() {
            let i = resolve_intrinsics(width as u32, height as u32);
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
            self.iview = IntrinsicsView::of(&i);
            self.intrinsics = Some(i);
        }

        // Overlay matches vs the previous frame (HUD only); time just this.
        let tm = Instant::now();
        let matches: Vec<Match> = match &self.prev {
            Some((pp, pd)) => brief::match_descriptors(pd, descs)
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
        let match_ms = ms(tm.elapsed());

        // Pre-init: accumulate parallax vs an anchor, run two-view init.
        // Post-init: project the map into each frame and refine the pose
        // (constant-velocity prediction + motion-only BA). Copy the
        // intrinsics-derived values out so `self.intrinsics` is no
        // longer borrowed while the state machine mutates other fields.
        if let Some((cam, fx)) = self
            .intrinsics
            .as_ref()
            .map(|i| (i.to_camera_model(), i.fx))
        {
            // Own the stage for this frame so arms can move between
            // states (Tracking⇄Lost); `next` overrides on transition,
            // else the (possibly mutated-in-place) `stage` is kept.
            let mut stage =
                std::mem::replace(&mut self.stage, Stage::Bootstrapping { anchor: None });
            let next = match &mut stage {
                Stage::Bootstrapping { anchor } => {
                    self.bootstrap(anchor, points, descs, width, cam)
                }
                Stage::Tracking(st) => self.track(st, points, descs, cam, fx, rotation_hint),
                Stage::Lost { since, track } => {
                    self.relocalize(since, track, points, descs, cam, fx)
                }
            };
            self.stage = next.unwrap_or(stage);
        }

        // Surface mapper loop closures in the HUD: the corrected
        // keyframes/points are already in the published snapshot (the
        // map moved under us); annotate the status so the event shows.
        let lc = self.loops.load(std::sync::atomic::Ordering::Relaxed);
        if lc != self.last_loops {
            self.last_loops = lc;
            self.map.status = format!("{} · loop closed (#{lc})", self.map.status);
        }

        // Debug-HUD counters: refreshed every frame so they stay
        // accurate while Lost/Bootstrapping, when `publish_map` (which
        // owns the geometry fields) is not called.
        self.map.loop_closures = lc;
        self.map.last_lost_reason = self.last_lost_reason.clone();
        self.map.n_lost = self.n_lost;
        {
            let p = self.place.lock().unwrap();
            self.map.bow_ready = p.is_ready();
            self.map.bow_words = p.vocab_words() as u32;
        }
        // Tracking-loss timer: surface the frame count + the active
        // give-up budget while `Stage::Lost` so the WebUI can render a
        // countdown to re-bootstrap. Zero when not lost.
        (self.map.lost_frames, self.map.lost_budget_frames) = match &self.stage {
            Stage::Lost { since, .. } => {
                let budget = if self.map.bow_ready {
                    RELOC_MAX_FRAMES_BOW
                } else {
                    RELOC_MAX_FRAMES
                };
                (*since as u32, budget as u32)
            }
            _ => (0, 0),
        };

        self.prev = Some((points.to_vec(), descs.to_vec()));
        FrameOut {
            matches,
            intrinsics: self.iview.clone(),
            map: self.map.clone(),
            match_ms,
        }
    }

    /// `Stage::Bootstrapping`: accumulate parallax against the anchor;
    /// on enough matches + median disparity, run two-view init, promote
    /// the two views to keyframes, seed the map, and transition to
    /// [`Stage::Tracking`]. A stale anchor (too few matches) is reset.
    fn bootstrap(
        &mut self,
        anchor: &mut Option<FrameFeatures>,
        points: &[FeaturePoint],
        descs: &[brief::Descriptor],
        width: usize,
        cam: CameraModel,
    ) -> Option<Stage> {
        let mut next: Option<Stage> = None;
        let mut want_anchor = false;
        // Debug telemetry surfaced via `/api/slam/map`; reset each frame
        // and re-populated below so a poller sees the live values per
        // frame. `boot_min_disp_px` is the (constant) parallax gate.
        self.map.boot_min_disp_px = width as f32 * INIT_MIN_DISP_FRAC;
        self.map.boot_matches = 0;
        self.map.boot_median_disp_px = 0.0;
        if anchor.is_some() {
            // Anchor survived another frame — age it.
            self.map.boot_anchor_age = self.map.boot_anchor_age.saturating_add(1);
        } else {
            self.map.boot_anchor_age = 0;
        }
        if let Some((apts, adesc)) = anchor.as_ref() {
            let mm = brief::match_descriptors(adesc, descs);
            self.map.boot_matches = mm.len() as u32;
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
                let min_disp = width as f32 * INIT_MIN_DISP_FRAC;
                self.map.boot_median_disp_px = med;
                if med >= min_disp {
                    let corrs: Vec<twoview::Corr> = mm
                        .iter()
                        .map(|&(ia, ib)| {
                            let a = apts[ia as usize];
                            let b = points[ib as usize];
                            (calib_norm(&cam, a.x, a.y), calib_norm(&cam, b.x, b.y))
                        })
                        .collect();
                    if let Some(tv) = twoview::initialize(&corrs, InitOptions::default()) {
                        // cam1 = world origin; cam2 = Tcw = [R|t].
                        let pose2 = Isometry3::from_parts(
                            Translation3::from(tv.t),
                            UnitQuaternion::from_rotation_matrix(
                                &Rotation3::from_matrix_unchecked(tv.r),
                            ),
                        );
                        let model = match tv.model {
                            twoview::Model::Essential => "essential",
                            twoview::Model::Homography => "homography",
                        };
                        // Promote the two views to keyframes
                        // kf0 (origin) / kf1 (= pose2) and the
                        // gated triangulated points to map
                        // points observed by both; record each
                        // view's feature→point binding so the
                        // local mapper does not re-create them.
                        let mut a_assigned = vec![None; apts.len()];
                        let mut b_assigned = vec![None; points.len()];
                        let (n, kf0, kf1) = {
                            let mut w = self.world.lock().unwrap();
                            let kf0 = w.add_keyframe(Isometry3::identity(), Vec::new());
                            let kf1 = w.add_keyframe(pose2, Vec::new());
                            let mut pi = 0usize;
                            let mut n = 0usize;
                            for (k, &(ia, ib)) in mm.iter().enumerate() {
                                if tv.inliers[k] {
                                    if let Some(&x) = tv.points.get(pi)
                                        && x.z > 0.0
                                        && x.x.is_finite()
                                        && x.z.is_finite()
                                        && let Some(pid) = w.add_point_observed(
                                            x,
                                            descs[ib as usize],
                                            kf0,
                                            corrs[k].0,
                                        )
                                    {
                                        w.add_observation(kf1, pid, corrs[k].1);
                                        a_assigned[ia as usize] = Some(pid);
                                        b_assigned[ib as usize] = Some(pid);
                                        n += 1;
                                    }
                                    pi += 1;
                                }
                            }
                            (n, kf0, kf1)
                        };
                        let _ = self.jobs.send(KeyframeJob {
                            kf_id: kf0,
                            cam,
                            pts: apts.clone(),
                            descs: adesc.clone(),
                            assigned: a_assigned,
                        });
                        let _ = self.jobs.send(KeyframeJob {
                            kf_id: kf1,
                            cam,
                            pts: points.to_vec(),
                            descs: descs.to_vec(),
                            assigned: b_assigned,
                        });
                        let ts = TrackState {
                            trajectory: vec![Isometry3::identity(), pose2],
                            ref_kf: kf1,
                            frames_since_kf: 0,
                            model: model.into(),
                            r_h: tv.r_h as f32,
                        };
                        self.map = {
                            let w = self.world.lock().unwrap();
                            publish_map(
                                &w,
                                &ts.trajectory,
                                &ts.model,
                                ts.r_h,
                                format!("initialized: {n} pts via {model}"),
                                n as u32,
                            )
                        };
                        info!(
                            "slam: two-view init OK — {n} pts, model={model}, \
                                     R_H={:.2}, matches={}",
                            tv.r_h,
                            mm.len()
                        );
                        next = Some(Stage::Tracking(ts));
                    }
                } else {
                    self.map.status =
                        format!("parallax {med:.0}/{min_disp:.0}px · {} matches", mm.len());
                }
            } else if mm.len() < ANCHOR_RESET_MATCHES {
                want_anchor = true;
            }
        } else {
            want_anchor = true;
        }
        if want_anchor && next.is_none() {
            *anchor = Some((points.to_vec(), descs.to_vec()));
            // Two-view init needs translational parallax. The car is
            // differential-drive (no strafing), but driving forward
            // still parallaxes every off-axis point — only pure
            // rotation yields none, so that is what we ask for.
            self.map.status = "anchor set — drive forward for parallax".into();
            self.map.boot_anchor_age = 0;
            self.map.boot_anchor_resets = self.map.boot_anchor_resets.saturating_add(1);
        }
        next
    }

    /// `Stage::Tracking`: re-find map points, predict (constant
    /// velocity) + motion-only BA the pose, insert a keyframe per the
    /// keyframe policy, and publish. A weak/failed solve transitions to
    /// [`Stage::Lost`] without corrupting the map.
    fn track(
        &mut self,
        st: &mut TrackState,
        points: &[FeaturePoint],
        descs: &[brief::Descriptor],
        cam: CameraModel,
        fx: f64,
        rotation_hint: Option<UnitQuaternion<f64>>,
    ) -> Option<Stage> {
        let mut next: Option<Stage> = None;
        // Re-find map points: snapshot (id, desc, pos) of
        // every alive point + the reference keyframe's point
        // count under a short lock, then match/solve without
        // holding it (the matcher is the cost, and the local
        // mapper needs the map meanwhile).
        let (ids, idesc, ipos, ref_kf_points) = {
            let w = self.world.lock().unwrap();
            let mut ids: Vec<u32> = Vec::new();
            let mut idesc: Vec<brief::Descriptor> = Vec::new();
            let mut ipos: Vec<Vector3<f64>> = Vec::new();
            for p in w.alive_points() {
                ids.push(p.id);
                idesc.push(p.desc);
                ipos.push(p.pos);
            }
            let rkp = w.keyframe(st.ref_kf).map_or(0, |k| k.obs.len());
            (ids, idesc, ipos, rkp)
        };
        let mm = if ids.is_empty() {
            Vec::new()
        } else {
            brief::match_descriptors(&idesc, descs)
        };
        // Hoisted out of the match-count branch so soft-lost failures
        // can still push a predicted pose to keep the trajectory
        // contiguous and the const-velocity model warm.
        //
        // Predict policy: if an external rotation hint is available
        // (gyro pre-integration over the inter-frame interval), use it
        // for the *rotation* and CV for the *translation*. Rationale:
        // the predict's rotation is what catches in-place spins (the
        // session-2 failure mode), and a constant-velocity rotation
        // model can't anticipate a sudden direction change, while a
        // CV-translation predict is fine because the wheels don't
        // reverse instantaneously. With no hint we fall back to pure
        // CV — the historical behaviour, preserved for the
        // `GINGER_IMU_PREDICT=0` kill switch.
        let np = st.trajectory.len();
        let cv = if np >= 2 {
            tracking::constant_velocity(&st.trajectory[np - 2], &st.trajectory[np - 1])
        } else {
            st.trajectory[np - 1]
        };
        let predict = match rotation_hint {
            Some(dr) if np >= 1 => {
                // Apply ΔR (camera frame) to the last pose's rotation.
                // Poses are T_cw (world→camera): a camera rotating by
                // ΔR in its own frame composes on the LEFT of the
                // current rotation, since R_cw expresses world *as
                // seen by* the camera. `cv.translation` keeps the
                // wheel-derived inter-frame translation guess.
                let new_rot = dr * st.trajectory[np - 1].rotation;
                Isometry3::from_parts(cv.translation, new_rot)
            }
            _ => cv,
        };
        if mm.len() >= TRACK_MIN_MATCHES {
            let obs: Vec<Observation> = mm
                .iter()
                .map(|&(im, ic)| {
                    let kp = points[ic as usize];
                    Observation {
                        point: ipos[im as usize],
                        obs: calib_norm(&cam, kp.x, kp.y),
                    }
                })
                .collect();
            let huber = 2.0 / fx;
            // Inlier reprojection-error gate (calibrated units, ≈ px/fx).
            // Widened 5 → 8 px on 2026-05-23: the original 5 px gate was
            // tight for descriptor-matched correspondences on the Pi
            // camera against the then-FOV-prior intrinsics, and a refine
            // that converges to a *good enough* pose was missing the
            // inlier count by counting too few of its agreeing matches.
            // A ChArUco calibration landed later the same day (intrinsics
            // now flag `verified=true`); with the tighter intrinsics this
            // gate may retighten toward 5–6 px. 8 px ≈ 1% of image width
            // — still well inside the noise floor of a well-conditioned
            // solve, but lenient enough that a 30/45 inlier configuration
            // doesn't get reported as 4/45 just because the threshold
            // cuts inside the measurement spread.
            let thr = 8.0 / fx;
            match tracking::track_pose(&obs, &predict, huber, thr) {
                // Trust the inlier count, not the formal `converged` flag.
                // The shared LM in slam-core uses a very tight
                // `gradient_tol = 1e-10` (good for offline BA); motion-only
                // tracking-frame BA reaches "good enough" long before that,
                // so a high inlier ratio with `converged == false` means the
                // pose is practically correct, just not at the tolerance LM
                // declares formal convergence at. Live trace (n_lost=1 with
                // 49/57 inliers, t≈3.4s post-init) showed tracking being
                // killed precisely there.
                Some(rep) if rep.n_inliers >= TRACK_MIN_INLIERS => {
                    // Healthy solve — reset the soft-lost counter.
                    self.consecutive_track_fails = 0;
                    st.trajectory.push(rep.pose);
                    if st.trajectory.len() > MAX_TRAJECTORY * 2 {
                        let drop = st.trajectory.len() - MAX_TRAJECTORY;
                        st.trajectory.drain(0..drop);
                    }
                    st.frames_since_kf += 1;
                    let mut status = format!("tracking: {}/{} inliers", rep.n_inliers, obs.len());
                    // Keyframe-insertion policy: on a healthy solve
                    // that has thinned vs the reference or gone stale,
                    // promote this frame to a keyframe (its tracked
                    // inliers as observations) and hand it + its raw
                    // features to the local mapper.
                    if ginger_slam_core::map::needs_keyframe(
                        rep.n_inliers,
                        ref_kf_points,
                        st.frames_since_kf,
                        TRACK_MIN_INLIERS,
                    ) {
                        let mut kf_obs = Vec::new();
                        let mut assigned = vec![None; points.len()];
                        for (k, &(im, ic)) in mm.iter().enumerate() {
                            if rep.inliers[k] {
                                let pid = ids[im as usize];
                                kf_obs.push((pid, obs[k].obs));
                                assigned[ic as usize] = Some(pid);
                            }
                        }
                        let kf = {
                            let mut w = self.world.lock().unwrap();
                            w.add_keyframe(rep.pose, kf_obs)
                        };
                        let _ = self.jobs.send(KeyframeJob {
                            kf_id: kf,
                            cam,
                            pts: points.to_vec(),
                            descs: descs.to_vec(),
                            assigned,
                        });
                        st.ref_kf = kf;
                        st.frames_since_kf = 0;
                        status = format!(
                            "keyframe {kf} · tracking: {}/{} inliers",
                            rep.n_inliers,
                            obs.len()
                        );
                    }
                    self.map = {
                        let w = self.world.lock().unwrap();
                        publish_map(
                            &w,
                            &st.trajectory,
                            &st.model,
                            st.r_h,
                            status,
                            rep.n_inliers as u32,
                        )
                    };
                }
                Some(rep) => {
                    let reason = format!("weak {}/{} inliers", rep.n_inliers, obs.len());
                    next = self.handle_track_fail(st, predict, reason, rep.n_inliers as u32);
                }
                None => {
                    next = self.handle_track_fail(st, predict, "solve failed".into(), 0);
                }
            }
        } else {
            let reason = format!("only {} map matches", mm.len());
            next = self.handle_track_fail(st, predict, reason, 0);
        }
        next
    }

    /// Soft-lost grace period: a single bad tracking frame (e.g. the
    /// motion-onset frame when the const-velocity prediction is briefly
    /// wrong) used to immediately promote to `Stage::Lost`, where 30
    /// frames of failed relocalization would destroy the entire map.
    /// Allow up to [`SOFT_LOST_MAX`] consecutive bad frames before
    /// committing to that — push the predicted pose to keep the
    /// trajectory contiguous and the const-velocity model warm, but
    /// don't touch the keyframe machinery. Returns `Some(Lost)` once
    /// the run of bad frames exceeds the budget; `None` keeps the
    /// current `Tracking` stage.
    fn handle_track_fail(
        &mut self,
        st: &mut TrackState,
        predict: Isometry3<f64>,
        reason: String,
        n_inliers: u32,
    ) -> Option<Stage> {
        self.consecutive_track_fails += 1;
        if self.consecutive_track_fails <= SOFT_LOST_MAX {
            st.trajectory.push(predict);
            if st.trajectory.len() > MAX_TRAJECTORY * 2 {
                let drop = st.trajectory.len() - MAX_TRAJECTORY;
                st.trajectory.drain(0..drop);
            }
            let status = format!(
                "tracking shaky: {reason} (predicting, {}/{} skips)",
                self.consecutive_track_fails, SOFT_LOST_MAX
            );
            self.map = {
                let w = self.world.lock().unwrap();
                publish_map(&w, &st.trajectory, &st.model, st.r_h, status, n_inliers)
            };
            return None;
        }
        // Budget exhausted: this is a real loss.
        self.consecutive_track_fails = 0;
        self.map.status = format!("tracking lost: {reason} — relocalizing");
        self.map.tracking = false;
        self.last_lost_reason = reason;
        self.n_lost += 1;
        Some(Stage::Lost {
            since: 0,
            track: st.clone(),
        })
    }

    /// `Stage::Lost`: BoW-query the place DB, gather candidate map
    /// points from the matched keyframes + covisible neighbours, and
    /// attempt PnP-RANSAC recovery. On success resume
    /// [`Stage::Tracking`] with the saved trajectory; else stay lost.
    fn relocalize(
        &mut self,
        since: &mut usize,
        track: &TrackState,
        points: &[FeaturePoint],
        descs: &[brief::Descriptor],
        cam: CameraModel,
        fx: f64,
    ) -> Option<Stage> {
        let mut next: Option<Stage> = None;
        *since += 1;
        // BoW place-recognition candidates for this frame.
        let cands = self
            .place
            .lock()
            .unwrap()
            .query(descs, RELOC_MAX_CAND, |_| false);
        // Gather candidate map points (pos + descriptor) from
        // the candidate keyframes + their covisible neighbours,
        // bounded. When BoW has no candidates (vocabulary not yet
        // self-trained, or just no covisible match for this view),
        // fall back to *all* alive map points — for a thin-map
        // session that was just lost, brute-force PnP-RANSAC against
        // the full map gives a real chance to recover, instead of
        // burning the timeout staring at an empty candidate set.
        let (cpos, cdesc) = if cands.is_empty() {
            let w = self.world.lock().unwrap();
            let mut cpos: Vec<Vector3<f64>> = Vec::new();
            let mut cdesc: Vec<brief::Descriptor> = Vec::new();
            for p in w.alive_points() {
                cpos.push(p.pos);
                cdesc.push(p.desc);
                if cpos.len() >= RELOC_MAX_PTS {
                    break;
                }
            }
            (cpos, cdesc)
        } else {
            let w = self.world.lock().unwrap();
            let mut seen = std::collections::HashSet::new();
            let mut cpos: Vec<Vector3<f64>> = Vec::new();
            let mut cdesc: Vec<brief::Descriptor> = Vec::new();
            'outer: for &(kf, _) in &cands {
                let mut kfs = vec![kf];
                kfs.extend(
                    w.covisibility(kf)
                        .into_iter()
                        .take(RELOC_COVIS)
                        .map(|(c, _)| c),
                );
                for k in kfs {
                    let Some(kfr) = w.keyframe(k) else { continue };
                    for &(pid, _) in &kfr.obs {
                        if seen.insert(pid)
                            && let Some(p) = w.point(pid)
                        {
                            cpos.push(p.pos);
                            cdesc.push(p.desc);
                            if cpos.len() >= RELOC_MAX_PTS {
                                break 'outer;
                            }
                        }
                    }
                }
            }
            (cpos, cdesc)
        };

        let mut report = None;
        if cdesc.len() >= RELOC_MIN_INLIERS {
            let pairs = brief::match_descriptors(&cdesc, descs);
            if pairs.len() >= RELOC_MIN_INLIERS {
                let obs: Vec<Observation> = pairs
                    .iter()
                    .map(|&(im, ic)| {
                        let kp = points[ic as usize];
                        Observation {
                            point: cpos[im as usize],
                            obs: calib_norm(&cam, kp.x, kp.y),
                        }
                    })
                    .collect();
                report = pnp::pnp_ransac(
                    &obs,
                    PnpOptions {
                        thresh: 5.0 / fx,
                        min_inliers: RELOC_MIN_INLIERS,
                        ..PnpOptions::default()
                    },
                );
            }
        }

        if let Some(rep) = report {
            // Reloc succeeded — clear the soft-lost counter for the
            // resumed tracking stage.
            self.consecutive_track_fails = 0;
            let mut tr = track.clone();
            // Zero out the constant-velocity prediction across the
            // lost window: the trajectory ends with bad shaky/predict
            // poses, so `(prev, rep.pose)` would carry a huge spurious
            // velocity into the next frame's predict — which is
            // exactly what triggered the immediate re-failure observed
            // in the live trace (track → shaky → lost → reloc → instant
            // shaky → lost cycle, 31 losses in 30 s, n_kf stuck at 24).
            // Force the next CV predict to "no motion" by aligning the
            // last two trajectory entries to the recovered pose.
            tr.trajectory.push(rep.pose);
            let n = tr.trajectory.len();
            if n >= 2 {
                tr.trajectory[n - 2] = rep.pose;
            }
            if tr.trajectory.len() > MAX_TRAJECTORY * 2 {
                let drop = tr.trajectory.len() - MAX_TRAJECTORY;
                tr.trajectory.drain(0..drop);
            }
            let best = cands.first().map(|&(k, _)| k).unwrap_or(0);
            let status = format!(
                "relocalized: {} inliers (kf {best}, lost {} frames)",
                rep.n_inliers, *since
            );
            self.map = {
                let w = self.world.lock().unwrap();
                publish_map(
                    &w,
                    &tr.trajectory,
                    &tr.model,
                    tr.r_h,
                    status,
                    rep.n_inliers as u32,
                )
            };
            next = Some(Stage::Tracking(tr));
        } else if *since > RELOC_MAX_FRAMES {
            // Two regimes (restored 2026-05-23 after "stuck Lost for
            // 3.3 minutes with 2 keyframes" was reported):
            //
            // * **BoW vocabulary not trained** (< N_VOCAB_KF=6
            //   keyframes ever existed): the map is a thin
            //   bootstrap-area patch, the full-map reloc fallback
            //   can't match against it once the camera turns away,
            //   and the user has no way out. Destroy the map,
            //   re-bootstrap. The user gets a fresh session
            //   wherever they ended up.
            //
            // * **BoW vocabulary trained**: the map is a real
            //   session worth preserving. Keep it for a longer
            //   recovery window (RELOC_MAX_FRAMES_BOW ≈ 60 s) so
            //   the user can swing back; only auto-rebootstrap
            //   after that, since "stuck for a minute" is worse
            //   than "lose the map".
            let bow_ready = self.place.lock().unwrap().is_ready();
            if !bow_ready {
                warn!(
                    "slam: relocalization stuck for {} frames (no vocabulary, \
                     {} kf, {} pts) — discarding thin map, re-bootstrapping",
                    *since, self.map.n_keyframes, self.map.n_points
                );
                self.reset_world();
                self.map.status = format!(
                    "relocalization gave up after {} frames — re-initializing",
                    *since
                );
                self.map.tracking = false;
                next = Some(Stage::Bootstrapping { anchor: None });
            } else if *since > RELOC_MAX_FRAMES_BOW {
                warn!(
                    "slam: relocalization stuck for {} frames with BoW ready \
                     ({} kf, {} pts) — discarding, re-bootstrapping",
                    *since, self.map.n_keyframes, self.map.n_points
                );
                self.reset_world();
                self.map.status = format!(
                    "relocalization gave up after {} frames — re-initializing",
                    *since
                );
                self.map.tracking = false;
                next = Some(Stage::Bootstrapping { anchor: None });
            } else {
                self.map.status = format!(
                    "lost {} frames — map preserved ({} kf, {} pts); swing camera back to mapped area",
                    *since, self.map.n_keyframes, self.map.n_points
                );
                self.map.tracking = false;
            }
        } else {
            self.map.status = format!("relocalizing… (lost {} frames)", *since);
            self.map.tracking = false;
        }
        next
    }

    /// Discard the current map and clear BoW + snapshot so a fresh
    /// `Stage::Bootstrapping` starts from a clean slate. Called only
    /// from the relocalize give-up path now that the map is otherwise
    /// preserved indefinitely.
    fn reset_world(&mut self) {
        self.world.lock().unwrap().reset();
        self.place.lock().unwrap().reset();
        self.map = MapSnapshot::initial();
        self.consecutive_track_fails = 0;
    }
}

/// Pipeline integration tests: drive the full [`Frontend`] state machine
/// (anchor → two-view init → tracking) with deterministic synthetic
/// features projected from a known 3D scene + camera trajectory. No
/// camera, server, image detection, or sleeping — bypasses the
/// (separately tested) image→features path by injecting features
/// directly.
#[cfg(test)]
mod pipeline_tests {
    use ginger_slam_core::camera::CameraModel;
    use ginger_slam_core::intrinsics::Intrinsics;
    use nalgebra::Isometry3;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;

    const W: usize = 640;
    const H: usize = 480;

    /// `n` world landmarks (varied depth/lateral spread) each with a
    /// unique, frame-stable BRIEF descriptor (mutually far apart → the
    /// matcher pairs them unambiguously, isolating pipeline behaviour
    /// from matcher robustness, which `brief` tests separately).
    fn scene(n: usize) -> (Vec<Vector3<f64>>, Vec<brief::Descriptor>) {
        let mut r = SmallRng::seed_from_u64(0x00AB_CDEF_1234_5678);
        let mut pts = Vec::with_capacity(n);
        let mut ds = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(Vector3::new(
                (r.random::<f64>() - 0.5) * 6.0,
                (r.random::<f64>() - 0.5) * 4.0,
                3.0 + r.random::<f64>() * 6.0,
            ));
            let mut d = [0u8; brief::DESC_BYTES];
            for b in d.iter_mut() {
                *b = r.random::<u8>();
            }
            ds.push(d);
        }
        (pts, ds)
    }

    /// `T_cw` for a camera that has translated `+tx` along world-x with
    /// no rotation (centre = `(tx, 0, 0)`); frame 0 (`tx = 0`) = world.
    fn pose(tx: f64) -> Isometry3<f64> {
        Isometry3::translation(-tx, 0.0, 0.0)
    }

    /// Project the visible landmarks into one frame's features.
    fn frame(
        pts: &[Vector3<f64>],
        ds: &[brief::Descriptor],
        cam: &CameraModel,
        tcw: &Isometry3<f64>,
    ) -> (Vec<FeaturePoint>, Vec<brief::Descriptor>) {
        let (mut fp, mut fd) = (Vec::new(), Vec::new());
        for (xw, d) in pts.iter().zip(ds) {
            let pc = tcw.rotation * xw + tcw.translation.vector;
            if pc.z <= 0.1 {
                continue;
            }
            if let Some(px) = cam.project(&pc)
                && px.x >= 0.0
                && px.y >= 0.0
                && px.x < W as f64 - 1.0
                && px.y < H as f64 - 1.0
            {
                fp.push(FeaturePoint {
                    x: px.x.round() as u16,
                    y: px.y.round() as u16,
                    level: 0,
                    score: 100,
                    angle: 0.0,
                });
                fd.push(*d);
            }
        }
        (fp, fd)
    }

    fn cam_model() -> CameraModel {
        Intrinsics::rev1_3_prior(W as u32, H as u32).to_camera_model()
    }

    /// One frame end-to-end, mirroring `slam::run`: feed the detected
    /// features, then synchronously drive the local mapper exactly as
    /// its thread would (the tested seam is identical to production —
    /// only the driver differs: sync pump vs `recv` loop).
    fn step(fe: &mut Frontend, fp: &[FeaturePoint], fd: &[brief::Descriptor]) -> FrameOut {
        let out = fe.on_frame(fp, fd, W, H, None);
        fe.pump_local_mapper();
        out
    }

    #[test]
    fn full_lifecycle_anchor_init_tracking() {
        let (pts, ds) = scene(240);
        let cam = cam_model();
        let mut fe = Frontend::new();

        // Sideways sweep; frame 0 is the anchor, parallax then grows.
        let mut last = None;
        let mut init_frame = None;
        for i in 0..24 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.08));
            assert!(
                fp.len() >= INIT_MIN_MATCHES,
                "frame {i}: only {} feats",
                fp.len()
            );
            let out = step(&mut fe, &fp, &fd);
            if out.map.tracking && init_frame.is_none() {
                init_frame = Some(i);
                assert!(["essential", "homography"].contains(&out.map.model.as_str()));
                assert!(out.map.n_points > 20, "thin map: {}", out.map.n_points);
                // Two bootstrap keyframes exist immediately on init.
                assert!(
                    out.map.keyframes.len() >= 2,
                    "no bootstrap keyframes: {}",
                    out.map.keyframes.len()
                );
            }
            last = Some(out);
        }

        let init_at = init_frame.expect("never initialized");
        assert!(init_at < 12, "init took too long: frame {init_at}");
        let out = last.unwrap();
        assert!(out.map.tracking);
        // Status is "tracking: …" or, on a keyframe frame,
        // "keyframe N · tracking: …".
        assert!(
            out.map.status.contains("tracking:"),
            "final status: {}",
            out.map.status
        );
        // Trajectory accumulated past the 2 init poses and moved.
        assert!(out.map.cameras.len() > 5, "short trajectory");
        let first = out.map.cameras[0];
        let lastc = *out.map.cameras.last().unwrap();
        let moved = ((first[0] - lastc[0]).powi(2) + (first[1] - lastc[1]).powi(2)).sqrt();
        assert!(moved > 1e-3, "camera did not move: {moved}");
    }

    #[test]
    fn trajectory_grows_each_tracked_frame() {
        let (pts, ds) = scene(240);
        let cam = cam_model();
        let mut fe = Frontend::new();
        for i in 0..8 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.1));
            step(&mut fe, &fp, &fd);
        }
        // Must be tracking by now.
        let (fp, fd) = frame(&pts, &ds, &cam, &pose(0.8));
        let a = step(&mut fe, &fp, &fd);
        assert!(
            a.map.tracking,
            "not tracking after warm-up: {}",
            a.map.status
        );
        let n0 = a.map.cameras.len();
        let (fp, fd) = frame(&pts, &ds, &cam, &pose(0.9));
        let b = step(&mut fe, &fp, &fd);
        assert_eq!(b.map.cameras.len(), n0 + 1, "trajectory did not extend");
        assert!(b.map.n_tracked >= TRACK_MIN_INLIERS as u32);
    }

    /// A longer sweep promotes keyframes and the local mapper grows +
    /// locally-bundle-adjusts the map (the visible deliverable).
    #[test]
    fn local_mapping_grows_keyframes_and_points() {
        let (pts, ds) = scene(180);
        let cam = cam_model();
        let mut fe = Frontend::new();

        // Long enough to bootstrap, go stale ≥ twice (20-frame policy),
        // and sweep fresh landmarks into view for the mapper to add.
        let mut init_pts = 0usize;
        let mut last = None;
        for i in 0..46 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.09));
            let out = step(&mut fe, &fp, &fd);
            if out.map.tracking && init_pts == 0 {
                init_pts = out.map.n_points as usize;
            }
            last = Some(out);
        }
        let out = last.unwrap();
        assert!(out.map.tracking, "lost tracking: {}", out.map.status);

        let (n_kf, n_pts) = fe.map_stats();
        // Keyframes accumulated beyond the 2 bootstrap views …
        assert!(n_kf >= 3, "too few keyframes: {n_kf}");
        // … and the local mapper triangulated new points beyond the
        // bootstrap set as fresh landmarks swept into view.
        assert!(n_pts > init_pts, "map did not grow: {init_pts} → {n_pts}");
        // Published snapshot stays consistent + finite.
        assert_eq!(out.map.n_points as usize, n_pts);
        // Debug-HUD counters mirror the live map.
        assert_eq!(out.map.n_keyframes as usize, n_kf, "HUD keyframe count");
        assert_eq!(out.map.loop_closures, fe.loop_closures(), "HUD loop count");
        assert!(out.map.keyframes.len() >= 3);
        assert!(
            out.map
                .points
                .iter()
                .chain(&out.map.keyframes)
                .all(|p| p[0].is_finite() && p[1].is_finite()),
            "non-finite map geometry published"
        );
        // Keyframe centres advance along the +x sweep (BA kept the map
        // metrically sane, not collapsed/diverged).
        let kx: Vec<f32> = out.map.keyframes.iter().map(|c| c[0]).collect();
        assert!(
            kx.last().unwrap() - kx[0] > 1e-2,
            "keyframes did not advance: {kx:?}"
        );
    }

    /// The full pipeline incl. local mapping is deterministic headless
    /// (the gating signal): identical input ⇒ identical map.
    #[test]
    fn pipeline_is_deterministic() {
        let run = || {
            let (pts, ds) = scene(200);
            let cam = cam_model();
            let mut fe = Frontend::new();
            let mut tail = None;
            for i in 0..24 {
                let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.09));
                tail = Some(step(&mut fe, &fp, &fd));
            }
            let (n_kf, n_pts) = fe.map_stats();
            let t = tail.unwrap();
            (n_kf, n_pts, *t.map.cameras.last().unwrap(), t.map.keyframes)
        };
        let a = run();
        let b = run();
        // Must have exercised the local mapper, not just bootstrap.
        assert!(a.0 > 2 && a.1 > 0, "determinism check too shallow: {a:?}");
        assert_eq!(a.0, b.0, "keyframe count not deterministic");
        assert_eq!(a.1, b.1, "map point count not deterministic");
        assert_eq!(a.2, b.2, "final camera centre not deterministic");
        assert_eq!(a.3, b.3, "keyframe centres not deterministic");
    }

    #[test]
    fn tracking_lost_on_unmatchable_frame_without_corruption() {
        let (pts, ds) = scene(240);
        let cam = cam_model();
        let mut fe = Frontend::new();
        for i in 0..10 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.1));
            step(&mut fe, &fp, &fd);
        }
        let good = {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(1.0));
            step(&mut fe, &fp, &fd)
        };
        assert!(good.map.tracking, "precondition: tracking");
        let n_pts = good.map.n_points;

        // Build a frame of pure garbage descriptors → no map matches.
        let mut r = SmallRng::seed_from_u64(0x99);
        let fd: Vec<brief::Descriptor> = (0..200)
            .map(|_| {
                let mut d = [0u8; brief::DESC_BYTES];
                for b in d.iter_mut() {
                    *b = r.random::<u8>();
                }
                d
            })
            .collect();
        let fp: Vec<FeaturePoint> = (0..200)
            .map(|i| FeaturePoint {
                x: (i % W) as u16,
                y: (i % H) as u16,
                level: 0,
                score: 1,
                angle: 0.0,
            })
            .collect();

        // Soft-lost grace period: the first SOFT_LOST_MAX bad frames
        // keep tracking alive (status reads "shaky", trajectory keeps
        // extending with predicted poses) — covering the motion-onset
        // case where one bad refine shouldn't destroy the map. The
        // (SOFT_LOST_MAX + 1)th tips to Stage::Lost.
        for i in 0..SOFT_LOST_MAX {
            let shaky = fe.on_frame(&fp, &fd, W, H, None);
            assert!(
                shaky.map.status.contains("shaky") || shaky.map.status.contains("predicting"),
                "frame {i}: expected shaky status, got: {}",
                shaky.map.status
            );
            assert_eq!(
                shaky.map.n_points, n_pts,
                "map points corrupted at frame {i}"
            );
        }
        let lost = fe.on_frame(&fp, &fd, W, H, None);
        assert!(
            lost.map.status.contains("lost"),
            "expected lost after grace period, got: {}",
            lost.map.status
        );
        // Map not corrupted by the bad-frame burst.
        assert_eq!(lost.map.n_points, n_pts);
    }

    /// A long enough sweep self-trains the BoW vocabulary and fills the
    /// place-recognition DB; a query then maps a place back to a
    /// keyframe near it (earlier place → earlier keyframe).
    #[test]
    fn place_db_self_trains_and_recognizes_place() {
        let (pts, ds) = scene(220);
        let cam = cam_model();
        let mut fe = Frontend::new();
        let mut tail = None;
        for i in 0..64 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.07));
            tail = Some(step(&mut fe, &fp, &fd));
        }
        let (ready, indexed) = fe.place_stats();
        assert!(ready, "vocabulary never self-trained");
        // Debug-HUD BoW fields reflect the self-trained vocabulary.
        let out = tail.unwrap();
        assert!(out.map.bow_ready, "HUD bow_ready false after self-train");
        assert!(out.map.bow_words > 0, "HUD bow_words zero after self-train");
        let (n_kf, _) = fe.map_stats();
        assert!(n_kf >= 6, "too few keyframes: {n_kf}");
        // Every alive keyframe is indexed (added in kf order, none culled).
        assert_eq!(indexed, n_kf, "DB/keyframe count mismatch");

        // Re-observe the start vs the far end of the sweep.
        let (_, q_start) = frame(&pts, &ds, &cam, &pose(0.03));
        let (_, q_end) = frame(&pts, &ds, &cam, &pose(63.0 * 0.07));
        let hit_start = fe.place_query(&q_start, 5);
        let hit_end = fe.place_query(&q_end, 5);
        assert!(!hit_start.is_empty() && !hit_start[0].1.is_nan());
        assert!(!hit_end.is_empty());
        assert!(hit_start[0].1 > 0.0 && hit_end[0].1 > 0.0);
        // Sorted best-first.
        for w in hit_start.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
        // Place recognition is spatially sane: the start place resolves
        // to an earlier keyframe than the far-end place.
        assert!(
            hit_start[0].0 < hit_end[0].0,
            "start kf {} not earlier than end kf {}",
            hit_start[0].0,
            hit_end[0].0
        );
        // Deterministic: identical query ⇒ identical top hit.
        assert_eq!(fe.place_query(&q_start, 5), hit_start);
    }

    /// Garbage frames lose the track (no map corruption); a recognizable
    /// frame then relocalizes via BoW + PnP-RANSAC and tracking resumes.
    #[test]
    fn relocalizes_after_track_loss() {
        let (pts, ds) = scene(220);
        let cam = cam_model();
        let mut fe = Frontend::new();
        let mut good = None;
        for i in 0..64 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.07));
            good = Some(step(&mut fe, &fp, &fd));
        }
        let good = good.unwrap();
        assert!(
            good.map.tracking,
            "precondition: tracking ({})",
            good.map.status
        );
        let (ready, _) = fe.place_stats();
        assert!(ready, "vocab must be trained for relocalization");
        let n_pts = good.map.n_points;
        let traj = good.map.cameras.len();

        // A run of pure-garbage frames → lost, and the map is *not*
        // corrupted while lost (points + trajectory frozen).
        //
        // The first `SOFT_LOST_MAX` bad frames are tolerated as
        // "shaky": each pushes a predicted pose to the trajectory and
        // stays in `Stage::Tracking`. Only the next frame tips to
        // `Stage::Lost` and freezes the trajectory. Burn through the
        // grace period explicitly so the lost-loop assertions reason
        // about the post-grace baseline.
        let mut r = SmallRng::seed_from_u64(0x6105);
        let garbage = |r: &mut SmallRng| {
            let fd: Vec<brief::Descriptor> = (0..200)
                .map(|_| {
                    let mut d = [0u8; brief::DESC_BYTES];
                    for b in d.iter_mut() {
                        *b = r.random::<u8>();
                    }
                    d
                })
                .collect();
            let fp: Vec<FeaturePoint> = (0..200)
                .map(|i| FeaturePoint {
                    x: (i % W) as u16,
                    y: (i % H) as u16,
                    level: 0,
                    score: 1,
                    angle: 0.0,
                })
                .collect();
            (fp, fd)
        };
        for _ in 0..SOFT_LOST_MAX {
            let (fp, fd) = garbage(&mut r);
            let shaky = step(&mut fe, &fp, &fd);
            assert!(
                shaky.map.status.contains("shaky"),
                "expected shaky during grace period, got: {}",
                shaky.map.status
            );
            assert_eq!(shaky.map.n_points, n_pts, "map corrupted while shaky");
        }
        // Post-grace baseline: the trajectory has `SOFT_LOST_MAX` extra
        // predicted poses; further garbage frames freeze it here.
        let traj_lost = traj + SOFT_LOST_MAX;
        for _ in 0..4 {
            let (fp, fd) = garbage(&mut r);
            let lost = step(&mut fe, &fp, &fd);
            assert!(
                lost.map.status.contains("lost") || lost.map.status.contains("relocaliz"),
                "expected lost/relocalizing, got: {}",
                lost.map.status
            );
            assert_eq!(lost.map.n_points, n_pts, "map corrupted while lost");
            assert_eq!(
                lost.map.cameras.len(),
                traj_lost,
                "trajectory grew while lost"
            );
        }

        // A frame from a previously-mapped place → relocalize.
        let (fp, fd) = frame(&pts, &ds, &cam, &pose(0.05));
        let reloc = step(&mut fe, &fp, &fd);
        assert!(
            reloc.map.status.contains("relocalized"),
            "did not relocalize: {}",
            reloc.map.status
        );
        assert!(reloc.map.tracking);
        // Map intact; exactly one recovered pose appended on top of
        // the soft-lost predicted poses.
        assert_eq!(
            reloc.map.n_points, n_pts,
            "relocalization corrupted the map"
        );
        assert_eq!(
            reloc.map.cameras.len(),
            traj_lost + 1,
            "recovered pose not appended"
        );

        // Tracking continues normally after recovery.
        let (fp, fd) = frame(&pts, &ds, &cam, &pose(0.12));
        let cont = step(&mut fe, &fp, &fd);
        assert!(
            cont.map.tracking,
            "tracking did not resume: {}",
            cont.map.status
        );
        assert!(cont.map.cameras.len() > traj);
    }

    /// A track lost *before* the BoW vocabulary self-trains (too few
    /// keyframes) can never relocalize — there is no vocabulary to
    /// query. Rather than hang in `Lost` forever, the frontend gives up
    /// after `RELOC_MAX_FRAMES` and re-bootstraps a fresh session.
    #[test]
    fn unrecoverable_loss_rebootstraps_fresh_session() {
        let (pts, ds) = scene(240);
        let cam = cam_model();
        let mut fe = Frontend::new();

        // Initialize, then stop the moment tracking goes live — so few
        // keyframes exist that the vocabulary cannot self-train.
        let mut inited = false;
        for i in 0..16 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.08));
            if step(&mut fe, &fp, &fd).map.tracking {
                inited = true;
                break;
            }
        }
        assert!(inited, "never initialized");
        let (ready, _) = fe.place_stats();
        assert!(!ready, "precondition: vocabulary not yet self-trained");

        // Garbage frames: the track is lost and stays lost — without a
        // vocabulary relocalization is structurally impossible. The
        // frontend must give up and announce a re-initialization.
        let mut r = SmallRng::seed_from_u64(0x0BAD_F00D);
        let mut gave_up = false;
        for _ in 0..(RELOC_MAX_FRAMES + 8) {
            let fd: Vec<brief::Descriptor> = (0..200)
                .map(|_| {
                    let mut d = [0u8; brief::DESC_BYTES];
                    for b in d.iter_mut() {
                        *b = r.random::<u8>();
                    }
                    d
                })
                .collect();
            let fp: Vec<FeaturePoint> = (0..200)
                .map(|i| FeaturePoint {
                    x: (i % W) as u16,
                    y: (i % H) as u16,
                    level: 0,
                    score: 1,
                    angle: 0.0,
                })
                .collect();
            let out = step(&mut fe, &fp, &fd);
            if out.map.status.contains("re-initializing") {
                // The map was discarded — snapshot back to a clean slate.
                assert!(!out.map.tracking, "still 'tracking' after give-up");
                assert!(out.map.cameras.is_empty() && out.map.points.is_empty());
                gave_up = true;
                break;
            }
        }
        assert!(gave_up, "never re-bootstrapped after an unrecoverable loss");

        // A fresh sideways sweep re-initializes from scratch and
        // tracking resumes — the relocalizing deadlock is gone.
        let mut reinit = false;
        for i in 0..16 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.08));
            let out = step(&mut fe, &fp, &fd);
            if out.map.tracking {
                assert!(out.map.n_points > 20, "thin re-init: {}", out.map.n_points);
                reinit = true;
                break;
            }
        }
        assert!(reinit, "did not re-initialize after re-bootstrap");
    }

    /// While `Stage::Lost`, the snapshot must expose a frame-budget the
    /// WebUI can render as a countdown to re-bootstrap, and it must zero
    /// out once the map is discarded and a fresh session begins.
    #[test]
    fn reboot_timer_surfaces_while_lost_and_clears_after_giveup() {
        let (pts, ds) = scene(240);
        let cam = cam_model();
        let mut fe = Frontend::new();

        // Bring tracking live. While Tracking, the timer must be zero.
        let mut inited = false;
        let mut last = None;
        for i in 0..16 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.08));
            let out = step(&mut fe, &fp, &fd);
            last = Some(out.map.clone());
            if out.map.tracking {
                inited = true;
                break;
            }
        }
        assert!(inited, "never initialized");
        let m = last.unwrap();
        assert_eq!(m.lost_frames, 0, "timer should be zero while tracking");
        assert_eq!(m.lost_budget_frames, 0);

        // Garbage frames push past the soft-lost grace into Lost; the
        // timer should be live and racing the budget. Budget matches the
        // no-BoW timeout (vocabulary did not self-train this fast).
        let mut r = SmallRng::seed_from_u64(0xCAFEBABE);
        let mut saw_live_timer = false;
        let mut saw_giveup = false;
        for _ in 0..(RELOC_MAX_FRAMES + 8) {
            let fd: Vec<brief::Descriptor> = (0..200)
                .map(|_| {
                    let mut d = [0u8; brief::DESC_BYTES];
                    for b in d.iter_mut() {
                        *b = r.random::<u8>();
                    }
                    d
                })
                .collect();
            let fp: Vec<FeaturePoint> = (0..200)
                .map(|i| FeaturePoint {
                    x: (i % W) as u16,
                    y: (i % H) as u16,
                    level: 0,
                    score: 1,
                    angle: 0.0,
                })
                .collect();
            let out = step(&mut fe, &fp, &fd);
            if out.map.lost_frames > 0 {
                assert!(
                    out.map.lost_budget_frames >= out.map.lost_frames,
                    "budget {} should not be below elapsed {}",
                    out.map.lost_budget_frames,
                    out.map.lost_frames,
                );
                assert_eq!(
                    out.map.lost_budget_frames, RELOC_MAX_FRAMES as u32,
                    "no-BoW lost should race the short budget",
                );
                saw_live_timer = true;
            }
            if out.map.status.contains("re-initializing") {
                // Once the map is discarded the timer clears.
                assert_eq!(
                    out.map.lost_frames, 0,
                    "timer should clear after re-bootstrap",
                );
                assert_eq!(out.map.lost_budget_frames, 0);
                saw_giveup = true;
                break;
            }
        }
        assert!(saw_live_timer, "lost timer never surfaced while lost");
        assert!(saw_giveup, "never re-bootstrapped");
    }

    /// The conservative loop-closure gates must NOT misfire on ordinary
    /// forward motion (a false loop corrupts the map irreversibly) —
    /// and the wiring runs end-to-end, deterministically, without
    /// corrupting the map. Closure *efficacy* on a genuinely drifted
    /// loop is gated by the slam-core unit tests
    /// (`sim3::optimize_pose_graph` / `sim3_ransac`); the synthetic
    /// harness (frame-stable descriptors + whole-map matching) shares
    /// points across a revisit so it can't manufacture the drift a
    /// closing loop needs (a known synthetic-harness limit, like the
    /// two-view init caveat).
    #[test]
    fn loop_closing_gated_and_no_false_positive() {
        let run = || {
            let (pts, ds) = scene(220);
            let cam = cam_model();
            let mut fe = Frontend::new();
            let mut tail = None;
            for i in 0..64 {
                let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.07));
                tail = Some(step(&mut fe, &fp, &fd));
            }
            let (n_kf, n_pts) = fe.map_stats();
            let t = tail.unwrap();
            (
                n_kf,
                n_pts,
                fe.loop_closures(),
                *t.map.cameras.last().unwrap(),
                t.map.tracking,
                t.map
                    .points
                    .iter()
                    .all(|p| p[0].is_finite() && p[1].is_finite()),
            )
        };
        let a = run();
        // No revisit → the gates must hold (no spurious closure).
        assert_eq!(a.2, 0, "false loop closure on straight motion");
        assert!(a.4, "tracking lost");
        assert!(a.5, "non-finite map geometry");
        assert!(a.0 >= 6 && a.1 > 0, "map too small: {a:?}");
        // Whole pipeline (incl. the loop-closure path) is deterministic.
        let b = run();
        assert_eq!(
            (a.0, a.1, a.2, a.3),
            (b.0, b.1, b.2, b.3),
            "pipeline not deterministic"
        );
    }
}
