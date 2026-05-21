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
const INIT_MIN_MATCHES: usize = 80;
const INIT_MIN_DISP_FRAC: f32 = 0.04;
const ANCHOR_RESET_MATCHES: usize = 25;

// Tracking gates: min map-point matches to attempt a pose solve, and
// min reprojection inliers for the refined pose to be trusted.
const TRACK_MIN_MATCHES: usize = 15;
const TRACK_MIN_INLIERS: usize = 10;

/// Relocalization gates. Kept conservative — a false relocalization
/// corrupts the trajectory, so favour staying lost.
const RELOC_MAX_CAND: usize = 5;
const RELOC_COVIS: usize = 5;
const RELOC_MAX_PTS: usize = 1500;
const RELOC_MIN_INLIERS: usize = 15;

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
    pub fn on_frame(
        &mut self,
        points: &[FeaturePoint],
        descs: &[brief::Descriptor],
        width: usize,
        height: usize,
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
                Stage::Tracking(st) => self.track(st, points, descs, cam, fx),
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
        if let Some((apts, adesc)) = anchor.as_ref() {
            let mm = brief::match_descriptors(adesc, descs);
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
                        let (n, kf1) = {
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
                            (n, kf1)
                        };
                        let _ = self.jobs.send(KeyframeJob {
                            kf_id: 0,
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
            self.map.status = "anchor set — translate sideways for parallax".into();
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
            let np = st.trajectory.len();
            let predict = if np >= 2 {
                tracking::constant_velocity(&st.trajectory[np - 2], &st.trajectory[np - 1])
            } else {
                st.trajectory[np - 1]
            };
            let huber = 2.0 / fx;
            let thr = 5.0 / fx;
            match tracking::track_pose(&obs, &predict, huber, thr) {
                Some(rep) if rep.converged && rep.n_inliers >= TRACK_MIN_INLIERS => {
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
                    self.map.status = format!(
                        "tracking lost: weak {}/{} inliers — relocalizing",
                        rep.n_inliers,
                        obs.len()
                    );
                    next = Some(Stage::Lost {
                        since: 0,
                        track: st.clone(),
                    });
                }
                None => {
                    self.map.status = "tracking lost (solve failed) — relocalizing".into();
                    next = Some(Stage::Lost {
                        since: 0,
                        track: st.clone(),
                    });
                }
            }
        } else {
            self.map.status = format!(
                "tracking lost: only {} map matches — relocalizing",
                mm.len()
            );
            next = Some(Stage::Lost {
                since: 0,
                track: st.clone(),
            });
        }
        next
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
        // the candidate keyframes + their covisible
        // neighbours, bounded.
        let (cpos, cdesc) = if cands.is_empty() {
            (Vec::new(), Vec::new())
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
            let mut tr = track.clone();
            tr.trajectory.push(rep.pose);
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
        } else {
            self.map.status = format!("relocalizing… (lost {} frames)", *since);
        }
        next
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
        let out = fe.on_frame(fp, fd, W, H);
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
        let traj = good.map.cameras.len();

        // A frame of pure garbage descriptors → no map matches.
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
        let lost = fe.on_frame(&fp, &fd, W, H);
        assert!(
            lost.map.status.contains("lost"),
            "status: {}",
            lost.map.status
        );
        // Map not corrupted: points unchanged, trajectory not extended.
        assert_eq!(lost.map.n_points, n_pts);
        assert_eq!(lost.map.cameras.len(), traj);
    }

    /// A long enough sweep self-trains the BoW vocabulary and fills the
    /// place-recognition DB; a query then maps a place back to a
    /// keyframe near it (earlier place → earlier keyframe).
    #[test]
    fn place_db_self_trains_and_recognizes_place() {
        let (pts, ds) = scene(220);
        let cam = cam_model();
        let mut fe = Frontend::new();
        for i in 0..64 {
            let (fp, fd) = frame(&pts, &ds, &cam, &pose(i as f64 * 0.07));
            step(&mut fe, &fp, &fd);
        }
        let (ready, indexed) = fe.place_stats();
        assert!(ready, "vocabulary never self-trained");
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
        let mut r = SmallRng::seed_from_u64(0x6105);
        for _ in 0..4 {
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
            let lost = step(&mut fe, &fp, &fd);
            assert!(
                lost.map.status.contains("lost") || lost.map.status.contains("relocaliz"),
                "expected lost/relocalizing, got: {}",
                lost.map.status
            );
            assert_eq!(lost.map.n_points, n_pts, "map corrupted while lost");
            assert_eq!(lost.map.cameras.len(), traj, "trajectory grew while lost");
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
        // Map intact; exactly one recovered pose appended.
        assert_eq!(
            reloc.map.n_points, n_pts,
            "relocalization corrupted the map"
        );
        assert_eq!(
            reloc.map.cameras.len(),
            traj + 1,
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
