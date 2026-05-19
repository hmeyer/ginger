//! M5 local-mapping thread: turn inserted keyframes into a growing,
//! locally bundle-adjusted map.
//!
//! Decoupled from per-frame tracking per the Pi-4 performance
//! strategy — the heavy block-sparse Schur local BA
//! ([`ginger_slam_core::local_ba`]) runs off the tracking core. The pure
//! work unit is [`LocalMapper::process_pending`]; [`super::run`] drives
//! it from a dedicated thread ([`LocalMapper::run_loop`]) while
//! `pipeline_tests` drive the *same* unit synchronously, so the tested
//! seam is identical to production.
//!
//! For each new keyframe the mapper (1) triangulates fresh map points
//! from its still-unmatched features against covisible keyframes
//! ([`ginger_slam_core::triangulation`], gated on parallax / cheirality
//! / reprojection) and (2) runs local BA over its covisibility window
//! ([`ginger_slam_core::map::Map::local_window`]), gauge-fixing the
//! oldest keyframe in the window. The shared map is a single
//! `Arc<Mutex<Map>>`; locks are kept short and never held across the
//! Schur solve's heavy inner work beyond the (bounded, L1-resident)
//! window — coarse-grained but adequate at Pi-class map sizes (a
//! finer-grained tracking/mapping handoff is an M6 refinement).
//!
//! M6-2b: each ingested keyframe is also registered with the shared BoW
//! [`PlaceDb`] (self-trains the vocabulary once enough keyframes exist)
//! so relocalization / loop detection can query it.
//!
//! M6-2d: after local BA, the just-processed keyframe is tested for a
//! **loop closure** — a BoW hit on an old, non-covisible keyframe,
//! geometrically verified by a robust `Sim3` over descriptor-matched
//! map points, then corrected by Essential-graph pose-graph
//! optimization (origin gauge-fixed) with the map points dragged along.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;

use ginger_slam_core::camera::CameraModel;
use ginger_slam_core::local_ba::{LocalBaOptions, local_bundle_adjust};
use ginger_slam_core::map::Map;
use ginger_slam_core::sim3::{
    PgEdge, PoseGraphOptions, Sim3, Sim3RansacOptions, optimize_pose_graph, sim3_ransac,
};
use ginger_slam_core::triangulation::{TriangulateOptions, triangulate};
use log::info;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector2, Vector3};

use super::FeaturePoint;
use super::brief::{self, Descriptor};
use super::place::PlaceDb;

/// `kf` + top-k covisible keyframes pulled into the local-BA window.
const LOCAL_BA_K: usize = 6;
/// Local-BA iterations per keyframe. Deliberately below the slam-core
/// solver default: the background mapper refines the map *incrementally*
/// across many keyframes (overlapping windows re-touch the same poses),
/// so a short budget per keyframe keeps the heavy Schur step off the
/// tracking core's critical path while still converging over time — the
/// "heavy BA at a lower cadence" strategy.
const LOCAL_BA_ITERS: usize = 5;
/// Keep raw features for at most this many recent keyframes (memory
/// bound). Older keyframes keep their map observations (so they still
/// constrain BA) but are no longer matched for *new* point creation.
const RAW_FEAT_KFS: usize = 12;

// ── Loop closing (M6-2d) ─────────────────────────────────────────────
// Conservative: a *false* loop corrupts the whole map irreversibly, so
// every gate favours a miss over a wrong closure.
/// Top BoW candidates considered per keyframe.
const LOOP_MAX_CAND: usize = 3;
/// A candidate must be at least this many keyframes older (a real
/// revisit, not a temporal neighbour).
const LOOP_MIN_GAP: u32 = 15;
/// Minimum BoW L1 similarity for a candidate to be worth verifying.
const LOOP_MIN_SCORE: f64 = 0.045;
/// Candidate + this many covisible neighbours pooled for verification.
const LOOP_COVIS: usize = 5;
/// Min `sim3_ransac` inliers (matched map points) to accept a loop.
const LOOP_MIN_INLIERS: usize = 20;
/// `sim3_ransac` 3D inlier gate (world units; the map is metric-ish up
/// to monocular scale).
const LOOP_SIM3_THRESH: f64 = 0.08;
/// Covisibility edges per keyframe added to the Essential graph.
const LOOP_COVIS_EDGES: usize = 5;
/// Loop-edge weight relative to the unit-weight rigidity edges.
const LOOP_EDGE_WEIGHT: f64 = 10.0;

/// One keyframe handed to the local mapper: its id (already inserted
/// into the shared map by the tracking side), the camera model, the
/// full detected feature set, and which of those features were already
/// bound to an existing map point at insertion (so they are not
/// re-triangulated).
pub struct KeyframeJob {
    pub kf_id: u32,
    pub cam: CameraModel,
    pub pts: Vec<FeaturePoint>,
    pub descs: Vec<Descriptor>,
    pub assigned: Vec<Option<u32>>,
}

struct RawKf {
    cam: CameraModel,
    pts: Vec<FeaturePoint>,
    descs: Vec<Descriptor>,
    /// Per detected feature: the map point it is bound to, if any.
    assigned: Vec<Option<u32>>,
}

/// Owns the raw-feature store + the shared map handle; consumes
/// [`KeyframeJob`]s off the channel.
pub struct LocalMapper {
    map: Arc<Mutex<Map>>,
    jobs: Receiver<KeyframeJob>,
    raw: HashMap<u32, RawKf>,
    recent: VecDeque<u32>,
    /// Shared BoW place-recognition index (M6): every ingested keyframe
    /// is registered so relocalization / loop detection can query it.
    place: Arc<Mutex<PlaceDb>>,
    /// Loop-closure counter, surfaced by the frontend (M6-2d).
    loops: Arc<AtomicU64>,
}

impl LocalMapper {
    pub fn new(
        map: Arc<Mutex<Map>>,
        jobs: Receiver<KeyframeJob>,
        place: Arc<Mutex<PlaceDb>>,
        loops: Arc<AtomicU64>,
    ) -> Self {
        Self {
            map,
            jobs,
            raw: HashMap::new(),
            recent: VecDeque::new(),
            place,
            loops,
        }
    }

    /// Pixel feature → calibrated/normalized image point (undistort,
    /// then `K⁻¹`); same contract as the tracking side's `calib_norm`.
    fn norm(cam: &CameraModel, p: &FeaturePoint) -> Vector2<f64> {
        let ud = cam.undistort_point(&Vector2::new(p.x as f64, p.y as f64));
        Vector2::new((ud.x - cam.cx) / cam.fx, (ud.y - cam.cy) / cam.fy)
    }

    /// Production driver: block on the channel, processing every
    /// keyframe (and draining any that piled up while busy). Returns
    /// when the tracking side drops the sender.
    pub fn run_loop(mut self) {
        info!("slam: local mapper started (triangulation + local BA)");
        while let Ok(job) = self.jobs.recv() {
            self.ingest(job);
            self.process_latest();
            while let Ok(job) = self.jobs.try_recv() {
                self.ingest(job);
                self.process_latest();
            }
        }
        info!("slam: local mapper stopped (tracking side closed)");
    }

    /// Synchronous driver for tests / inline use: ingest + process every
    /// queued keyframe. Returns the number processed.
    pub fn process_pending(&mut self) -> usize {
        let mut n = 0;
        while let Ok(job) = self.jobs.try_recv() {
            self.ingest(job);
            self.process_latest();
            n += 1;
        }
        n
    }

    fn ingest(&mut self, job: KeyframeJob) {
        // Register the keyframe with the BoW place-recognition index
        // (self-trains the vocabulary once enough keyframes are seen).
        if let Ok(mut p) = self.place.lock() {
            p.on_keyframe(job.kf_id, &job.descs);
        }
        self.raw.insert(
            job.kf_id,
            RawKf {
                cam: job.cam,
                pts: job.pts,
                descs: job.descs,
                assigned: job.assigned,
            },
        );
        self.recent.push_back(job.kf_id);
        while self.recent.len() > RAW_FEAT_KFS {
            if let Some(old) = self.recent.pop_front() {
                self.raw.remove(&old);
            }
        }
    }

    /// Triangulate new points for the most-recently-ingested keyframe
    /// against its covisible neighbours that still hold raw features,
    /// then run local BA over its covisibility window.
    fn process_latest(&mut self) {
        let Some(&kf) = self.recent.back() else {
            return;
        };
        let Some(cam) = self.raw.get(&kf).map(|r| r.cam) else {
            return;
        };

        // Short lock: this keyframe's pose + its triangulation
        // neighbours (covisible, plus the spanning-tree parent as a
        // guaranteed parallax source even when covisibility is thin).
        let (pose_kf, neighbours) = {
            let m = self.map.lock().unwrap();
            let Some(k) = m.keyframe(kf) else {
                return;
            };
            let pose = k.pose;
            let parent = k.parent;
            let mut ns: Vec<u32> = m.covisibility(kf).into_iter().map(|(c, _)| c).collect();
            if let Some(p) = parent
                && !ns.contains(&p)
            {
                ns.push(p);
            }
            (pose, ns)
        };

        let mut created = 0usize;
        for c in neighbours {
            if self.raw.contains_key(&c) {
                created += self.triangulate_pair(kf, c, &cam, &pose_kf);
            }
        }

        // Local BA over the covisibility window; gauge-fix the oldest
        // keyframe in it (anchors monocular scale + origin).
        let (window, fixed) = {
            let m = self.map.lock().unwrap();
            let w = m.local_window(kf, LOCAL_BA_K);
            let f: Vec<u32> = w.iter().copied().min().into_iter().collect();
            (w, f)
        };
        if window.len() >= 2 {
            let mut m = self.map.lock().unwrap();
            let rep = local_bundle_adjust(
                &mut m,
                &window,
                &fixed,
                LocalBaOptions {
                    iters: LOCAL_BA_ITERS,
                    ..LocalBaOptions::default()
                },
            );
            drop(m);
            if created > 0 || rep.iters > 0 {
                info!(
                    "slam: local map kf{kf} +{created} pts · BA {cams}c/{pts}p \
                     cost {:.3e}→{:.3e}",
                    rep.cost0,
                    rep.cost1, /**/
                    cams = rep.cameras,
                    pts = rep.points,
                );
            }
        }

        // M6-2d: a revisit detected here closes the loop + pose-graph
        // corrects the map (rare; gated; off the tracking core).
        self.try_close_loop(kf);
    }

    /// Sim3 (scale 1) for a `T_cw` keyframe pose.
    fn iso_to_sim3(t: &Isometry3<f64>) -> Sim3 {
        Sim3::new(1.0, t.rotation.to_rotation_matrix(), t.translation.vector)
    }

    /// `T_cw` for a corrected Sim3 keyframe pose: keep the rotation and
    /// fold the similarity scale into the translation so the camera
    /// centre is preserved in the (metric, SE3) map.
    fn sim3_to_iso(s: &Sim3) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::from(s.t / s.s),
            UnitQuaternion::from_rotation_matrix(&s.r),
        )
    }

    /// Detect + close a loop for keyframe `kf`: BoW candidate (old,
    /// non-covisible) → robust `Sim3` over descriptor-matched map points
    /// → Essential-graph pose-graph optimization → write corrected
    /// keyframe poses + dragged map points back. No-op unless every gate
    /// passes (a false loop is unrecoverable).
    fn try_close_loop(&mut self, kf: u32) {
        let kf_descs = match self.raw.get(&kf) {
            Some(r) => r.descs.clone(),
            None => return,
        };
        let covis: HashSet<u32> = {
            let m = self.map.lock().unwrap();
            m.covisibility(kf).into_iter().map(|(c, _)| c).collect()
        };
        let cand = {
            let p = self.place.lock().unwrap();
            p.query(&kf_descs, LOOP_MAX_CAND, |c| {
                c == kf || covis.contains(&c) || kf.saturating_sub(c) < LOOP_MIN_GAP
            })
            .into_iter()
            .next()
        };
        let Some((c, score)) = cand else { return };
        if score < LOOP_MIN_SCORE {
            return;
        }

        // Pooled observed map points (pos + representative descriptor)
        // for kf alone and for c + its covisible neighbours.
        let gather = |m: &Map, center: u32, with_covis: bool| {
            let mut ids = vec![center];
            if with_covis {
                ids.extend(
                    m.covisibility(center)
                        .into_iter()
                        .take(LOOP_COVIS)
                        .map(|(x, _)| x),
                );
            }
            let mut seen = HashSet::new();
            let (mut ps, mut ds) = (Vec::new(), Vec::new());
            for id in ids {
                if let Some(k) = m.keyframe(id) {
                    for &(pid, _) in &k.obs {
                        if seen.insert(pid)
                            && let Some(p) = m.point(pid)
                        {
                            ps.push(p.pos);
                            ds.push(p.desc);
                        }
                    }
                }
            }
            (ps, ds)
        };
        let (kf_pts, kf_d, c_pts, c_d) = {
            let m = self.map.lock().unwrap();
            let (a, ad) = gather(&m, kf, false);
            let (b, bd) = gather(&m, c, true);
            (a, ad, b, bd)
        };
        if kf_d.len() < LOOP_MIN_INLIERS || c_d.len() < LOOP_MIN_INLIERS {
            return;
        }
        let pairs = brief::match_descriptors(&kf_d, &c_d);
        if pairs.len() < LOOP_MIN_INLIERS {
            return;
        }
        let src: Vec<Vector3<f64>> = pairs.iter().map(|&(i, _)| kf_pts[i as usize]).collect();
        let dst: Vec<Vector3<f64>> = pairs.iter().map(|&(_, j)| c_pts[j as usize]).collect();
        let Some(srep) = sim3_ransac(
            &src,
            &dst,
            Sim3RansacOptions {
                thresh: LOOP_SIM3_THRESH,
                min_inliers: LOOP_MIN_INLIERS,
                ..Sim3RansacOptions::default()
            },
        ) else {
            return;
        };
        let s_loop = srep.pose;

        // Snapshot the keyframe graph (poses as Sim3, parents, covis) +
        // the Essential-graph edges, then optimize off-lock.
        let (old, mut poses, edges, n) = {
            let m = self.map.lock().unwrap();
            let n = m.alive_keyframes().map(|k| k.id).max().unwrap_or(0) as usize + 1;
            let mut poses = vec![Sim3::identity(); n];
            for k in m.alive_keyframes() {
                poses[k.id as usize] = Self::iso_to_sim3(&k.pose);
            }
            let mut edges: Vec<PgEdge> = Vec::new();
            for k in m.alive_keyframes() {
                let i = k.id as usize;
                if let Some(p) = k.parent {
                    edges.push(PgEdge {
                        i,
                        j: p as usize,
                        meas: poses[i].then(&poses[p as usize].inverse()),
                        weight: 1.0,
                    });
                }
                for (q, _) in m.covisibility(k.id).into_iter().take(LOOP_COVIS_EDGES) {
                    edges.push(PgEdge {
                        i,
                        j: q as usize,
                        meas: poses[i].then(&poses[q as usize].inverse()),
                        weight: 1.0,
                    });
                }
            }
            // Loop edge: at the current (drifted) estimate this has a
            // non-zero residual that pulls the graph straight.
            edges.push(PgEdge {
                i: kf as usize,
                j: c as usize,
                meas: poses[kf as usize]
                    .then(&s_loop.inverse())
                    .then(&poses[c as usize].inverse()),
                weight: LOOP_EDGE_WEIGHT,
            });
            (poses.clone(), poses, edges, n)
        };
        optimize_pose_graph(&mut poses, &edges, 0, PoseGraphOptions::default());

        // Write back: corrected keyframe poses + map points dragged by
        // the Sim3 correction of a reference observing keyframe.
        {
            let mut m = self.map.lock().unwrap();
            let live: Vec<u32> = m.alive_keyframes().map(|k| k.id).collect();
            for id in &live {
                m.set_keyframe_pose(*id, Self::sim3_to_iso(&poses[*id as usize]));
            }
            let mut moved: Vec<(u32, Vector3<f64>)> = Vec::new();
            for p in m.alive_points() {
                if let Some(&(rk, _)) = p.obs.iter().min_by_key(|&&(k, _)| k) {
                    let rk = rk as usize;
                    if rk < n {
                        let g = poses[rk].inverse().then(&old[rk]);
                        moved.push((p.id, g.transform(&p.pos)));
                    }
                }
            }
            for (pid, pos) in moved {
                m.set_point_pos(pid, pos);
            }
        }
        self.loops.fetch_add(1, Ordering::Relaxed);
        info!(
            "slam: loop closed kf{kf} ↔ kf{c} — {} sim3 inliers, BoW {score:.3}, {n} kfs",
            srep.n_inliers
        );
    }

    /// Match `kf`'s still-unbound features against `c`'s, triangulate
    /// the geometrically-trustworthy ones into the shared map, and bind
    /// them so they are not re-created. Returns the count added.
    fn triangulate_pair(
        &mut self,
        kf: u32,
        c: u32,
        cam: &CameraModel,
        pose_kf: &Isometry3<f64>,
    ) -> usize {
        let Some(pose_c) = ({
            let m = self.map.lock().unwrap();
            m.keyframe(c).map(|k| k.pose)
        }) else {
            return 0;
        };

        // Small clones (keyframe-rate, ≤ a few hundred 32-byte descs):
        // keeps the map mutex out of the brute-force matcher.
        let (kf_descs, kf_pts) = {
            let r = &self.raw[&kf];
            (r.descs.clone(), r.pts.clone())
        };
        let (c_descs, c_pts) = {
            let r = &self.raw[&c];
            (r.descs.clone(), r.pts.clone())
        };
        let pairs = brief::match_descriptors(&kf_descs, &c_descs);
        let opt = TriangulateOptions::default();

        let mut links: Vec<(usize, usize, u32)> = Vec::new();
        {
            let mut m = self.map.lock().unwrap();
            for (i, j) in pairs {
                let (i, j) = (i as usize, j as usize);
                let bound =
                    self.raw[&kf].assigned[i].is_some() || self.raw[&c].assigned[j].is_some();
                if bound {
                    continue;
                }
                let z_kf = Self::norm(cam, &kf_pts[i]);
                let z_c = Self::norm(cam, &c_pts[j]);
                if let Some(x) = triangulate(pose_kf, &pose_c, z_kf, z_c, opt)
                    && let Some(pid) = m.add_point_observed(x, kf_descs[i], kf, z_kf)
                {
                    m.add_observation(c, pid, z_c);
                    links.push((i, j, pid));
                }
            }
        }

        if let Some(r) = self.raw.get_mut(&kf) {
            for &(i, _, pid) in &links {
                r.assigned[i] = Some(pid);
            }
        }
        if let Some(r) = self.raw.get_mut(&c) {
            for &(_, j, pid) in &links {
                r.assigned[j] = Some(pid);
            }
        }
        links.len()
    }
}
