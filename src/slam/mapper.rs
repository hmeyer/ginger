//! M5 local-mapping thread: turn inserted keyframes into a growing,
//! locally bundle-adjusted map.
//!
//! Decoupled from per-frame tracking per the PLAN.md Pi-4 performance
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

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use ginger_slam_core::camera::CameraModel;
use ginger_slam_core::local_ba::{LocalBaOptions, local_bundle_adjust};
use ginger_slam_core::map::Map;
use ginger_slam_core::triangulation::{TriangulateOptions, triangulate};
use log::info;
use nalgebra::{Isometry3, Vector2};

use super::FeaturePoint;
use super::brief::{self, Descriptor};

/// `kf` + top-k covisible keyframes pulled into the local-BA window.
const LOCAL_BA_K: usize = 6;
/// Local-BA iterations per keyframe. Deliberately below the slam-core
/// solver default: the background mapper refines the map *incrementally*
/// across many keyframes (overlapping windows re-touch the same poses),
/// so a short budget per keyframe keeps the heavy Schur step off the
/// tracking core's critical path while still converging over time — the
/// PLAN.md "heavy BA at a lower cadence" strategy.
const LOCAL_BA_ITERS: usize = 5;
/// Keep raw features for at most this many recent keyframes (memory
/// bound). Older keyframes keep their map observations (so they still
/// constrain BA) but are no longer matched for *new* point creation.
const RAW_FEAT_KFS: usize = 12;

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
}

impl LocalMapper {
    pub fn new(map: Arc<Mutex<Map>>, jobs: Receiver<KeyframeJob>) -> Self {
        Self {
            map,
            jobs,
            raw: HashMap::new(),
            recent: VecDeque::new(),
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
                    rep.cost0, rep.cost1, /**/
                    cams = rep.cameras,
                    pts = rep.points,
                );
            }
        }
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
                let bound = self.raw[&kf].assigned[i].is_some()
                    || self.raw[&c].assigned[j].is_some();
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
