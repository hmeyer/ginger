//! Map structure for local mapping (M5): keyframes, map points, the
//! covisibility graph + spanning tree, the keyframe-insertion policy,
//! the local-BA window selector, and culling.
//!
//! Camera-free (nalgebra only). Ids are stable `u32` indices into
//! tombstone-able `Vec`s so culling never reindexes live entities.
//! Descriptors are opaque 32-byte blobs — the Hamming matcher lives in
//! the frontend; the map only stores a representative descriptor so a
//! point can be re-found.

use std::collections::HashMap;

use nalgebra::{Isometry3, Vector2, Vector3};

/// Opaque 256-bit descriptor (ORB/BRIEF); the map never compares these,
/// it just carries one per point for the frontend to re-match.
pub type Descriptor = [u8; 32];

/// A triangulated world point, plus the keyframes that observe it.
#[derive(Clone, Debug)]
pub struct MapPoint {
    pub id: u32,
    pub pos: Vector3<f64>,
    pub desc: Descriptor,
    /// `(keyframe id, keypoint index within that keyframe)`.
    pub obs: Vec<(u32, u32)>,
    pub alive: bool,
}

/// A keyframe: its `T_cw` pose and the map points it observes (with the
/// calibrated/normalized image coordinate of each observation).
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub id: u32,
    pub pose: Isometry3<f64>,
    /// `(map point id, normalized image observation)`.
    pub obs: Vec<(u32, Vector2<f64>)>,
    /// Spanning-tree parent (max-covisibility keyframe at insertion).
    pub parent: Option<u32>,
    pub alive: bool,
}

/// Keyframes + map points with the covisibility graph derived on demand.
#[derive(Clone, Debug, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    points: Vec<MapPoint>,
}

impl Map {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a map point first seen by `kf_id` at keypoint `kp_idx`.
    pub fn add_point(
        &mut self,
        pos: Vector3<f64>,
        desc: Descriptor,
        kf_id: u32,
        kp_idx: u32,
    ) -> u32 {
        let id = self.points.len() as u32;
        self.points.push(MapPoint {
            id,
            pos,
            desc,
            obs: vec![(kf_id, kp_idx)],
            alive: true,
        });
        id
    }

    /// Insert a keyframe observing `obs = [(map_point_id, norm_xy)]`.
    /// Links the observations back into the points, then sets the
    /// spanning-tree parent to the most-covisible existing keyframe.
    pub fn add_keyframe(&mut self, pose: Isometry3<f64>, obs: Vec<(u32, Vector2<f64>)>) -> u32 {
        let id = self.keyframes.len() as u32;
        for (kp_idx, &(pid, _)) in obs.iter().enumerate() {
            if let Some(p) = self.points.get_mut(pid as usize)
                && p.alive
            {
                p.obs.push((id, kp_idx as u32));
            }
        }
        self.keyframes.push(Keyframe {
            id,
            pose,
            obs,
            parent: None,
            alive: true,
        });
        let parent = self
            .covisibility(id)
            .first()
            .filter(|&&(_, w)| w > 0)
            .map(|&(kf, _)| kf);
        self.keyframes[id as usize].parent = parent;
        id
    }

    /// Overwrite a live keyframe's pose (local-BA write-back).
    pub fn set_keyframe_pose(&mut self, id: u32, pose: Isometry3<f64>) {
        if let Some(k) = self.keyframes.get_mut(id as usize)
            && k.alive
        {
            k.pose = pose;
        }
    }

    /// Overwrite a live map point's position (local-BA write-back).
    pub fn set_point_pos(&mut self, id: u32, pos: Vector3<f64>) {
        if let Some(p) = self.points.get_mut(id as usize)
            && p.alive
        {
            p.pos = pos;
        }
    }

    /// Record that live keyframe `kf_id` observes live point `pid` at
    /// normalized image coord `z` (links both directions). No-op if
    /// either is dead.
    pub fn add_observation(&mut self, kf_id: u32, pid: u32, z: Vector2<f64>) {
        let ok = self.keyframe(kf_id).is_some() && self.point(pid).is_some();
        if !ok {
            return;
        }
        let kf = &mut self.keyframes[kf_id as usize];
        let kp_idx = kf.obs.len() as u32;
        kf.obs.push((pid, z));
        self.points[pid as usize].obs.push((kf_id, kp_idx));
    }

    pub fn keyframe(&self, id: u32) -> Option<&Keyframe> {
        self.keyframes.get(id as usize).filter(|k| k.alive)
    }

    pub fn point(&self, id: u32) -> Option<&MapPoint> {
        self.points.get(id as usize).filter(|p| p.alive)
    }

    pub fn alive_keyframes(&self) -> impl Iterator<Item = &Keyframe> {
        self.keyframes.iter().filter(|k| k.alive)
    }

    pub fn alive_points(&self) -> impl Iterator<Item = &MapPoint> {
        self.points.iter().filter(|p| p.alive)
    }

    pub fn n_keyframes(&self) -> usize {
        self.keyframes.iter().filter(|k| k.alive).count()
    }

    pub fn n_points(&self) -> usize {
        self.points.iter().filter(|p| p.alive).count()
    }

    /// Covisible keyframes of `kf_id`: `(other_kf, shared_point_count)`,
    /// strongest first, alive only.
    pub fn covisibility(&self, kf_id: u32) -> Vec<(u32, usize)> {
        let Some(kf) = self.keyframe(kf_id) else {
            return Vec::new();
        };
        let mut w: HashMap<u32, usize> = HashMap::new();
        for &(pid, _) in &kf.obs {
            let Some(p) = self.point(pid) else { continue };
            for &(other, _) in &p.obs {
                if other != kf_id && self.keyframes[other as usize].alive {
                    *w.entry(other).or_insert(0) += 1;
                }
            }
        }
        let mut v: Vec<(u32, usize)> = w.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    /// Covisible keyframes sharing at least `min_w` points.
    pub fn connected(&self, kf_id: u32, min_w: usize) -> Vec<u32> {
        self.covisibility(kf_id)
            .into_iter()
            .filter(|&(_, w)| w >= min_w)
            .map(|(k, _)| k)
            .collect()
    }

    /// Local-BA window: `kf_id` plus its top-`k` covisible keyframes.
    pub fn local_window(&self, kf_id: u32, k: usize) -> Vec<u32> {
        let mut out = vec![kf_id];
        out.extend(self.covisibility(kf_id).into_iter().take(k).map(|(c, _)| c));
        out
    }

    /// Tombstone a map point and unlink it from its keyframes.
    pub fn cull_point(&mut self, id: u32) {
        if let Some(p) = self.points.get_mut(id as usize) {
            p.alive = false;
            p.obs.clear();
        }
        for kf in &mut self.keyframes {
            kf.obs.retain(|&(pid, _)| pid != id);
        }
    }

    /// Tombstone a keyframe and unlink it from its points + children
    /// (orphaned children are re-parented to this keyframe's parent).
    pub fn cull_keyframe(&mut self, id: u32) {
        let parent = self.keyframes.get(id as usize).and_then(|k| k.parent);
        if let Some(k) = self.keyframes.get_mut(id as usize) {
            k.alive = false;
            k.obs.clear();
        }
        for p in &mut self.points {
            p.obs.retain(|&(kf, _)| kf != id);
        }
        for k in &mut self.keyframes {
            if k.parent == Some(id) {
                k.parent = parent;
            }
        }
    }
}

/// Keyframe-insertion policy (ORB-SLAM-style, pure): insert when
/// tracking is healthy *and* the current frame either has drifted to
/// few tracked points relative to the reference keyframe, or enough
/// frames have passed since the last keyframe.
pub fn needs_keyframe(
    tracked_inliers: usize,
    ref_kf_points: usize,
    frames_since_last_kf: usize,
    min_tracked: usize,
) -> bool {
    if tracked_inliers < min_tracked {
        return false; // tracking too weak — don't pollute the map
    }
    let thin = ref_kf_points > 0 && tracked_inliers < (ref_kf_points * 9) / 10;
    let stale = frames_since_last_kf >= 20;
    thin || stale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso(tx: f64) -> Isometry3<f64> {
        Isometry3::translation(tx, 0.0, 0.0)
    }
    fn v2(x: f64) -> Vector2<f64> {
        Vector2::new(x, 0.0)
    }

    /// Build a 4-keyframe chain. Points: p0,p1 seen by kf0&kf1&kf2;
    /// p2 by kf1&kf2; p3 by kf2&kf3 only.
    fn sample() -> Map {
        let mut m = Map::new();
        // kf0 creates p0, p1.
        let kf0 = m.add_keyframe(iso(0.0), vec![]);
        let p0 = m.add_point(Vector3::new(0.0, 0.0, 5.0), [1; 32], kf0, 0);
        let p1 = m.add_point(Vector3::new(1.0, 0.0, 5.0), [2; 32], kf0, 1);
        m.keyframes[kf0 as usize].obs = vec![(p0, v2(0.0)), (p1, v2(0.1))];
        m.points[p0 as usize].obs = vec![(kf0, 0)];
        m.points[p1 as usize].obs = vec![(kf0, 1)];
        // kf1 sees p0,p1 and creates p2.
        let kf1 = m.add_keyframe(iso(0.2), vec![(p0, v2(0.0)), (p1, v2(0.1))]);
        let p2 = m.add_point(Vector3::new(2.0, 0.0, 5.0), [3; 32], kf1, 2);
        m.keyframes[kf1 as usize].obs.push((p2, v2(0.2)));
        m.points[p2 as usize].obs = vec![(kf1, 2)];
        // kf2 sees p0,p1,p2 and creates p3.
        let kf2 = m.add_keyframe(iso(0.4), vec![(p0, v2(0.0)), (p1, v2(0.1)), (p2, v2(0.2))]);
        let p3 = m.add_point(Vector3::new(3.0, 0.0, 5.0), [4; 32], kf2, 3);
        m.keyframes[kf2 as usize].obs.push((p3, v2(0.3)));
        m.points[p3 as usize].obs = vec![(kf2, 3)];
        // kf3 sees only p3.
        m.add_keyframe(iso(0.6), vec![(p3, v2(0.3))]);
        m
    }

    #[test]
    fn covisibility_weights_and_order() {
        let m = sample();
        // kf2 shares p0,p1,p2 with kf1 (3) and p0,p1 with kf0 (2),
        // p3 with kf3 (1).
        let cov = m.covisibility(2);
        assert_eq!(cov, vec![(1, 3), (0, 2), (3, 1)]);
        // connected at min weight 2 drops kf3.
        assert_eq!(m.connected(2, 2), vec![1, 0]);
    }

    #[test]
    fn spanning_tree_parent_is_max_covisible() {
        let m = sample();
        assert_eq!(m.keyframe(0).unwrap().parent, None); // first KF
        assert_eq!(m.keyframe(1).unwrap().parent, Some(0));
        assert_eq!(m.keyframe(2).unwrap().parent, Some(1)); // shares most with kf1
        assert_eq!(m.keyframe(3).unwrap().parent, Some(2));
    }

    #[test]
    fn local_window_is_kf_plus_top_k_covisible() {
        let m = sample();
        assert_eq!(m.local_window(2, 2), vec![2, 1, 0]);
        assert_eq!(m.local_window(2, 1), vec![2, 1]);
    }

    #[test]
    fn culling_unlinks_and_reparents() {
        let mut m = sample();
        m.cull_point(0);
        assert!(m.point(0).is_none());
        assert!(m.keyframe(2).unwrap().obs.iter().all(|&(pid, _)| pid != 0));
        // kf2 still covisible with kf1 via p1,p2 (weight 2 now).
        assert_eq!(
            m.covisibility(2).iter().find(|&&(k, _)| k == 1),
            Some(&(1, 2))
        );

        m.cull_keyframe(1);
        assert!(m.keyframe(1).is_none());
        assert_eq!(m.n_keyframes(), 3);
        // kf2's parent (was kf1) re-parented to kf1's parent (kf0).
        assert_eq!(m.keyframe(2).unwrap().parent, Some(0));
        assert!(m.point(1).unwrap().obs.iter().all(|&(kf, _)| kf != 1));
    }

    #[test]
    fn keyframe_policy() {
        // Healthy + thinned vs reference → insert.
        assert!(needs_keyframe(60, 200, 3, 30));
        // Healthy but still dense + recent → no.
        assert!(!needs_keyframe(190, 200, 3, 30));
        // Stale (≥20 frames) → insert even if dense.
        assert!(needs_keyframe(190, 200, 20, 30));
        // Tracking too weak → never (don't pollute the map).
        assert!(!needs_keyframe(10, 200, 50, 30));
    }
}
