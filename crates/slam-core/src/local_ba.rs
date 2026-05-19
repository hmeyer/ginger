//! Local bundle adjustment (M5-1b): jointly refine a window of keyframe
//! poses + the map points they observe to minimise robust reprojection
//! error.
//!
//! This is the Pi-4 performance crux, so the structure is the planned
//! **block-sparse Schur complement**: 6×6 camera blocks, 3×3
//! point blocks, 6×3 coupling. Points are eliminated per-block (3×3
//! inverse) into a small dense reduced camera system (≤ `6·|window|`,
//! L1-resident) solved by Cholesky, then back-substituted. LM damping +
//! Huber, accept/reject like the dense solver. Per-observation 2×6 / 2×3
//! Jacobians are central finite differences — convention-proof; an
//! analytic Jacobian is a later, parity-tested perf step.
//!
//! Camera-free; calibrated/normalized observations (same contract as
//! `tracking`/`twoview`). Gauge: keyframes in `fixed` (and any window
//! keyframe not listed in `window`) are held constant — the caller must
//! fix ≥1 keyframe or the problem is rank-deficient.

use std::collections::HashMap;

use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix2x3, Matrix2x6, Matrix3, Matrix6, Matrix6x3, Vector2,
    Vector3, Vector6,
};

use crate::lie::se3_exp;
use crate::map::Map;

#[derive(Clone, Copy, Debug)]
pub struct LocalBaOptions {
    pub iters: usize,
    /// Huber threshold in calibrated units (≈ px / focal); `0` = plain.
    pub huber: f64,
    pub lambda0: f64,
}

impl Default for LocalBaOptions {
    fn default() -> Self {
        Self {
            iters: 12,
            huber: 0.01,
            lambda0: 1e-3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocalBaReport {
    pub cost0: f64,
    pub cost1: f64,
    pub iters: usize,
    pub cameras: usize,
    pub points: usize,
}

fn project(tcw: &Isometry3<f64>, x: &Vector3<f64>) -> Vector2<f64> {
    let pc = tcw.rotation * x + tcw.translation.vector;
    let z = if pc.z > 1e-4 { pc.z } else { 1e-4 };
    Vector2::new(pc.x / z, pc.y / z)
}

struct Ob {
    /// Index into the window-camera pose array.
    cam: usize,
    /// Index into the optimized-camera array, if this camera is free.
    opt: Option<usize>,
    pt: usize,
    z: Vector2<f64>,
}

/// Refine `window` keyframes + their points in place. `fixed` keyframes
/// (and any not in `window`) stay constant for gauge.
pub fn local_bundle_adjust(
    map: &mut Map,
    window: &[u32],
    fixed: &[u32],
    opt: LocalBaOptions,
) -> LocalBaReport {
    // Window cameras (alive), deterministic order.
    let mut win: Vec<u32> = window
        .iter()
        .copied()
        .filter(|k| map.keyframe(*k).is_some())
        .collect();
    win.sort_unstable();
    win.dedup();
    let cam_pos: HashMap<u32, usize> = win.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    // Optimized = window minus fixed; opt index per window camera.
    let mut opt_of: Vec<Option<usize>> = vec![None; win.len()];
    let mut opt_kf: Vec<u32> = Vec::new();
    for (wi, &k) in win.iter().enumerate() {
        if !fixed.contains(&k) {
            opt_of[wi] = Some(opt_kf.len());
            opt_kf.push(k);
        }
    }
    // Points: alive, observed by ≥1 optimized camera; deterministic.
    let mut pt_ids: Vec<u32> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &k in &opt_kf {
            for &(pid, _) in &map.keyframe(k).unwrap().obs {
                if map.point(pid).is_some() && seen.insert(pid) {
                    pt_ids.push(pid);
                }
            }
        }
    }
    pt_ids.sort_unstable();
    let pt_pos: HashMap<u32, usize> = pt_ids.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let (nc, np) = (opt_kf.len(), pt_ids.len());
    if nc == 0 || np == 0 {
        return LocalBaReport {
            cost0: 0.0,
            cost1: 0.0,
            iters: 0,
            cameras: nc,
            points: np,
        };
    }

    // Poses for *all* window cameras (fixed ones stay constant); points.
    let mut pose: Vec<Isometry3<f64>> =
        win.iter().map(|&k| map.keyframe(k).unwrap().pose).collect();
    let mut pts: Vec<Vector3<f64>> = pt_ids.iter().map(|&p| map.point(p).unwrap().pos).collect();

    // Observations (deterministic: window order, then keyframe obs order).
    let mut obs: Vec<Ob> = Vec::new();
    for (wi, &k) in win.iter().enumerate() {
        for &(pid, z) in &map.keyframe(k).unwrap().obs {
            if let Some(&pt) = pt_pos.get(&pid) {
                obs.push(Ob {
                    cam: wi,
                    opt: opt_of[wi],
                    pt,
                    z,
                });
            }
        }
    }

    let hub = opt.huber;
    let rho = |e: f64| -> f64 {
        if hub > 0.0 && e > hub {
            hub * (e - 0.5 * hub)
        } else {
            0.5 * e * e
        }
    };
    let sqrt_w = |e: f64| -> f64 {
        if hub > 0.0 && e > hub {
            (hub / e).sqrt()
        } else {
            1.0
        }
    };
    let total_cost = |pose: &[Isometry3<f64>], pts: &[Vector3<f64>]| -> f64 {
        obs.iter()
            .map(|o| rho((project(&pose[o.cam], &pts[o.pt]) - o.z).norm()))
            .sum()
    };

    let cost0 = total_cost(&pose, &pts);
    let mut lambda = opt.lambda0;
    let mut iters_done = 0;
    let eps = 1e-6;

    for _ in 0..opt.iters {
        iters_done += 1;
        let mut hcc = vec![Matrix6::<f64>::zeros(); nc];
        let mut gc = vec![Vector6::<f64>::zeros(); nc];
        let mut hpp = vec![Matrix3::<f64>::zeros(); np];
        let mut gp = vec![Vector3::<f64>::zeros(); np];
        // Per point: the (optimized camera, Hcp 6×3) blocks coupling it.
        let mut by_pt: Vec<Vec<(usize, Matrix6x3<f64>)>> = vec![Vec::new(); np];

        for o in &obs {
            let tcw = pose[o.cam];
            let x = pts[o.pt];
            let r = project(&tcw, &x) - o.z;
            let w = sqrt_w(r.norm());
            let rw = r * w;

            let mut jx = Matrix2x3::<f64>::zeros();
            for d in 0..3 {
                let (mut xp, mut xm) = (x, x);
                xp[d] += eps;
                xm[d] -= eps;
                let col =
                    ((project(&tcw, &xp) - o.z) * w - (project(&tcw, &xm) - o.z) * w) / (2.0 * eps);
                jx[(0, d)] = col.x;
                jx[(1, d)] = col.y;
            }
            hpp[o.pt] += jx.transpose() * jx;
            gp[o.pt] += jx.transpose() * rw;

            if let Some(ci) = o.opt {
                let mut jc = Matrix2x6::<f64>::zeros();
                for d in 0..6 {
                    let (mut dp, mut dm) = (Vector6::zeros(), Vector6::zeros());
                    dp[d] = eps;
                    dm[d] = -eps;
                    let rp = (project(&(se3_exp(&dp) * tcw), &x) - o.z) * w;
                    let rm = (project(&(se3_exp(&dm) * tcw), &x) - o.z) * w;
                    let col = (rp - rm) / (2.0 * eps);
                    jc[(0, d)] = col.x;
                    jc[(1, d)] = col.y;
                }
                hcc[ci] += jc.transpose() * jc;
                gc[ci] += jc.transpose() * rw;
                by_pt[o.pt].push((ci, jc.transpose() * jx));
            }
        }

        // LM damping on the block diagonals.
        for h in hcc.iter_mut() {
            for d in 0..6 {
                h[(d, d)] += lambda * h[(d, d)].max(1e-9);
            }
        }
        for h in hpp.iter_mut() {
            for d in 0..3 {
                h[(d, d)] += lambda * h[(d, d)].max(1e-9);
            }
        }
        let hpp_inv: Vec<Matrix3<f64>> = hpp
            .iter()
            .map(|m| m.try_inverse().unwrap_or_else(|| Matrix3::identity() * 1e6))
            .collect();

        // Reduced camera system  S Δc = v  (Schur, points eliminated).
        let dim = 6 * nc;
        let mut s = DMatrix::<f64>::zeros(dim, dim);
        let mut v = DVector::<f64>::zeros(dim);
        for i in 0..nc {
            for a in 0..6 {
                v[6 * i + a] = -gc[i][a];
                for b in 0..6 {
                    s[(6 * i + a, 6 * i + b)] = hcc[i][(a, b)];
                }
            }
        }
        for j in 0..np {
            let hinv = &hpp_inv[j];
            let hg = hinv * gp[j];
            for &(ca, ha) in &by_pt[j] {
                let vc = ha * hg;
                for a in 0..6 {
                    v[6 * ca + a] += vc[a];
                }
                for &(cb, hb) in &by_pt[j] {
                    let m = ha * hinv * hb.transpose();
                    for a in 0..6 {
                        for b in 0..6 {
                            s[(6 * ca + a, 6 * cb + b)] -= m[(a, b)];
                        }
                    }
                }
            }
        }

        let Some(chol) = s.clone().cholesky() else {
            lambda *= 10.0;
            continue;
        };
        let dc = chol.solve(&v);

        // Back-substitute points: Δp = Hpp⁻¹(−gp − Hcpᵀ Δc).
        let mut dp = vec![Vector3::<f64>::zeros(); np];
        for j in 0..np {
            let mut rhs = -gp[j];
            for &(ci, hcpm) in &by_pt[j] {
                let dci = Vector6::from_fn(|a, _| dc[6 * ci + a]);
                rhs -= hcpm.transpose() * dci;
            }
            dp[j] = hpp_inv[j] * rhs;
        }

        // Tentative step; accept iff the robust cost drops (LM).
        let mut pose_try = pose.clone();
        for (wi, &k) in win.iter().enumerate() {
            if let Some(ci) = opt_of[wi] {
                let _ = k;
                let d = Vector6::from_fn(|a, _| dc[6 * ci + a]);
                pose_try[wi] = se3_exp(&d) * pose[wi];
            }
        }
        let pts_try: Vec<Vector3<f64>> = pts.iter().zip(&dp).map(|(p, d)| p + d).collect();

        if total_cost(&pose_try, &pts_try) < total_cost(&pose, &pts) {
            pose = pose_try;
            pts = pts_try;
            lambda = (lambda * 0.3).max(1e-12);
        } else {
            lambda *= 10.0;
        }
    }

    // Write back optimized cameras + all points.
    for (wi, &k) in win.iter().enumerate() {
        if opt_of[wi].is_some() {
            map.set_keyframe_pose(k, pose[wi]);
        }
    }
    for (j, &pid) in pt_ids.iter().enumerate() {
        map.set_point_pos(pid, pts[j]);
    }
    let _ = cam_pos;

    let cost1 = total_cost(&pose, &pts);
    LocalBaReport {
        cost0,
        cost1,
        iters: iters_done,
        cameras: nc,
        points: np,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn pose(tx: f64, ry: f64) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(tx, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, ry, 0.0),
        )
    }
    fn proj(tcw: &Isometry3<f64>, x: &Vector3<f64>) -> Vector2<f64> {
        let pc = tcw.rotation * x + tcw.translation.vector;
        Vector2::new(pc.x / pc.z, pc.y / pc.z)
    }

    /// `(map, truth poses, truth points, keyframe ids)`.
    type Scene = (Map, Vec<Isometry3<f64>>, Vec<Vector3<f64>>, Vec<u32>);

    /// Ground-truth scene: 4 cameras along x with slight yaw, N points;
    /// cam0 fixed at truth (gauge).
    fn scene(noise: f64) -> Scene {
        let mut r = Rng(0xBEEF);
        let truth_pose: Vec<Isometry3<f64>> = (0..4)
            .map(|i| pose(i as f64 * 0.3, i as f64 * 0.02))
            .collect();
        let truth_pts: Vec<Vector3<f64>> = (0..60)
            .map(|_| Vector3::new((r.f() - 0.5) * 4.0, (r.f() - 0.5) * 3.0, 4.0 + r.f() * 4.0))
            .collect();

        let mut m = Map::new();
        // Perturb everything except cam0 (kept at truth → gauge).
        let mut kf_ids = Vec::new();
        for (i, tp) in truth_pose.iter().enumerate() {
            let p = if i == 0 {
                *tp
            } else {
                pose(
                    i as f64 * 0.3 + (r.f() - 0.5) * 0.1,
                    i as f64 * 0.02 + (r.f() - 0.5) * 0.05,
                )
            };
            let obs: Vec<(u32, Vector2<f64>)> = if i == 0 {
                Vec::new()
            } else {
                truth_pts
                    .iter()
                    .enumerate()
                    .map(|(j, x)| {
                        let mut z = proj(tp, x);
                        z.x += (r.f() - 0.5) * 2.0 * noise;
                        z.y += (r.f() - 0.5) * 2.0 * noise;
                        (j as u32, z)
                    })
                    .collect()
            };
            kf_ids.push(m.add_keyframe(p, obs));
        }
        // Create points off-truth (so BA must move them), observed by kf0.
        let mut kf0_obs = Vec::new();
        for (j, tx) in truth_pts.iter().enumerate() {
            let perturbed = tx + Vector3::new(r.f() - 0.5, r.f() - 0.5, r.f() - 0.5) * 0.3;
            let pid = m.add_point(perturbed, [0u8; 32], kf_ids[0], j as u32);
            let mut z = proj(&truth_pose[0], tx);
            z.x += (r.f() - 0.5) * 2.0 * noise;
            z.y += (r.f() - 0.5) * 2.0 * noise;
            kf0_obs.push((pid, z));
        }
        // Attach kf0's observations (kf0 fixed but constrains points).
        for &(pid, z) in &kf0_obs {
            m.add_observation(kf_ids[0], pid, z);
        }
        (m, truth_pose, truth_pts, kf_ids)
    }

    #[test]
    fn local_ba_recovers_poses_and_points_noise_free() {
        let (mut m, tp, tx, kf) = scene(0.0);
        let win = vec![kf[0], kf[1], kf[2], kf[3]];
        let rep = local_bundle_adjust(&mut m, &win, &[kf[0]], LocalBaOptions::default());
        assert!(
            rep.cost1 < rep.cost0 * 1e-4,
            "cost {} -> {}",
            rep.cost0,
            rep.cost1
        );
        // Optimized cameras recovered.
        for i in 1..4 {
            let est = m.keyframe(kf[i]).unwrap().pose;
            let d = crate::lie::se3_log(&(est.inverse() * tp[i])).norm();
            assert!(d < 1e-3, "cam{i} Δ={d}");
        }
        // Points recovered. (Absolute 3D tolerance is geometry-bound:
        // monocular small-baseline depth is weakly observable, so a
        // ~1e-4 reprojection cost still permits ~1e-2 depth error —
        // still a >15× cut from the 0.3-magnitude perturbation.)
        let max = tx
            .iter()
            .enumerate()
            .map(|(j, x)| (m.point(j as u32).unwrap().pos - x).norm())
            .fold(0.0, f64::max);
        assert!(max < 1.5e-2, "max point err={max}");
    }

    #[test]
    fn local_ba_reduces_cost_under_noise() {
        let (mut m, tp, _tx, kf) = scene(0.002);
        let win = vec![kf[0], kf[1], kf[2], kf[3]];
        let rep = local_bundle_adjust(&mut m, &win, &[kf[0]], LocalBaOptions::default());
        assert!(
            rep.cost1 < 0.2 * rep.cost0,
            "cost {} -> {}",
            rep.cost0,
            rep.cost1
        );
        for i in 1..4 {
            let est = m.keyframe(kf[i]).unwrap().pose;
            let d = crate::lie::se3_log(&(est.inverse() * tp[i])).norm();
            assert!(d < 0.05, "cam{i} Δ={d}");
        }
    }

    #[test]
    fn empty_window_is_noop() {
        let mut m = Map::new();
        let r = local_bundle_adjust(&mut m, &[], &[], LocalBaOptions::default());
        assert_eq!((r.cameras, r.points, r.iters), (0, 0, 0));
    }
}
