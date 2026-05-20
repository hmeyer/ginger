//! P3P + RANSAC pose recovery (M6-1b): the camera pose `T_cw` from
//! 3D↔2D correspondences, robust to a high outlier ratio.
//!
//! This is the recovery solver M6 relocalization is built on: when
//! tracking is lost, BoW ([`crate::bow`]) shortlists candidate
//! keyframes and descriptor matching yields map-point ↔ feature pairs —
//! many of them wrong. P3P is the minimal solver (3 correspondences →
//! up to 4 poses), wrapped in RANSAC so the inlier set drives a final
//! pose; the same primitive verifies loop candidates later.
//!
//! Camera-free; calibrated/normalized observations and `T_cw` (world →
//! camera, `x_cam = R·X + t`) — the same contract as
//! [`crate::tracking`] / [`crate::twoview`] / [`crate::triangulation`].
//! It reuses [`Observation`] as the correspondence type and polishes the
//! RANSAC estimate with the tested motion-only BA
//! ([`crate::tracking::track_pose`]).
//!
//! The P3P core is Grunert's reduction (law of cosines → a quartic in
//! the distance ratio), with the quartic coefficients derived *in code*
//! by polynomial elimination rather than transcribed magic constants,
//! and its real roots taken as companion-matrix eigenvalues. Correctness
//! is pinned by the exact-recovery unit test.

use nalgebra::{Complex, DMatrix, Isometry3, Matrix3, Rotation3, Translation3, Vector2, Vector3};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::tracking::{Observation, track_pose};

/// RANSAC configuration.
#[derive(Clone, Copy, Debug)]
pub struct PnpOptions {
    pub iters: usize,
    /// Inlier reprojection-error gate, calibrated units (≈ px / focal).
    pub thresh: f64,
    /// Minimum inliers for the refined pose to be trusted.
    pub min_inliers: usize,
    /// Deterministic sampler seed (headless-determinism gate).
    pub seed: u64,
}

impl Default for PnpOptions {
    fn default() -> Self {
        Self {
            iters: 300,
            thresh: 6.0 / 1000.0, // ~3 px at f ≈ 500
            min_inliers: 12,
            seed: 0x5EED_2718,
        }
    }
}

/// Outcome of [`pnp_ransac`].
#[derive(Clone, Debug)]
pub struct PnpReport {
    /// Refined `T_cw` (world → camera).
    pub pose: Isometry3<f64>,
    /// Per-correspondence inlier mask under the final pose.
    pub inliers: Vec<bool>,
    pub n_inliers: usize,
}

/// Convolve two coefficient vectors (ascending powers).
fn conv(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            c[i + j] += ai * bj;
        }
    }
    c
}

/// Real roots of a polynomial (ascending coeffs) via the companion
/// matrix's eigenvalues; near-zero leading terms are trimmed so a
/// degenerate quartic falls back to its true degree.
fn real_roots(coeffs: &[f64]) -> Vec<f64> {
    let mut c = coeffs.to_vec();
    while c.len() > 1 && c.last().unwrap().abs() < 1e-12 {
        c.pop();
    }
    let d = c.len() - 1;
    if d == 0 {
        return Vec::new();
    }
    if d == 1 {
        return vec![-c[0] / c[1]];
    }
    // Monic companion (d×d); eigenvalues = roots.
    let lead = c[d];
    let mon: Vec<f64> = c.iter().map(|x| x / lead).collect();
    let mut comp = DMatrix::<f64>::zeros(d, d);
    for i in 0..d {
        comp[(i, d - 1)] = -mon[i];
        if i + 1 < d {
            comp[(i + 1, i)] = 1.0;
        }
    }
    let ev: Vec<Complex<f64>> = comp.complex_eigenvalues().iter().copied().collect();
    ev.into_iter()
        .filter(|z| z.im.abs() <= 1e-7 * (1.0 + z.re.abs()))
        .map(|z| z.re)
        .collect()
}

/// Best rigid `R, t` mapping `src[i] → dst[i]` (Kabsch/Umeyama, with
/// reflection fix). `None` if the points are (near-)collinear.
fn kabsch(src: &[Vector3<f64>; 3], dst: &[Vector3<f64>; 3]) -> Option<Isometry3<f64>> {
    let sc = (src[0] + src[1] + src[2]) / 3.0;
    let dc = (dst[0] + dst[1] + dst[2]) / 3.0;
    let mut h = Matrix3::zeros();
    for i in 0..3 {
        h += (src[i] - sc) * (dst[i] - dc).transpose();
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    // Collinear / coincident sample → rank-deficient → unreliable.
    let s = svd.singular_values;
    if s[1] < 1e-9 || s[0] < 1e-12 {
        return None;
    }
    let mut r = vt.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut vfix = vt.transpose();
        vfix.column_mut(2).neg_mut();
        r = vfix * u.transpose();
    }
    let rot = Rotation3::from_matrix_unchecked(r);
    let t = dc - rot * sc;
    Some(Isometry3::from_parts(Translation3::from(t), rot.into()))
}

/// Grunert P3P: up to 4 `T_cw` poses from 3 world points and their
/// observed **normalized image points** (`(X/Z, Y/Z)`). Solutions with a
/// point behind the camera or non-finite geometry are dropped.
pub fn p3p(world: &[Vector3<f64>; 3], obs: &[Vector2<f64>; 3]) -> Vec<Isometry3<f64>> {
    // Unit bearings from the normalized image points.
    let j: [Vector3<f64>; 3] = [
        Vector3::new(obs[0].x, obs[0].y, 1.0).normalize(),
        Vector3::new(obs[1].x, obs[1].y, 1.0).normalize(),
        Vector3::new(obs[2].x, obs[2].y, 1.0).normalize(),
    ];
    let a2 = (world[1] - world[2]).norm_squared(); // opposite ray pair (2,3)
    let b2 = (world[0] - world[2]).norm_squared(); // (1,3)
    let c2 = (world[0] - world[1]).norm_squared(); // (1,2)
    if !(a2 > 0.0 && b2 > 0.0 && c2 > 0.0) {
        return Vec::new();
    }
    let ca = j[1].dot(&j[2]); // cos α  (between rays 2,3)
    let cb = j[0].dot(&j[2]); // cos β  (between rays 1,3)
    let cg = j[0].dot(&j[1]); // cos γ  (between rays 1,2)
    let (k1, k2) = (a2 / c2, b2 / c2);
    if k2.abs() < 1e-12 {
        return Vec::new();
    }

    // Eliminating s1 leaves the system linear in u = s2/s1 once u² is
    // substituted out; u = -C(v)/U(v), and back-substitution gives the
    // quartic  C² + 2cosγ·C·U + W·U² = 0  in v = s3/s1.
    let g = (1.0 - k1) / k2;
    let cc = [g - 1.0, -2.0 * g * cb, g + 1.0]; // C(v), deg 2
    let uu = [2.0 * cg, -2.0 * ca]; // U(v), deg 1
    let ww = [1.0 - 1.0 / k2, 2.0 * cb / k2, -1.0 / k2]; // W(v), deg 2

    let c_sq = conv(&cc, &cc);
    let c_u = conv(&cc, &uu);
    let u_sq = conv(&uu, &uu);
    let w_u2 = conv(&ww, &u_sq);
    let mut q = vec![0.0; 5];
    for (i, &x) in c_sq.iter().enumerate() {
        q[i] += x;
    }
    for (i, &x) in c_u.iter().enumerate() {
        q[i] += 2.0 * cg * x;
    }
    for (i, &x) in w_u2.iter().enumerate() {
        q[i] += x;
    }

    let mut out = Vec::new();
    for v in real_roots(&q) {
        if !(v.is_finite() && v > 0.0) {
            continue;
        }
        let cv = cc[0] + cc[1] * v + cc[2] * v * v;
        let uv = uu[0] + uu[1] * v;
        if uv.abs() < 1e-12 {
            continue;
        }
        let u = -cv / uv;
        if !(u.is_finite() && u > 0.0) {
            continue;
        }
        let den = 1.0 + u * u - 2.0 * u * cg;
        if den <= 1e-12 {
            continue;
        }
        let s1 = (c2 / den).sqrt();
        if !s1.is_finite() || s1 <= 0.0 {
            continue;
        }
        let (s2, s3) = (u * s1, v * s1);
        let cam = [s1 * j[0], s2 * j[1], s3 * j[2]];
        if cam
            .iter()
            .any(|p| !p.iter().all(|x| x.is_finite()) || p.z <= 0.0)
        {
            continue;
        }
        if let Some(t_cw) = kabsch(world, &cam) {
            out.push(t_cw);
        }
    }
    out
}

#[inline]
fn reproj_err(t: &Isometry3<f64>, x: &Vector3<f64>, z: &Vector2<f64>) -> f64 {
    let pc = t.rotation * x + t.translation.vector;
    if pc.z <= 1e-6 {
        return f64::INFINITY;
    }
    (Vector2::new(pc.x / pc.z, pc.y / pc.z) - z).norm()
}

fn count_inliers(t: &Isometry3<f64>, obs: &[Observation], thr: f64) -> (Vec<bool>, usize) {
    let mut mask = Vec::with_capacity(obs.len());
    let mut n = 0;
    for o in obs {
        let ok = reproj_err(t, &o.point, &o.obs) <= thr;
        mask.push(ok);
        n += ok as usize;
    }
    (mask, n)
}

/// Recover `T_cw` from `obs` (3D world points ↔ normalized image
/// points) by P3P-RANSAC, then polish on the inlier set with
/// motion-only BA. `None` if there are too few correspondences or no
/// pose reaches `min_inliers`.
pub fn pnp_ransac(obs: &[Observation], opt: PnpOptions) -> Option<PnpReport> {
    let n = obs.len();
    if n < 4 {
        return None;
    }
    let mut rng = SmallRng::seed_from_u64(opt.seed | 1);
    let mut best: Option<(usize, Isometry3<f64>)> = None;

    for _ in 0..opt.iters {
        // Three distinct correspondences.
        let i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n);
        while j == i {
            j = rng.random_range(0..n);
        }
        let mut k = rng.random_range(0..n);
        while k == i || k == j {
            k = rng.random_range(0..n);
        }
        let w = [obs[i].point, obs[j].point, obs[k].point];
        let z = [obs[i].obs, obs[j].obs, obs[k].obs];
        for pose in p3p(&w, &z) {
            let (_, ninl) = count_inliers(&pose, obs, opt.thresh);
            if best.as_ref().is_none_or(|&(b, _)| ninl > b) {
                best = Some((ninl, pose));
            }
        }
    }

    let (_, coarse) = best?;
    // Polish: motion-only BA over the current inliers, then re-score all
    // correspondences under the refined pose (the tested refiner already
    // carries a Huber robustifier).
    let (mask0, _) = count_inliers(&coarse, obs, opt.thresh);
    let inl: Vec<Observation> = obs
        .iter()
        .zip(&mask0)
        .filter(|&(_, &m)| m)
        .map(|(o, _)| *o)
        .collect();
    let pose = if inl.len() >= 4 {
        match track_pose(&inl, &coarse, opt.thresh, opt.thresh) {
            Some(r) if r.converged => r.pose,
            _ => coarse,
        }
    } else {
        coarse
    };
    let (inliers, n_inliers) = count_inliers(&pose, obs, opt.thresh);
    // Keep whichever of refined/coarse explains more (refinement can
    // occasionally diverge from a thin inlier set).
    let (pose, inliers, n_inliers) = {
        let (m0, c0) = count_inliers(&coarse, obs, opt.thresh);
        if c0 > n_inliers {
            (coarse, m0, c0)
        } else {
            (pose, inliers, n_inliers)
        }
    };
    (n_inliers >= opt.min_inliers).then_some(PnpReport {
        pose,
        inliers,
        n_inliers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie::se3_log;
    use nalgebra::UnitQuaternion;

    fn iso(rx: f64, ry: f64, rz: f64, tx: f64, ty: f64, tz: f64) -> Isometry3<f64> {
        let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(rx, ry, rz));
        Isometry3::from_parts(Translation3::new(tx, ty, tz), rot)
    }

    fn project(t: &Isometry3<f64>, x: &Vector3<f64>) -> Vector2<f64> {
        let pc = t.rotation * x + t.translation.vector;
        Vector2::new(pc.x / pc.z, pc.y / pc.z)
    }

    fn cloud(n: usize, seed: u64) -> Vec<Vector3<f64>> {
        let mut r = SmallRng::seed_from_u64(seed | 1);
        (0..n)
            .map(|_| {
                Vector3::new(
                    (r.random::<f64>() - 0.5) * 4.0,
                    (r.random::<f64>() - 0.5) * 3.0,
                    2.5 + r.random::<f64>() * 5.0,
                )
            })
            .collect()
    }

    fn pose_err(a: &Isometry3<f64>, b: &Isometry3<f64>) -> f64 {
        se3_log(&(a.inverse() * b)).norm()
    }

    #[test]
    fn p3p_recovers_exact_pose() {
        let gt = iso(0.07, -0.12, 0.05, 0.4, -0.2, 0.3);
        // Several well-conditioned triples must each yield the true pose.
        for s in 0..6 {
            let pts = cloud(3, 10 + s);
            let w = [pts[0], pts[1], pts[2]];
            let z = [
                project(&gt, &w[0]),
                project(&gt, &w[1]),
                project(&gt, &w[2]),
            ];
            let sols = p3p(&w, &z);
            assert!(!sols.is_empty(), "no P3P solution (triple {s})");
            let best = sols
                .iter()
                .map(|p| pose_err(p, &gt))
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-6, "P3P off by {best} (triple {s})");
        }
    }

    #[test]
    fn ransac_recovers_pose_with_outliers_and_noise() {
        let gt = iso(-0.05, 0.09, -0.03, -0.3, 0.15, 0.5);
        let pts = cloud(80, 42);
        let mut r = SmallRng::seed_from_u64(7);
        let mut obs: Vec<Observation> = pts
            .iter()
            .map(|&x| {
                let mut z = project(&gt, &x);
                z.x += (r.random::<f64>() - 0.5) * 2.0 * 0.001; // mild pixel noise
                z.y += (r.random::<f64>() - 0.5) * 2.0 * 0.001;
                Observation { point: x, obs: z }
            })
            .collect();
        // Corrupt ~25% with gross-wrong observations.
        for o in obs.iter_mut().step_by(4) {
            o.obs = Vector2::new(o.obs.x + 0.4, o.obs.y - 0.35);
        }
        let rep = pnp_ransac(&obs, PnpOptions::default()).expect("relocalized");
        assert!(
            pose_err(&rep.pose, &gt) < 0.02,
            "pose Δ={}",
            pose_err(&rep.pose, &gt)
        );
        // Most genuine points are inliers; the planted outliers are not.
        assert!(rep.n_inliers >= 50, "only {} inliers", rep.n_inliers);
        let bad_in = obs
            .iter()
            .zip(&rep.inliers)
            .step_by(4)
            .filter(|&(_, &m)| m)
            .count();
        assert!(bad_in <= 2, "outliers leaked into inliers: {bad_in}");
    }

    #[test]
    fn deterministic() {
        let gt = iso(0.02, 0.2, -0.1, 0.6, -0.1, 0.4);
        let pts = cloud(50, 99);
        let obs: Vec<Observation> = pts
            .iter()
            .map(|&x| Observation {
                point: x,
                obs: project(&gt, &x),
            })
            .collect();
        let a = pnp_ransac(&obs, PnpOptions::default()).unwrap();
        let b = pnp_ransac(&obs, PnpOptions::default()).unwrap();
        assert_eq!(a.n_inliers, b.n_inliers);
        assert_eq!(a.inliers, b.inliers);
        assert!(pose_err(&a.pose, &b.pose) < 1e-12);
    }

    #[test]
    fn too_few_points_is_none() {
        let gt = Isometry3::identity();
        let obs: Vec<Observation> = cloud(3, 1)
            .iter()
            .map(|&x| Observation {
                point: x,
                obs: project(&gt, &x),
            })
            .collect();
        assert!(pnp_ransac(&obs, PnpOptions::default()).is_none());
    }

    #[test]
    fn collinear_sample_is_safe() {
        // A degenerate (collinear) triple must not panic and yields no
        // usable pose from that sample.
        let gt = iso(0.0, 0.1, 0.0, 0.2, 0.0, 0.3);
        let w = [
            Vector3::new(0.0, 0.0, 5.0),
            Vector3::new(0.5, 0.5, 5.0),
            Vector3::new(1.0, 1.0, 5.0), // colinear with the first two
        ];
        let z = [
            project(&gt, &w[0]),
            project(&gt, &w[1]),
            project(&gt, &w[2]),
        ];
        let _ = p3p(&w, &z); // must not panic; solutions (if any) are dropped by Kabsch
    }
}
