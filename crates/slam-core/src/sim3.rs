//! Sim(3) similarity group + Essential-graph pose-graph optimization
//! (M6-1c): the loop-closure correction stage.
//!
//! Monocular SLAM has no metric scale and it *drifts*, so when a loop is
//! detected the two ends of the trajectory disagree by a **similarity**
//! (scale + rotation + translation), not a rigid transform. The fix:
//! represent keyframe poses as [`Sim3`], constrain the keyframe graph
//! (spanning tree + covisibility "essential" edges) plus the detected
//! loop edge, and optimize so the loop closes and the scale drift is
//! distributed over the graph.
//!
//! Provided here (camera-free, deterministic, headless-tested):
//! - [`Sim3`] with `exp`/`log` (Strasdat closed form), inverse, compose,
//!   point action — conventions pinned by tests, in the same hand-rolled
//!   spirit as [`crate::lie`].
//! - [`sim3_align`] — closed-form Umeyama similarity from 3D↔3D
//!   correspondences (recovers scale); how loop detection turns matched
//!   map points into the loop edge's measured Sim3.
//! - [`optimize_pose_graph`] — LM over relative-Sim3 residuals (finite-
//!   difference Jacobians, accept/reject, gauge-fixed), the same numeric
//!   style as `local_ba`/`tracking`.
//!
//! Deferred to M6-2 wiring: building the edge set from [`crate::map`]'s
//! spanning tree/covisibility and running this after a BoW + PnP/Sim3
//! verified loop. Dense normal equations are adequate at keyframe-graph
//! sizes (it runs rarely, off the tracking core); a sparse solver is a
//! later perf step if measured.

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

use crate::lie::{so3_exp, so3_log};

const EPS: f64 = 1e-5;

/// A 3D similarity transform: `y = s·R·x + t` (7 DoF).
#[derive(Clone, Copy, Debug)]
pub struct Sim3 {
    pub s: f64,
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
}

impl Sim3 {
    pub fn identity() -> Self {
        Self {
            s: 1.0,
            r: Rotation3::identity(),
            t: Vector3::zeros(),
        }
    }

    pub fn new(s: f64, r: Rotation3<f64>, t: Vector3<f64>) -> Self {
        Self { s, r, t }
    }

    /// Apply to a point: `s·R·p + t`.
    pub fn transform(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.s * (self.r * p) + self.t
    }

    /// Inverse similarity (`x` from `y = s·R·x + t`).
    pub fn inverse(&self) -> Self {
        let si = 1.0 / self.s;
        let ri = self.r.inverse();
        Self {
            s: si,
            r: ri,
            t: -si * (ri * self.t),
        }
    }

    /// Composition `self ∘ other` (apply `other` first).
    pub fn then(&self, other: &Sim3) -> Self {
        Self {
            s: self.s * other.s,
            r: self.r * other.r,
            t: self.s * (self.r * other.t) + self.t,
        }
    }
}

impl std::ops::Mul for Sim3 {
    type Output = Sim3;
    fn mul(self, rhs: Sim3) -> Sim3 {
        self.then(&rhs)
    }
}

/// The Sim(3) `exp` translation-coupling matrix
/// `W = C·I + A·Ω + B·Ω²` (Strasdat), with the small-`σ`/small-`θ`
/// series so it stays finite. `Ω = [φ]ₓ`, scale `= e^σ`.
fn calc_w(sigma: f64, phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    let om = crate::lie::hat(phi);
    let om2 = om * om;
    let s = sigma.exp();
    let (a, b, c);
    if sigma.abs() < EPS {
        c = 1.0;
        if theta < EPS {
            a = 0.5;
            b = 1.0 / 6.0;
        } else {
            a = (1.0 - theta.cos()) / (theta * theta);
            b = (theta - theta.sin()) / (theta * theta * theta);
        }
    } else {
        c = (s - 1.0) / sigma;
        if theta < EPS {
            let s2 = sigma * sigma;
            a = ((sigma - 1.0) * s + 1.0) / s2;
            b = (s * (0.5 * s2 - sigma + 1.0) - 1.0) / (s2 * sigma);
        } else {
            let sn = s * theta.sin();
            let cs = s * theta.cos();
            let cc = theta * theta + sigma * sigma;
            a = (sn * sigma + (1.0 - cs) * theta) / (theta * cc);
            b = (c - ((cs - 1.0) * sigma + sn * theta) / cc) / (theta * theta);
        }
    }
    c * Matrix3::identity() + a * om + b * om2
}

/// Sim(3) exponential: tangent `[υ(3); φ(3); σ]` → [`Sim3`]
/// (`scale = e^σ`, `R = exp(φ)`, `t = W·υ`).
pub fn sim3_exp(xi: &[f64; 7]) -> Sim3 {
    let upsilon = Vector3::new(xi[0], xi[1], xi[2]);
    let phi = Vector3::new(xi[3], xi[4], xi[5]);
    let sigma = xi[6];
    let r = Rotation3::from_matrix_unchecked(so3_exp(&phi));
    let t = calc_w(sigma, &phi) * upsilon;
    Sim3 {
        s: sigma.exp(),
        r,
        t,
    }
}

/// Sim(3) logarithm: inverse of [`sim3_exp`].
pub fn sim3_log(g: &Sim3) -> [f64; 7] {
    let sigma = g.s.ln();
    let phi = so3_log(g.r.matrix());
    let w = calc_w(sigma, &phi);
    let upsilon = w
        .try_inverse()
        .map(|wi| wi * g.t)
        .unwrap_or_else(|| g.t / g.s);
    [upsilon.x, upsilon.y, upsilon.z, phi.x, phi.y, phi.z, sigma]
}

/// Closed-form Umeyama similarity: the `Sim3` (with scale) best mapping
/// `src[i] → dst[i]` in least squares. `None` for < 3 points or a
/// degenerate (collinear/coincident) source.
pub fn sim3_align(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Option<Sim3> {
    let n = src.len();
    if n < 3 || dst.len() != n {
        return None;
    }
    let inv_n = 1.0 / n as f64;
    let mu_s = src.iter().sum::<Vector3<f64>>() * inv_n;
    let mu_d = dst.iter().sum::<Vector3<f64>>() * inv_n;
    let mut cov = Matrix3::zeros();
    let mut var_s = 0.0;
    for (a, b) in src.iter().zip(dst) {
        let da = a - mu_s;
        cov += (b - mu_d) * da.transpose();
        var_s += da.norm_squared();
    }
    cov *= inv_n;
    var_s *= inv_n;
    if var_s < 1e-12 {
        return None;
    }
    let svd = cov.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let d = svd.singular_values;
    if d[1] < 1e-9 {
        return None; // collinear source → similarity ill-posed
    }
    let mut w = Vector3::new(1.0, 1.0, 1.0);
    if (u.determinant() * v_t.determinant()) < 0.0 {
        w.z = -1.0;
    }
    let r = u * Matrix3::from_diagonal(&w) * v_t;
    let rot = Rotation3::from_matrix_unchecked(r);
    let scale = (d[0] * w.x + d[1] * w.y + d[2] * w.z) / var_s;
    let t = mu_d - scale * (rot * mu_s);
    Some(Sim3::new(scale, rot, t))
}

/// Deterministic xorshift PRNG (shared scheme across the core).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn upto(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sim3RansacOptions {
    pub iters: usize,
    /// Inlier gate on `‖S·src − dst‖` (same units as the points).
    pub thresh: f64,
    pub min_inliers: usize,
    pub seed: u64,
}

impl Default for Sim3RansacOptions {
    fn default() -> Self {
        Self {
            iters: 200,
            thresh: 0.05,
            min_inliers: 8,
            seed: 0x5132_0DED,
        }
    }
}

/// Outcome of [`sim3_ransac`].
#[derive(Clone, Debug)]
pub struct Sim3Report {
    pub pose: Sim3,
    pub inliers: Vec<bool>,
    pub n_inliers: usize,
}

/// Robust similarity from putative 3D↔3D correspondences (loop-closure
/// verification): RANSAC over minimal [`sim3_align`] fits (3 samples),
/// then a final refit on the largest consensus set. `None` if fewer
/// than `min_inliers` agree. Deterministic given `opt.seed`.
pub fn sim3_ransac(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    opt: Sim3RansacOptions,
) -> Option<Sim3Report> {
    let n = src.len();
    if n < 3 || dst.len() != n {
        return None;
    }
    let count = |s: &Sim3| -> usize {
        src.iter()
            .zip(dst)
            .filter(|(a, b)| (s.transform(a) - *b).norm() <= opt.thresh)
            .count()
    };
    let mut rng = Rng(opt.seed | 1);
    let mut best: Option<(usize, Sim3)> = None;
    for _ in 0..opt.iters {
        let i = rng.upto(n);
        let mut j = rng.upto(n);
        while j == i {
            j = rng.upto(n);
        }
        let mut k = rng.upto(n);
        while k == i || k == j {
            k = rng.upto(n);
        }
        let s3 = [src[i], src[j], src[k]];
        let d3 = [dst[i], dst[j], dst[k]];
        if let Some(s) = sim3_align(&s3, &d3) {
            let c = count(&s);
            if best.as_ref().is_none_or(|&(b, _)| c > b) {
                best = Some((c, s));
            }
        }
    }
    let (_, coarse) = best?;
    // Refit on the full consensus set.
    let (mut si, mut so): (Vec<Vector3<f64>>, Vec<Vector3<f64>>) = (Vec::new(), Vec::new());
    for (a, b) in src.iter().zip(dst) {
        if (coarse.transform(a) - b).norm() <= opt.thresh {
            si.push(*a);
            so.push(*b);
        }
    }
    let pose = if si.len() >= 3 {
        sim3_align(&si, &so).unwrap_or(coarse)
    } else {
        coarse
    };
    let inliers: Vec<bool> = src
        .iter()
        .zip(dst)
        .map(|(a, b)| (pose.transform(a) - b).norm() <= opt.thresh)
        .collect();
    let n_inliers = inliers.iter().filter(|&&x| x).count();
    (n_inliers >= opt.min_inliers).then_some(Sim3Report {
        pose,
        inliers,
        n_inliers,
    })
}

/// One pose-graph constraint: the measured relative similarity such that
/// ideally `poses[i] = meas ∘ poses[j]` (residual zero). `weight` scales
/// the squared error (e.g. covisibility strength).
#[derive(Clone, Copy, Debug)]
pub struct PgEdge {
    pub i: usize,
    pub j: usize,
    pub meas: Sim3,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphOptions {
    pub iters: usize,
    pub lambda0: f64,
}

impl Default for PoseGraphOptions {
    fn default() -> Self {
        Self {
            iters: 20,
            lambda0: 1e-3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphReport {
    pub cost0: f64,
    pub cost1: f64,
    pub iters: usize,
    pub edges: usize,
}

/// Residual of edge `(i, j)`: `log( measᐩ ∘ Sᵢ ∘ Sⱼᐩ )`, weighted.
fn edge_residual(poses: &[Sim3], e: &PgEdge) -> [f64; 7] {
    let rel = e
        .meas
        .inverse()
        .then(&poses[e.i])
        .then(&poses[e.j].inverse());
    let l = sim3_log(&rel);
    let w = e.weight.sqrt();
    let mut r = [0.0; 7];
    for k in 0..7 {
        r[k] = l[k] * w;
    }
    r
}

/// Optimize keyframe `Sim3` poses to satisfy the relative-pose
/// constraints (spanning-tree + covisibility + loop edges), absorbing
/// monocular scale drift so a detected loop closes. `fixed` is held
/// constant for gauge (anchors origin + global scale). LM with central
/// finite-difference Jacobians + accept/reject, like `local_ba`.
pub fn optimize_pose_graph(
    poses: &mut [Sim3],
    edges: &[PgEdge],
    fixed: usize,
    opt: PoseGraphOptions,
) -> PoseGraphReport {
    let n = poses.len();
    // Free poses get a contiguous parameter-block index.
    let mut blk = vec![usize::MAX; n];
    let mut free = 0;
    for (i, b) in blk.iter_mut().enumerate() {
        if i != fixed {
            *b = free;
            free += 1;
        }
    }
    let dim = 7 * free;
    let total_cost = |p: &[Sim3]| -> f64 {
        edges
            .iter()
            .map(|e| edge_residual(p, e).iter().map(|x| x * x).sum::<f64>())
            .sum()
    };
    let cost0 = total_cost(poses);
    if dim == 0 || edges.is_empty() {
        return PoseGraphReport {
            cost0,
            cost1: cost0,
            iters: 0,
            edges: edges.len(),
        };
    }

    // Left-perturbation retraction: Sₖ ← exp(δₖ) ∘ Sₖ for free k.
    let retract = |p: &[Sim3], step: &DVector<f64>| -> Vec<Sim3> {
        let mut out = p.to_vec();
        for k in 0..n {
            if blk[k] == usize::MAX {
                continue;
            }
            let o = 7 * blk[k];
            let d = [
                step[o],
                step[o + 1],
                step[o + 2],
                step[o + 3],
                step[o + 4],
                step[o + 5],
                step[o + 6],
            ];
            out[k] = sim3_exp(&d).then(&p[k]);
        }
        out
    };

    let m = 7 * edges.len();
    let eps = 1e-6;
    let mut lambda = opt.lambda0;
    let mut iters_done = 0;

    for _ in 0..opt.iters {
        iters_done += 1;
        // r and J about the current estimate (finite differences).
        let mut r = DVector::zeros(m);
        for (ei, e) in edges.iter().enumerate() {
            let res = edge_residual(poses, e);
            for k in 0..7 {
                r[7 * ei + k] = res[k];
            }
        }
        let mut j = DMatrix::zeros(m, dim);
        for (k, &bk) in blk.iter().enumerate() {
            if bk == usize::MAX {
                continue;
            }
            let col0 = 7 * bk;
            for d in 0..7 {
                let mut step = DVector::zeros(dim);
                step[col0 + d] = eps;
                let pp = retract(poses, &step);
                step[col0 + d] = -eps;
                let pm = retract(poses, &step);
                for (ei, e) in edges.iter().enumerate() {
                    if e.i != k && e.j != k {
                        continue; // edge independent of this pose
                    }
                    let rp = edge_residual(&pp, e);
                    let rm = edge_residual(&pm, e);
                    for row in 0..7 {
                        j[(7 * ei + row, col0 + d)] = (rp[row] - rm[row]) / (2.0 * eps);
                    }
                }
            }
        }

        let jt = j.transpose();
        let mut h = &jt * &j;
        let g = &jt * &r;
        for d in 0..dim {
            h[(d, d)] += lambda * h[(d, d)].max(1e-9);
        }
        let Some(chol) = h.clone().cholesky() else {
            lambda *= 10.0;
            continue;
        };
        let step = chol.solve(&(-&g));
        let trial = retract(poses, &step);
        if total_cost(&trial) < total_cost(poses) {
            poses.copy_from_slice(&trial);
            lambda = (lambda * 0.3).max(1e-12);
        } else {
            lambda *= 10.0;
        }
    }

    PoseGraphReport {
        cost0,
        cost1: total_cost(poses),
        iters: iters_done,
        edges: edges.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn rot(rx: f64, ry: f64, rz: f64) -> Rotation3<f64> {
        Rotation3::from_euler_angles(rx, ry, rz)
    }

    #[test]
    fn exp_log_roundtrip_incl_tiny_and_pure_scale() {
        let cases: [[f64; 7]; 5] = [
            [0.5, -1.2, 3.0, 0.2, -0.7, 0.9, 0.4],
            [1.0, 2.0, -0.5, 1e-9, 0.0, -1e-9, 1e-9], // tiny rot+scale
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      // identity
            [0.3, 0.1, -0.2, 0.0, 0.0, 0.0, 0.8],     // pure scale+trans
            [-2.0, 0.4, 1.1, 0.9, -1.3, 0.5, -0.6],   // shrink
        ];
        for xi in cases {
            let g = sim3_exp(&xi);
            let back = sim3_log(&g);
            let dn: f64 = (0..7).map(|k| (back[k] - xi[k]).powi(2)).sum();
            assert!(dn.sqrt() < 1e-7, "xi={xi:?} back={back:?}");
        }
    }

    #[test]
    fn group_axioms_inverse_compose_action() {
        let a = sim3_exp(&[0.4, -0.3, 0.7, 0.2, 0.5, -0.1, 0.3]);
        let b = sim3_exp(&[-0.6, 0.2, 0.1, -0.4, 0.3, 0.25, -0.2]);
        let p = Vector3::new(1.3, -2.1, 0.7);
        // (a∘b)(p) == a(b(p))
        let lhs = (a * b).transform(&p);
        let rhs = a.transform(&b.transform(&p));
        assert!((lhs - rhs).norm() < 1e-12);
        // a∘a⁻¹ == identity (acts as identity on p).
        let id = a * a.inverse();
        assert!((id.transform(&p) - p).norm() < 1e-10);
        assert!((id.s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sim3_align_recovers_scaled_rigid() {
        let mut r = Rng(0xA11);
        let truth = Sim3::new(2.3, rot(0.2, -0.4, 0.7), Vector3::new(1.0, -2.0, 0.5));
        let src: Vec<Vector3<f64>> = (0..30)
            .map(|_| Vector3::new(r.f() * 4.0, r.f() * 3.0, r.f() * 5.0))
            .collect();
        let dst: Vec<Vector3<f64>> = src.iter().map(|p| truth.transform(p)).collect();
        // Frobenius-norm rotation compare (clamp-free, unlike acos-based
        // angle metrics that go NaN at a ~0 relative angle).
        let rot_err = |a: &Rotation3<f64>, b: &Rotation3<f64>| (a.matrix() - b.matrix()).norm();
        let est = sim3_align(&src, &dst).expect("aligned");
        assert!(
            (est.s - truth.s).abs() < 1e-9,
            "scale {} vs {}",
            est.s,
            truth.s
        );
        assert!(rot_err(&est.r, &truth.r) < 1e-9);
        assert!((est.t - truth.t).norm() < 1e-8);
        // Noisy: still close.
        let dst_n: Vec<Vector3<f64>> = dst
            .iter()
            .map(|p| p + Vector3::new(r.f() - 0.5, r.f() - 0.5, r.f() - 0.5) * 0.02)
            .collect();
        let en = sim3_align(&src, &dst_n).expect("aligned noisy");
        assert!((en.s - truth.s).abs() < 0.05);
        assert!(rot_err(&en.r, &truth.r) < 0.1);
        // Degenerate guards.
        assert!(sim3_align(&src[..2], &dst[..2]).is_none());
        let line: Vec<Vector3<f64>> = (0..5).map(|k| Vector3::new(k as f64, 0.0, 0.0)).collect();
        assert!(sim3_align(&line, &line).is_none());
    }

    /// A drifted keyframe chain + a loop constraint: pose-graph
    /// optimization must drop the cost, close the loop, and pull the
    /// accumulated scale drift back toward the true (unit) scale.
    #[test]
    fn pose_graph_closes_a_drifted_loop() {
        // Ground-truth: 12 keyframes round a square-ish loop, unit scale.
        let n = 12;
        let truth: Vec<Sim3> = (0..n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                // World→keyframe poses around a circle, looking inward-ish.
                let t = Vector3::new(3.0 * a.cos(), 0.0, 3.0 * a.sin());
                Sim3::new(1.0, rot(0.0, -a, 0.0), -(rot(0.0, -a, 0.0) * t))
            })
            .collect();

        // Odometry edges (consecutive) measured at truth.
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            let meas = truth[i + 1].then(&truth[i].inverse());
            edges.push(PgEdge {
                i: i + 1,
                j: i,
                meas,
                weight: 1.0,
            });
        }
        // Loop edge: keyframe n-1 back to 0, measured at truth.
        edges.push(PgEdge {
            i: n - 1,
            j: 0,
            meas: truth[n - 1].then(&truth[0].inverse()),
            weight: 1.0,
        });

        // Estimate: integrate odometry but inject per-step scale + yaw
        // drift, so the chain spirals and the loop is wide open.
        let mut est = vec![truth[0]];
        for i in 0..n - 1 {
            let mut rel = truth[i + 1].then(&truth[i].inverse());
            rel.s *= 1.05; // 5%/step scale drift
            rel.r *= rot(0.0, 0.03, 0.0);
            est.push(rel.then(&est[i]));
        }
        let gap_before = (est[n - 1].inverse().t - truth[n - 1].inverse().t).norm();
        let scale_drift_before = est.iter().map(|p| (p.s - 1.0).abs()).fold(0.0, f64::max);

        let rep = optimize_pose_graph(&mut est, &edges, 0, PoseGraphOptions::default());
        assert!(
            rep.cost1 < 0.05 * rep.cost0 + 1e-9,
            "cost {} -> {}",
            rep.cost0,
            rep.cost1
        );
        // Loop now closed: last keyframe centre near truth.
        let gap_after = (est[n - 1].inverse().t - truth[n - 1].inverse().t).norm();
        assert!(
            gap_after < 0.1 * gap_before,
            "loop gap {gap_before} -> {gap_after}"
        );
        // Scale drift pulled back toward unity.
        let scale_drift_after = est.iter().map(|p| (p.s - 1.0).abs()).fold(0.0, f64::max);
        assert!(
            scale_drift_after < 0.3 * scale_drift_before,
            "scale drift {scale_drift_before} -> {scale_drift_after}"
        );
        // Gauge keyframe untouched.
        assert!((est[0].s - truth[0].s).abs() < 1e-12);
    }

    #[test]
    fn pose_graph_noop_without_free_or_edges() {
        let mut p = vec![Sim3::identity(), Sim3::identity()];
        let r = optimize_pose_graph(&mut p, &[], 0, PoseGraphOptions::default());
        assert_eq!((r.iters, r.edges), (0, 0));
    }

    #[test]
    fn sim3_ransac_robust_to_outliers() {
        let mut r = Rng(0xC0FFEE);
        let truth = Sim3::new(1.7, rot(-0.1, 0.3, 0.2), Vector3::new(-1.0, 2.0, 0.7));
        let src: Vec<Vector3<f64>> = (0..60)
            .map(|_| Vector3::new(r.f() * 4.0, r.f() * 3.0, r.f() * 5.0))
            .collect();
        let mut dst: Vec<Vector3<f64>> = src.iter().map(|p| truth.transform(p)).collect();
        // ~30% gross outliers.
        for d in dst.iter_mut().step_by(3) {
            *d += Vector3::new(r.f() * 10.0 - 5.0, r.f() * 10.0 - 5.0, r.f() * 10.0 - 5.0);
        }
        let rep = sim3_ransac(
            &src,
            &dst,
            Sim3RansacOptions {
                thresh: 1e-6,
                ..Sim3RansacOptions::default()
            },
        )
        .expect("verified");
        assert!((rep.pose.s - truth.s).abs() < 1e-6, "scale {}", rep.pose.s);
        assert!((rep.pose.r.matrix() - truth.r.matrix()).norm() < 1e-6);
        assert!((rep.pose.t - truth.t).norm() < 1e-6);
        assert!(rep.n_inliers >= 35, "only {} inliers", rep.n_inliers);
        // Planted outliers are flagged out.
        let bad_in = rep.inliers.iter().step_by(3).filter(|&&x| x).count();
        assert!(bad_in <= 2, "outliers leaked: {bad_in}");
        // Determinism + too-few-correspondences guard.
        let rep2 = sim3_ransac(
            &src,
            &dst,
            Sim3RansacOptions {
                thresh: 1e-6,
                ..Sim3RansacOptions::default()
            },
        )
        .unwrap();
        assert_eq!(rep.n_inliers, rep2.n_inliers);
        assert!(sim3_ransac(&src[..2], &dst[..2], Sim3RansacOptions::default()).is_none());
    }
}
