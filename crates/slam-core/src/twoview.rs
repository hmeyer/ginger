//! Two-view monocular initialization geometry (M3).
//!
//! Camera-free and operating on **calibrated/normalized** image points
//! (`x = X/Z, y = Y/Z`; the caller undistorts + unprojects pixels via
//! [`crate::camera::CameraModel`] first). In calibrated coordinates the
//! fundamental matrix *is* the essential matrix, which keeps this a pure
//! geometry module that unit-tests deterministically against synthetic
//! scenes.
//!
//! Pipeline (ORB-SLAM-style): RANSAC an essential matrix (normalized
//! 8-point) **and** a homography (normalized 4-point DLT) in parallel,
//! score both with the symmetric transfer error, pick the model by the
//! `R_H = S_H / (S_H + S_F)` heuristic, then recover relative pose from
//! the chosen model — essential via SVD decomposition, homography via
//! the Faugeras–Lustman decomposition (the planar / low-parallax path)
//! — disambiguated by cheirality, with a fall-back to the other model
//! if the chosen one yields no valid pose. Triangulates the initial
//! points.

use ginger_rand::Rng64;
use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};

/// A calibrated correspondence: normalized image point in view 1 and 2.
pub type Corr = (Vector2<f64>, Vector2<f64>);

/// Relative pose `view1 → view2` (rotation, unit-free translation).
type Pose = (Matrix3<f64>, Vector3<f64>);
/// The four pose candidates from an essential-matrix decomposition.
type PoseCandidates = [Pose; 4];
/// A pose plus the points it triangulates.
type PoseWithPoints = (Matrix3<f64>, Vector3<f64>, Vec<Vector3<f64>>);

/// `k` distinct indices in `[0, n)`. RANSAC sampling is seeded (via
/// [`Rng64`]) so model selection and inlier sets are reproducible
/// across runs (CI).
fn sample(rng: &mut Rng64, n: usize, k: usize, out: &mut Vec<usize>) {
    out.clear();
    while out.len() < k {
        let i = rng.below(n);
        if !out.contains(&i) {
            out.push(i);
        }
    }
}

/// Hartley normalization: similarity `T` sending the centroid to the
/// origin and the mean distance to √2. Returns the transformed points
/// and `T` (so the un-normalized model is recovered by conjugation).
fn normalize(pts: &[Vector2<f64>]) -> (Vec<Vector2<f64>>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in pts {
        cx += p.x;
        cy += p.y;
    }
    cx /= n;
    cy /= n;
    let mut mean_d = 0.0;
    for p in pts {
        mean_d += ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt();
    }
    mean_d /= n;
    let s = if mean_d > 1e-12 {
        (2.0_f64).sqrt() / mean_d
    } else {
        1.0
    };
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let out = pts
        .iter()
        .map(|p| Vector2::new(s * (p.x - cx), s * (p.y - cy)))
        .collect();
    (out, t)
}

/// Smallest-eigenvector solution of `A x = 0`, via the eigendecomposition
/// of `AᵀA`. Going through the `n×n` normal matrix (rather than the SVD
/// of `A`) keeps this correct for *wide* `A` too — a minimal sample
/// makes `A` 8×9, where a thin SVD of `A` would not return the true
/// null direction.
fn null_vector(a: DMatrix<f64>) -> Vec<f64> {
    let ata = a.transpose() * a;
    let se = ata.symmetric_eigen();
    let mut mi = 0;
    for i in 1..se.eigenvalues.len() {
        if se.eigenvalues[i] < se.eigenvalues[mi] {
            mi = i;
        }
    }
    se.eigenvectors.column(mi).iter().copied().collect()
}

/// Essential matrix from ≥8 calibrated correspondences (normalized
/// 8-point), with the essential constraint `diag(1, 1, 0)` enforced.
pub fn essential_8point(corrs: &[Corr]) -> Option<Matrix3<f64>> {
    if corrs.len() < 8 {
        return None;
    }
    let p1: Vec<_> = corrs.iter().map(|c| c.0).collect();
    let p2: Vec<_> = corrs.iter().map(|c| c.1).collect();
    let (q1, t1) = normalize(&p1);
    let (q2, t2) = normalize(&p2);

    let mut a = DMatrix::zeros(corrs.len(), 9);
    for (i, (u, v)) in q1.iter().zip(&q2).enumerate() {
        let row = [
            v.x * u.x,
            v.x * u.y,
            v.x,
            v.y * u.x,
            v.y * u.y,
            v.y,
            u.x,
            u.y,
            1.0,
        ];
        for (c, val) in row.iter().enumerate() {
            a[(i, c)] = *val;
        }
    }
    let f = null_vector(a);
    let fmat = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);

    // Denormalize *first* (the similarity T is not a rotation, so it
    // does not preserve the essential singular-value structure)…
    let f_denorm = t2.transpose() * fmat * t1;
    // …then enforce the essential structure: two equal non-zero
    // singular values, third zero.
    let svd = f_denorm.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let s = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    Some(u * s * vt)
}

/// Homography from ≥4 correspondences (normalized 4-point DLT).
pub fn homography_4point(corrs: &[Corr]) -> Option<Matrix3<f64>> {
    if corrs.len() < 4 {
        return None;
    }
    let p1: Vec<_> = corrs.iter().map(|c| c.0).collect();
    let p2: Vec<_> = corrs.iter().map(|c| c.1).collect();
    let (q1, t1) = normalize(&p1);
    let (q2, t2) = normalize(&p2);

    let mut a = DMatrix::zeros(2 * corrs.len(), 9);
    for (i, (u, v)) in q1.iter().zip(&q2).enumerate() {
        let r0 = [-u.x, -u.y, -1.0, 0.0, 0.0, 0.0, v.x * u.x, v.x * u.y, v.x];
        let r1 = [0.0, 0.0, 0.0, -u.x, -u.y, -1.0, v.y * u.x, v.y * u.y, v.y];
        for c in 0..9 {
            a[(2 * i, c)] = r0[c];
            a[(2 * i + 1, c)] = r1[c];
        }
    }
    let h = null_vector(a);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    // H = T2⁻¹ H_n T1.
    let t2_inv = t2.try_inverse()?;
    let hm = t2_inv * hn * t1;
    Some(hm / hm[(2, 2)])
}

/// Symmetric epipolar (transfer) distance for one correspondence.
fn epipolar_err(e: &Matrix3<f64>, c: &Corr) -> f64 {
    let x1 = Vector3::new(c.0.x, c.0.y, 1.0);
    let x2 = Vector3::new(c.1.x, c.1.y, 1.0);
    let l2 = e * x1;
    let l1 = e.transpose() * x2;
    let d = (x2.dot(&(e * x1))).powi(2);
    d / (l2.x * l2.x + l2.y * l2.y) + d / (l1.x * l1.x + l1.y * l1.y)
}

/// Symmetric transfer distance for a homography.
fn homography_err(h: &Matrix3<f64>, hinv: &Matrix3<f64>, c: &Corr) -> f64 {
    let x1 = Vector3::new(c.0.x, c.0.y, 1.0);
    let x2 = Vector3::new(c.1.x, c.1.y, 1.0);
    let p2 = h * x1;
    let p1 = hinv * x2;
    let e2 = (Vector2::new(p2.x / p2.z, p2.y / p2.z) - c.1).norm_squared();
    let e1 = (Vector2::new(p1.x / p1.z, p1.y / p1.z) - c.0).norm_squared();
    e1 + e2
}

/// ORB-SLAM-style score for a model: `Σ_inliers (thScore − d)`.
fn score(errs: impl Iterator<Item = f64>, th: f64, th_score: f64) -> (f64, Vec<bool>) {
    let mut s = 0.0;
    let mask: Vec<bool> = errs
        .map(|d| {
            if d < th {
                s += th_score - d;
                true
            } else {
                false
            }
        })
        .collect();
    (s, mask)
}

/// Which two-view model RANSAC selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Essential,
    Homography,
}

/// Result of [`initialize`]: the chosen model, relative pose
/// `view1 → view2` (`t` is unit-norm — monocular scale is arbitrary),
/// the triangulated points (in view-1 calibrated frame), and the inlier
/// mask over the input correspondences.
#[derive(Clone, Debug)]
pub struct TwoView {
    pub model: Model,
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
    pub points: Vec<Vector3<f64>>,
    pub inliers: Vec<bool>,
    pub r_h: f64,
}

/// Linear DLT triangulation for `P1 = [I|0]`, `P2 = [R|t]` (calibrated).
fn triangulate(r: &Matrix3<f64>, t: &Vector3<f64>, c: &Corr) -> Vector3<f64> {
    let mut a = DMatrix::zeros(4, 4);
    // P1 rows are [I|0]; P2 rows are [R|t].
    let p1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let p2 = [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)], t.x],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)], t.y],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)], t.z],
    ];
    for k in 0..4 {
        a[(0, k)] = c.0.x * p1[2][k] - p1[0][k];
        a[(1, k)] = c.0.y * p1[2][k] - p1[1][k];
        a[(2, k)] = c.1.x * p2[2][k] - p2[0][k];
        a[(3, k)] = c.1.y * p2[2][k] - p2[1][k];
    }
    let x = null_vector(a);
    Vector3::new(x[0] / x[3], x[1] / x[3], x[2] / x[3])
}

/// Decompose `E` into the 4 (R, t) candidates.
fn decompose_essential(e: &Matrix3<f64>) -> PoseCandidates {
    let svd = e.svd(true, true);
    let mut u = svd.u.unwrap();
    let mut vt = svd.v_t.unwrap();
    // Force det(U) = det(V) = +1 so the recovered factors are proper
    // rotations (negating a whole factor keeps E = U Σ Vᵀ valid).
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * vt;
    let r2 = u * w.transpose() * vt;
    let t = Vector3::new(u[(0, 2)], u[(1, 2)], u[(2, 2)]);
    [(r1, t), (r1, -t), (r2, t), (r2, -t)]
}

/// Decompose a *calibrated* homography into its physically distinct
/// (R, t) candidates (Faugeras–Lustman SVD method). Translation is
/// up-to-scale; cheirality (`best_pose`) disambiguates the ≤ 8
/// solutions, same as the essential path.
fn decompose_homography(h: &Matrix3<f64>) -> Vec<Pose> {
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let sv = svd.singular_values;
    let (d1, d2, d3) = (sv[0], sv[1], sv[2]); // descending
    // Normalize so the middle singular value is 1 (Faugeras scaling).
    if d2.abs() < 1e-12 {
        return Vec::new();
    }
    let v = vt.transpose();
    let s = u.determinant() * v.determinant();

    // Near-pure-rotation (d1 ≈ d3): plane normal is undetermined.
    if (d1 - d3).abs() < 1e-9 {
        return vec![(s * u * vt, Vector3::zeros())];
    }

    let aux1 = ((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3)).max(0.0).sqrt();
    let aux3 = ((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3)).max(0.0).sqrt();
    let e1 = [1.0, -1.0, 1.0, -1.0];
    let e3 = [1.0, 1.0, -1.0, -1.0];
    let mut out = Vec::with_capacity(8);

    // Case d' = +d2.
    let aux_st = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).max(0.0).sqrt() / ((d1 + d3) * d2);
    let ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    for k in 0..4 {
        let stheta = aux_st * e1[k] * e3[k];
        let rp = Matrix3::new(ctheta, 0.0, -stheta, 0.0, 1.0, 0.0, stheta, 0.0, ctheta);
        let r = s * u * rp * vt;
        let tp = Vector3::new((d1 - d3) * aux1 * e1[k], 0.0, -(d1 - d3) * aux3 * e3[k]);
        out.push((r, u * tp));
    }
    // Case d' = -d2.
    let aux_sp = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).max(0.0).sqrt() / ((d1 - d3) * d2);
    let cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    for k in 0..4 {
        let sphi = aux_sp * e1[k] * e3[k];
        let rp = Matrix3::new(cphi, 0.0, sphi, 0.0, -1.0, 0.0, sphi, 0.0, -cphi);
        let r = s * u * rp * vt;
        let tp = Vector3::new((d1 + d3) * aux1 * e1[k], 0.0, (d1 + d3) * aux3 * e3[k]);
        out.push((r, u * tp));
    }
    out
}

/// Pick the (R, t) with the most inlier points in front of *both*
/// cameras (cheirality), returning that pose and its triangulated set.
/// Works for any candidate count (essential gives 4, homography ≤ 8).
fn best_pose(cands: &[Pose], corrs: &[Corr], inliers: &[bool]) -> Option<PoseWithPoints> {
    let mut best: Option<(usize, PoseWithPoints)> = None;
    for &(r, t) in cands.iter() {
        let mut pts = Vec::new();
        let mut good = 0usize;
        for (c, &inl) in corrs.iter().zip(inliers) {
            if !inl {
                continue;
            }
            let x = triangulate(&r, &t, c);
            let x2 = r * x + t;
            if x.z > 0.0 && x2.z > 0.0 && x.z.is_finite() && x2.z.is_finite() {
                good += 1;
            }
            pts.push(x);
        }
        if best.as_ref().is_none_or(|(b, _)| good > *b) {
            best = Some((good, (r, t, pts)));
        }
    }
    let (g, pose) = best?;
    if g == 0 { None } else { Some(pose) }
}

/// RANSAC config. `sigma` is the measurement noise std in the *same
/// units as the input* (normalized coords), so the chi-square gates
/// scale correctly.
#[derive(Clone, Copy, Debug)]
pub struct InitOptions {
    pub iters: usize,
    pub sigma: f64,
    pub seed: u64,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            iters: 300,
            sigma: 1.0 / 500.0, // ~1 px at f≈500
            seed: 0x1234_5678,
        }
    }
}

/// Two-view initialization from calibrated correspondences. Returns
/// `None` if neither model yields a cheirality-consistent pose.
pub fn initialize(corrs: &[Corr], opt: InitOptions) -> Option<TwoView> {
    if corrs.len() < 8 {
        return None;
    }
    let s2 = opt.sigma * opt.sigma;
    // ORB-SLAM chi-square gates (2-DOF F transfer uses the 1-DOF-style
    // 3.84 gate per direction; H symmetric transfer uses 5.99), scaled
    // by the measurement variance.
    let th_e = 3.841 * s2;
    let th_h = 5.991 * s2;
    let th_score = 5.991 * s2;

    let mut rng = Rng64::new(opt.seed | 1);
    let mut idx = Vec::new();

    // ── RANSAC essential ──
    let (mut best_e, mut best_se, mut mask_e) = (Matrix3::zeros(), -1.0, vec![]);
    for _ in 0..opt.iters {
        sample(&mut rng, corrs.len(), 8, &mut idx);
        let s: Vec<Corr> = idx.iter().map(|&i| corrs[i]).collect();
        let Some(e) = essential_8point(&s) else {
            continue;
        };
        let (sc, mask) = score(corrs.iter().map(|c| epipolar_err(&e, c)), th_e, th_score);
        if sc > best_se {
            best_se = sc;
            best_e = e;
            mask_e = mask;
        }
    }

    // ── RANSAC homography ──
    let (mut best_h, mut best_sh, mut mask_h) = (Matrix3::zeros(), -1.0, vec![]);
    for _ in 0..opt.iters {
        sample(&mut rng, corrs.len(), 4, &mut idx);
        let s: Vec<Corr> = idx.iter().map(|&i| corrs[i]).collect();
        let Some(h) = homography_4point(&s) else {
            continue;
        };
        let Some(hinv) = h.try_inverse() else {
            continue;
        };
        let (sc, mask) = score(
            corrs.iter().map(|c| homography_err(&h, &hinv, c)),
            th_h,
            th_score,
        );
        if sc > best_sh {
            best_sh = sc;
            best_h = h;
            mask_h = mask;
        }
    }

    let (se, sh) = (best_se.max(0.0), best_sh.max(0.0));
    let r_h = if se + sh > 0.0 { sh / (se + sh) } else { 0.0 };

    // ORB-SLAM: prefer the homography when it explains ≳45% of the
    // combined score (planar / low-parallax), recovering pose from H;
    // otherwise from the essential matrix. Fall back to E if the H
    // decomposition yields no cheirality-consistent pose.
    let model = if r_h > 0.45 {
        Model::Homography
    } else {
        Model::Essential
    };

    let recover = |m: Model| -> Option<(PoseWithPoints, Vec<bool>)> {
        match m {
            Model::Homography => {
                let cands = decompose_homography(&best_h);
                best_pose(&cands, corrs, &mask_h).map(|p| (p, mask_h.clone()))
            }
            Model::Essential => {
                let cands = decompose_essential(&best_e);
                best_pose(&cands[..], corrs, &mask_e).map(|p| (p, mask_e.clone()))
            }
        }
    };
    let order = if model == Model::Homography {
        [Model::Homography, Model::Essential]
    } else {
        [Model::Essential, Model::Homography]
    };
    let (used, ((r, t, points), inliers)) =
        order.into_iter().find_map(|m| recover(m).map(|p| (m, p)))?;
    Some(TwoView {
        model: used,
        r,
        t: t.normalize(),
        points,
        inliers,
        r_h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    /// Deterministic scene: `n` 3D points, projected into view 1
    /// (`[I|0]`) and view 2 (`[R|t]`), returned as calibrated corrs
    /// plus the ground-truth pose.
    fn scene(n: usize, planar: bool, noise: f64) -> (Vec<Corr>, Matrix3<f64>, Vector3<f64>) {
        let r = *Rotation3::from_euler_angles(0.03, -0.12, 0.05).matrix();
        let t = Vector3::new(0.9, -0.15, 0.08); // sideways translation
        let mut rng = Rng64::new(99);
        let mut corrs = Vec::new();
        for i in 0..n {
            let u = (rng.next_u64() as f64 / u64::MAX as f64) - 0.5;
            let v = (rng.next_u64() as f64 / u64::MAX as f64) - 0.5;
            let z = if planar {
                4.0 // single fronto-parallel plane
            } else {
                2.5 + 3.0 * (rng.next_u64() as f64 / u64::MAX as f64)
            };
            let p = Vector3::new(u * z * 0.8, v * z * 0.8, z);
            let p2 = r * p + t;
            if p.z <= 0.0 || p2.z <= 0.0 {
                continue;
            }
            let mut a = Vector2::new(p.x / p.z, p.y / p.z);
            let mut b = Vector2::new(p2.x / p2.z, p2.y / p2.z);
            if noise > 0.0 {
                let g =
                    |rng: &mut Rng64| (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 2.0 * noise;
                a.x += g(&mut rng);
                a.y += g(&mut rng);
                b.x += g(&mut rng);
                b.y += g(&mut rng);
            }
            corrs.push((a, b));
            let _ = i;
        }
        (corrs, r, t)
    }

    fn rot_angle_deg(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
        let rel = a.transpose() * b;
        (((rel.trace() - 1.0) * 0.5).clamp(-1.0, 1.0))
            .acos()
            .to_degrees()
    }

    #[test]
    fn essential_recovers_pose_noise_free() {
        let (corrs, r_gt, t_gt) = scene(120, false, 0.0);
        let tv = initialize(&corrs, InitOptions::default()).expect("init");
        assert_eq!(tv.model, Model::Essential);
        assert!(rot_angle_deg(&tv.r, &r_gt) < 1.0, "R off: {tv:?}");
        // Translation is up-to-scale: compare directions.
        let cos = tv.t.normalize().dot(&t_gt.normalize()).abs();
        assert!(cos > 0.999, "t dir off: cos={cos}");
        // Points align with the inlier correspondences (best_pose
        // pushes one per inlier, in order); they reproject onto view 1
        // with small error.
        let inlier_corrs: Vec<&Corr> = corrs
            .iter()
            .zip(&tv.inliers)
            .filter(|&(_, &i)| i)
            .map(|(c, _)| c)
            .collect();
        assert_eq!(tv.points.len(), inlier_corrs.len());
        let mut max_e = 0.0;
        for (x, c) in tv.points.iter().zip(&inlier_corrs) {
            if x.z > 0.0 {
                let e = (Vector2::new(x.x / x.z, x.y / x.z) - c.0).norm();
                max_e = f64::max(max_e, e);
            }
        }
        assert!(max_e < 1e-3, "reproj {max_e}");
    }

    #[test]
    fn essential_robust_to_noise_and_outliers() {
        let (mut corrs, r_gt, _) = scene(200, false, 0.0008);
        // 15% gross outliers (random garbage matches).
        let mut rng = Rng64::new(7);
        for k in 0..corrs.len() / 7 {
            corrs[k * 7].1 = Vector2::new(
                rng.next_u64() as f64 / u64::MAX as f64 - 0.5,
                rng.next_u64() as f64 / u64::MAX as f64 - 0.5,
            );
        }
        let tv = initialize(&corrs, InitOptions::default()).expect("init");
        assert!(rot_angle_deg(&tv.r, &r_gt) < 2.0, "R off under noise");
        let inl = tv.inliers.iter().filter(|&&b| b).count();
        assert!(inl > corrs.len() / 2, "too few inliers: {inl}");
    }

    #[test]
    fn planar_scene_selects_homography() {
        let (corrs, _, _) = scene(120, true, 0.0);
        let tv = initialize(&corrs, InitOptions::default()).expect("init");
        // A pure plane is degenerate for E; the H score should dominate.
        assert!(tv.r_h > 0.45, "R_H={} (expected homography)", tv.r_h);
        assert_eq!(tv.model, Model::Homography);
    }

    #[test]
    fn homography_is_exact_for_a_plane() {
        let (corrs, _, _) = scene(60, true, 0.0);
        let h = homography_4point(&corrs).expect("H");
        let hinv = h.try_inverse().unwrap();
        let max = corrs
            .iter()
            .map(|c| homography_err(&h, &hinv, c))
            .fold(0.0, f64::max);
        assert!(max < 1e-10, "planar H residual {max}");
    }

    #[test]
    fn essential_direct_is_exact() {
        // No RANSAC: E from all noise-free corrs must be machine-exact
        // (guards the `null_vector` / decomposition / triangulation
        // primitives independently of model selection).
        let (corrs, r_gt, t_gt) = scene(120, false, 0.0);
        let e = essential_8point(&corrs).expect("E");
        let max_err = corrs
            .iter()
            .map(|c| epipolar_err(&e, c))
            .fold(0.0, f64::max);
        assert!(max_err < 1e-12, "direct E inconsistent: {max_err:e}");
        let cands = decompose_essential(&e);
        let inl = vec![true; corrs.len()];
        let (r, t, pts) = best_pose(&cands, &corrs, &inl).expect("pose");
        assert!(rot_angle_deg(&r, &r_gt) < 1e-6);
        assert!(t.normalize().dot(&t_gt.normalize()).abs() > 1.0 - 1e-9);
        assert_eq!(pts.iter().filter(|p| p.z <= 0.0).count(), 0);
    }

    #[test]
    fn homography_recovers_pose_on_plane() {
        let (corrs, r_gt, t_gt) = scene(150, true, 0.0);
        let tv = initialize(&corrs, InitOptions::default()).expect("init");
        assert_eq!(tv.model, Model::Homography);
        let dr = rot_angle_deg(&tv.r, &r_gt);
        assert!(dr < 1.5, "R off on plane: {dr:.4}°");
        let cos = tv.t.normalize().dot(&t_gt.normalize()).abs();
        assert!(cos > 0.99, "t dir off on plane: cos={cos}");
    }

    #[test]
    fn too_few_points_is_none() {
        let (corrs, _, _) = scene(5, false, 0.0);
        assert!(initialize(&corrs, InitOptions::default()).is_none());
    }
}
