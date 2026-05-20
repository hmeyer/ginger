//! Frame-to-frame tracking (M4): a constant-velocity motion model plus
//! **motion-only bundle adjustment** — refine the 6-DOF camera pose that
//! best reprojects known 3D map points onto their observed (calibrated)
//! image points, robust to outliers.
//!
//! Camera-free and on calibrated/normalized coordinates (same contract
//! as [`crate::twoview`]). The pose is `T_cw` (world → camera):
//! `x_cam = R·X_world + t`. It rides the hardened
//! [`crate::optimize::levenberg_marquardt`] (SE3 parameterized via
//! [`crate::lie`], Huber robustifier). The Jacobian is central finite
//! differences — exact enough and convention-proof; an analytic
//! Jacobian is a later, parity-tested perf step.

use nalgebra::{DMatrix, DVector, Isometry3, Vector2, Vector3, Vector6};

use crate::lie::{se3_exp, se3_log};
use crate::optimize::{LeastSquaresProblem, LmOptions, levenberg_marquardt};

/// Predict the next pose assuming the last inter-frame motion repeats
/// (constant velocity). With `T_cw` poses the relative motion is
/// `V = T_prev · T_prev_prev⁻¹`, applied again: `T_pred = V · T_prev`.
pub fn constant_velocity(prev_prev: &Isometry3<f64>, prev: &Isometry3<f64>) -> Isometry3<f64> {
    let v = prev * prev_prev.inverse();
    v * prev
}

/// One 3D↔2D constraint: a map point (world frame) and its observed
/// calibrated image point in the current frame.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub point: Vector3<f64>,
    pub obs: Vector2<f64>,
}

/// Motion-only BA problem: params are the SE3 twist `xi` with
/// `T_cw = exp(xi)`; residuals are the per-observation reprojection
/// errors (2 per point). Points behind the camera get a bounded penalty
/// so a bad initial guess still has a smooth descent direction.
struct MotionOnlyBa<'a> {
    obs: &'a [Observation],
}

impl MotionOnlyBa<'_> {
    fn pose(p: &DVector<f64>) -> Isometry3<f64> {
        se3_exp(&Vector6::new(p[0], p[1], p[2], p[3], p[4], p[5]))
    }
}

impl LeastSquaresProblem for MotionOnlyBa<'_> {
    fn residuals(&self, p: &DVector<f64>) -> DVector<f64> {
        let t = Self::pose(p);
        let mut r = DVector::zeros(self.obs.len() * 2);
        for (i, o) in self.obs.iter().enumerate() {
            let pc = t.rotation * o.point + t.translation.vector;
            // Behind / near the camera: project at a clamped depth so
            // the residual stays finite and keeps a usable gradient.
            let z = if pc.z > 1e-3 { pc.z } else { 1e-3 };
            r[2 * i] = pc.x / z - o.obs.x;
            r[2 * i + 1] = pc.y / z - o.obs.y;
        }
        r
    }

    fn jacobian(&self, p: &DVector<f64>) -> DMatrix<f64> {
        // Central finite differences (6 params → 12 residual evals).
        let n = self.obs.len() * 2;
        let mut j = DMatrix::zeros(n, 6);
        let eps = 1e-6;
        for k in 0..6 {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[k] += eps;
            pm[k] -= eps;
            let rp = self.residuals(&pp);
            let rm = self.residuals(&pm);
            for row in 0..n {
                j[(row, k)] = (rp[row] - rm[row]) / (2.0 * eps);
            }
        }
        j
    }
}

/// Outcome of [`track_pose`].
#[derive(Clone, Debug)]
pub struct TrackReport {
    /// Refined `T_cw` (world → camera).
    pub pose: Isometry3<f64>,
    /// Per-observation final reprojection error norm (calibrated units).
    pub errors: Vec<f64>,
    /// Observations whose final error is within `inlier_thresh`.
    pub inliers: Vec<bool>,
    pub n_inliers: usize,
    pub converged: bool,
}

/// Refine `init` (e.g. the constant-velocity prediction) by motion-only
/// BA against `obs`. `huber` and `inlier_thresh` are in calibrated
/// units (≈ pixel / focal length).
pub fn track_pose(
    obs: &[Observation],
    init: &Isometry3<f64>,
    huber: f64,
    inlier_thresh: f64,
) -> Option<TrackReport> {
    if obs.len() < 3 {
        return None;
    }
    let prob = MotionOnlyBa { obs };
    let x0 = se3_log(init);
    let report = levenberg_marquardt(
        &prob,
        DVector::from_vec(x0.as_slice().to_vec()),
        LmOptions {
            max_iters: 40,
            huber_delta: Some(huber),
            ..LmOptions::default()
        },
    );
    let pose = MotionOnlyBa::pose(&report.params);
    let r = prob.residuals(&report.params);
    let mut errors = Vec::with_capacity(obs.len());
    let mut inliers = Vec::with_capacity(obs.len());
    let mut n_inliers = 0;
    for i in 0..obs.len() {
        let e = (r[2 * i].powi(2) + r[2 * i + 1].powi(2)).sqrt();
        let ok = e <= inlier_thresh;
        if ok {
            n_inliers += 1;
        }
        errors.push(e);
        inliers.push(ok);
    }
    Some(TrackReport {
        pose,
        errors,
        inliers,
        n_inliers,
        converged: report.converged,
    })
}

#[cfg(test)]
mod tests {
    use nalgebra::{Rotation3, Translation3, UnitQuaternion};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;

    fn iso(rx: f64, ry: f64, rz: f64, tx: f64, ty: f64, tz: f64) -> Isometry3<f64> {
        let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(rx, ry, rz));
        Isometry3::from_parts(Translation3::new(tx, ty, tz), rot)
    }

    /// Deterministic point cloud + a ground-truth pose; project to
    /// calibrated observations.
    fn scene(n: usize, pose: &Isometry3<f64>) -> Vec<Observation> {
        let mut r = SmallRng::seed_from_u64(1234567);
        (0..n)
            .map(|_| {
                let point = Vector3::new(
                    (r.random::<f64>() - 0.5) * 4.0,
                    (r.random::<f64>() - 0.5) * 3.0,
                    2.0 + r.random::<f64>() * 4.0,
                );
                let pc = pose.rotation * point + pose.translation.vector;
                Observation {
                    point,
                    obs: Vector2::new(pc.x / pc.z, pc.y / pc.z),
                }
            })
            .collect()
    }

    fn pose_err(a: &Isometry3<f64>, b: &Isometry3<f64>) -> f64 {
        se3_log(&(a.inverse() * b)).norm()
    }

    #[test]
    fn constant_velocity_is_exact_on_constant_motion() {
        let step = iso(0.01, -0.02, 0.015, 0.1, -0.05, 0.2);
        let p0 = iso(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p1 = step * p0;
        let p2 = step * p1;
        let pred = constant_velocity(&p0, &p1);
        assert!(pose_err(&pred, &p2) < 1e-9, "Δ={}", pose_err(&pred, &p2));
    }

    #[test]
    fn recovers_pose_noise_free() {
        let gt = iso(0.05, -0.08, 0.03, 0.3, -0.15, 0.1);
        let obs = scene(80, &gt);
        // Start from a noticeably perturbed guess.
        let init = iso(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let rep = track_pose(&obs, &init, 1.0, 1e-3).expect("track");
        assert!(rep.converged);
        assert!(
            pose_err(&rep.pose, &gt) < 1e-6,
            "pose Δ={}",
            pose_err(&rep.pose, &gt)
        );
        assert_eq!(rep.n_inliers, obs.len());
    }

    #[test]
    fn huber_rejects_outliers() {
        let gt = iso(-0.04, 0.06, 0.02, -0.2, 0.1, 0.25);
        let mut obs = scene(120, &gt);
        // Corrupt 15% with gross wrong observations.
        for o in obs.iter_mut().step_by(7) {
            o.obs = Vector2::new(o.obs.x + 0.4, o.obs.y - 0.35);
        }
        // A realistic motion-model prediction: slightly off ground truth.
        let init = se3_exp(&Vector6::new(0.01, -0.01, 0.02, 0.015, 0.02, -0.01)) * gt;
        // Robust (small Huber δ) vs effectively plain LS (huge δ → all
        // weights 1) on the *same* data isolates the robustifier.
        let robust = track_pose(&obs, &init, 0.01, 0.02).expect("robust");
        let plain = track_pose(&obs, &init, 1e9, 0.02).expect("plain");
        let er = pose_err(&robust.pose, &gt);
        let ep = pose_err(&plain.pose, &gt);
        assert!(er < 0.02, "robust pose Δ={er}");
        assert!(er < 0.4 * ep, "Huber didn't help: robust={er} plain={ep}");
        let outliers = robust.inliers.iter().filter(|&&b| !b).count();
        assert!(
            outliers >= obs.len() / 8,
            "flagged too few outliers: {outliers}"
        );
    }

    #[test]
    fn too_few_points_is_none() {
        let gt = iso(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert!(track_pose(&scene(2, &gt), &gt, 1.0, 1e-3).is_none());
    }
}
