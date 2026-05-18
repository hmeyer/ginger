//! SO(3) / SE(3) exponential and logarithm maps.
//!
//! Hand-rolled (not in nalgebra, which is not a Lie-group library) so the
//! convention is pinned and tested rather than implicit. The matrix
//! arithmetic underneath is nalgebra.
//!
//! Convention: a twist is `xi = [rho; phi]` (`Vector6`), translation part
//! `rho` first, rotation part `phi` last. `se3_exp(xi)` yields
//! `T = [R | t]` with `R = so3_exp(phi)` and `t = V(phi) * rho`, where
//! `V` is the SO(3) left Jacobian. `se3_log` is its exact inverse. All
//! formulas carry a small-angle Taylor branch so they stay finite and
//! accurate as `theta -> 0`.

use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3, Vector6};

/// Below this rotation angle (rad) the closed forms divide by ~0, so we
/// switch to the Taylor expansion. ~1e-4 keeps f64 error < 1e-12.
const EPS: f64 = 1e-7;

/// `[w]_x` — the skew-symmetric cross-product matrix of `w`.
#[inline]
pub fn hat(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -w.z, w.y, //
        w.z, 0.0, -w.x, //
        -w.y, w.x, 0.0,
    )
}

/// Inverse of [`hat`]: pull the axis vector out of a skew matrix.
#[inline]
pub fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m.m32, m.m13, m.m21)
}

/// SO(3) exponential: rotation-vector (axis * angle) → rotation matrix
/// (Rodrigues).
pub fn so3_exp(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta2 = phi.norm_squared();
    if theta2 < EPS * EPS {
        // I + [phi]_x + ½[phi]_x²  (second order — exact to O(θ⁴)).
        let k = hat(phi);
        return Matrix3::identity() + k + 0.5 * k * k;
    }
    let theta = theta2.sqrt();
    let k = hat(&(phi / theta));
    Matrix3::identity() + theta.sin() * k + (1.0 - theta.cos()) * k * k
}

/// SO(3) logarithm: rotation matrix → rotation vector. Robust at
/// `theta ≈ 0` and `theta ≈ π`.
pub fn so3_log(r: &Matrix3<f64>) -> Vector3<f64> {
    let cos_theta = ((r.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < EPS {
        // sin θ ≈ θ → axis ≈ ½ vee(R - Rᵀ).
        return 0.5 * vee(&(r - r.transpose()));
    }
    if theta > std::f64::consts::PI - 1e-5 {
        // Near π: R - Rᵀ collapses; recover the axis from R + I, whose
        // columns are ∝ the rotation axis, then fix the sign.
        let a = r + Matrix3::identity();
        let mut axis = a.column(0).into_owned();
        if axis.norm_squared() < 1e-9 {
            axis = a.column(1).into_owned();
        }
        if axis.norm_squared() < 1e-9 {
            axis = a.column(2).into_owned();
        }
        let axis = axis.normalize();
        let v = axis * theta;
        // Sign: vee(R - Rᵀ) points along +axis when θ < π.
        let w = vee(&(r - r.transpose()));
        return if w.dot(&axis) >= 0.0 { v } else { -v };
    }
    (theta / (2.0 * theta.sin())) * vee(&(r - r.transpose()))
}

/// SO(3) left Jacobian `V(phi)` and its inverse — the `exp`/`log`
/// translation coupling for SE(3).
fn so3_left_jacobian(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta2 = phi.norm_squared();
    let k = hat(phi);
    if theta2 < EPS * EPS {
        return Matrix3::identity() + 0.5 * k + (1.0 / 6.0) * k * k;
    }
    let theta = theta2.sqrt();
    let a = (1.0 - theta.cos()) / theta2;
    let b = (theta - theta.sin()) / (theta2 * theta);
    Matrix3::identity() + a * k + b * k * k
}

fn so3_left_jacobian_inv(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta2 = phi.norm_squared();
    let k = hat(phi);
    if theta2 < EPS * EPS {
        return Matrix3::identity() - 0.5 * k + (1.0 / 12.0) * k * k;
    }
    let theta = theta2.sqrt();
    // c = 1/θ² - (1 + cosθ) / (2θ sinθ)
    let c = 1.0 / theta2 - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
    Matrix3::identity() - 0.5 * k + c * k * k
}

/// SE(3) exponential: twist `[rho; phi]` → rigid transform.
pub fn se3_exp(xi: &Vector6<f64>) -> Isometry3<f64> {
    let rho = Vector3::new(xi[0], xi[1], xi[2]);
    let phi = Vector3::new(xi[3], xi[4], xi[5]);
    let r = so3_exp(&phi);
    let t = so3_left_jacobian(&phi) * rho;
    let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Isometry3::from_parts(Translation3::from(t), rot)
}

/// SE(3) logarithm: rigid transform → twist `[rho; phi]` (inverse of
/// [`se3_exp`]).
pub fn se3_log(t: &Isometry3<f64>) -> Vector6<f64> {
    let r = t.rotation.to_rotation_matrix();
    let phi = so3_log(r.matrix());
    let rho = so3_left_jacobian_inv(&phi) * t.translation.vector;
    Vector6::new(rho.x, rho.y, rho.z, phi.x, phi.y, phi.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn hat_vee_roundtrip() {
        let w = Vector3::new(0.3, -1.1, 2.0);
        assert!((vee(&hat(&w)) - w).norm() < 1e-15);
    }

    #[test]
    fn so3_exp_log_roundtrip_general_and_tiny() {
        for phi in [
            Vector3::new(0.4, -0.2, 1.3),
            Vector3::new(1e-9, 0.0, -2e-9), // Taylor branch
            Vector3::new(0.0, 0.0, 0.0),
        ] {
            let r = so3_exp(&phi);
            // Valid rotation.
            assert!((r * r.transpose() - Matrix3::identity()).norm() < 1e-12);
            assert!(approx(r.determinant(), 1.0, 1e-12));
            // exp∘log identity.
            assert!((so3_log(&r) - phi).norm() < 1e-9, "phi={phi:?}");
        }
    }

    #[test]
    fn so3_log_handles_pi() {
        // 180° about a tilted axis — the degenerate branch.
        let axis = Vector3::new(1.0, 0.5, -0.3).normalize();
        let phi = axis * PI;
        let r = so3_exp(&phi);
        let back = so3_log(&r);
        // ±phi are equivalent at exactly π; compare the rotations.
        assert!((so3_exp(&back) - r).norm() < 1e-9);
    }

    #[test]
    fn se3_exp_log_roundtrip() {
        for xi in [
            Vector6::new(0.5, -1.2, 3.0, 0.2, -0.7, 0.9),
            Vector6::new(1.0, 2.0, -0.5, 1e-10, 0.0, -1e-10), // tiny rotation
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ] {
            let t = se3_exp(&xi);
            assert!((se3_log(&t) - xi).norm() < 1e-8, "xi={xi:?}");
        }
    }

    #[test]
    fn se3_exp_matches_manual_pure_translation() {
        let xi = Vector6::new(1.0, -2.0, 3.0, 0.0, 0.0, 0.0);
        let t = se3_exp(&xi);
        assert!((t.translation.vector - Vector3::new(1.0, -2.0, 3.0)).norm() < 1e-15);
        assert!((t.rotation.angle()).abs() < 1e-15);
    }

    #[test]
    fn se3_compose_inverse_is_identity() {
        let xi = Vector6::new(0.3, 0.1, -0.4, 0.2, 0.5, -0.1);
        let t = se3_exp(&xi);
        let i = t * t.inverse();
        assert!(se3_log(&i).norm() < 1e-12);
    }
}
