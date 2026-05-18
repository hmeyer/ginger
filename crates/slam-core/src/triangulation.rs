//! Gated two-view triangulation (M5-1c): turn a calibrated
//! correspondence between two keyframes into a world point, *only* if it
//! is geometrically trustworthy.
//!
//! Camera-free; calibrated/normalized observations and `T_cw` poses
//! (same contract as `twoview`/`tracking`/`local_ba`). The local mapper
//! (M5-2) calls this for inter-keyframe correspondences that aren't yet
//! map points; the gates (positive depth in *both* views, enough
//! parallax, low symmetric reprojection error) keep degenerate /
//! low-parallax matches out of the map.

use nalgebra::{Isometry3, Matrix4, Vector2, Vector3};

/// Acceptance gates. Defaults are ORB-SLAM-ish for calibrated coords.
#[derive(Clone, Copy, Debug)]
pub struct TriangulateOptions {
    /// Minimum parallax angle between the two viewing rays (radians).
    pub min_parallax_rad: f64,
    /// Max symmetric reprojection error (calibrated units ≈ px/focal).
    pub max_reproj: f64,
}

impl Default for TriangulateOptions {
    fn default() -> Self {
        Self {
            min_parallax_rad: 1.0_f64.to_radians(),
            max_reproj: 4.0 / 500.0, // ~4 px at f≈500
        }
    }
}

/// Smallest-eigenvector solution of `A x = 0` via `AᵀA` (robust for the
/// 4×4 DLT system; same approach as `twoview::null_vector`).
fn null4(a: &Matrix4<f64>) -> Vector3<f64> {
    let se = (a.transpose() * a).symmetric_eigen();
    let mut mi = 0;
    for i in 1..4 {
        if se.eigenvalues[i] < se.eigenvalues[mi] {
            mi = i;
        }
    }
    let v = se.eigenvectors.column(mi);
    Vector3::new(v[0] / v[3], v[1] / v[3], v[2] / v[3])
}

fn project(t: &Isometry3<f64>, x: &Vector3<f64>) -> Option<Vector2<f64>> {
    let pc = t.rotation * x + t.translation.vector;
    (pc.z > 1e-6).then(|| Vector2::new(pc.x / pc.z, pc.y / pc.z))
}

/// Triangulate the world point seen at calibrated `obs_a` in keyframe
/// `t_a` (`T_cw`) and `obs_b` in `t_b`. `None` unless it is in front of
/// both cameras, the rays have ≥ `min_parallax` angle, and the
/// symmetric reprojection error is ≤ `max_reproj`.
pub fn triangulate(
    t_a: &Isometry3<f64>,
    t_b: &Isometry3<f64>,
    obs_a: Vector2<f64>,
    obs_b: Vector2<f64>,
    opt: TriangulateOptions,
) -> Option<Vector3<f64>> {
    // P = [R | t] (calibrated). DLT rows: u·P₃ − P₁, v·P₃ − P₂.
    let row = |t: &Isometry3<f64>, o: Vector2<f64>| {
        let r = t.rotation.to_rotation_matrix();
        let tr = t.translation.vector;
        let p = |i: usize| [r[(i, 0)], r[(i, 1)], r[(i, 2)], tr[i]];
        let (p0, p1, p2) = (p(0), p(1), p(2));
        let mut a = [[0.0; 4]; 2];
        for k in 0..4 {
            a[0][k] = o.x * p2[k] - p0[k];
            a[1][k] = o.y * p2[k] - p1[k];
        }
        a
    };
    let ra = row(t_a, obs_a);
    let rb = row(t_b, obs_b);
    let a = Matrix4::new(
        ra[0][0], ra[0][1], ra[0][2], ra[0][3], //
        ra[1][0], ra[1][1], ra[1][2], ra[1][3], //
        rb[0][0], rb[0][1], rb[0][2], rb[0][3], //
        rb[1][0], rb[1][1], rb[1][2], rb[1][3],
    );
    let x = null4(&a);
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Cheirality: in front of both cameras.
    let za = (t_a.rotation * x + t_a.translation.vector).z;
    let zb = (t_b.rotation * x + t_b.translation.vector).z;
    if za <= 0.0 || zb <= 0.0 {
        return None;
    }

    // Parallax: angle between the two viewing rays from each centre.
    let ca = t_a.inverse().translation.vector;
    let cb = t_b.inverse().translation.vector;
    let ra_v = (x - ca).normalize();
    let rb_v = (x - cb).normalize();
    let cos = ra_v.dot(&rb_v).clamp(-1.0, 1.0);
    if cos.acos() < opt.min_parallax_rad {
        return None;
    }

    // Symmetric reprojection error.
    let pa = project(t_a, &x)?;
    let pb = project(t_b, &x)?;
    if (pa - obs_a).norm() > opt.max_reproj || (pb - obs_b).norm() > opt.max_reproj {
        return None;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    fn pose(tx: f64) -> Isometry3<f64> {
        // Tcw for a camera whose centre is at (tx,0,0), looking +z.
        Isometry3::from_parts(Translation3::new(-tx, 0.0, 0.0), UnitQuaternion::identity())
    }
    fn obs(t: &Isometry3<f64>, x: &Vector3<f64>) -> Vector2<f64> {
        let pc = t.rotation * x + t.translation.vector;
        Vector2::new(pc.x / pc.z, pc.y / pc.z)
    }

    #[test]
    fn recovers_point_with_baseline() {
        let (ta, tb) = (pose(0.0), pose(0.6));
        let x = Vector3::new(0.4, -0.3, 5.0);
        let got = triangulate(
            &ta,
            &tb,
            obs(&ta, &x),
            obs(&tb, &x),
            TriangulateOptions::default(),
        )
        .expect("accepted");
        assert!((got - x).norm() < 1e-9, "got {got:?}");
    }

    #[test]
    fn rejects_low_parallax() {
        // Tiny baseline vs depth 5 → ~0.06° parallax ≪ 1° gate.
        let (ta, tb) = (pose(0.0), pose(0.005));
        let x = Vector3::new(0.1, 0.0, 5.0);
        assert!(
            triangulate(
                &ta,
                &tb,
                obs(&ta, &x),
                obs(&tb, &x),
                TriangulateOptions::default()
            )
            .is_none()
        );
    }

    #[test]
    fn rejects_behind_camera() {
        // Point behind: feed the (degenerate) projections of a -z point.
        let (ta, tb) = (pose(0.0), pose(0.6));
        let x = Vector3::new(0.2, 0.1, 5.0);
        // Swap observations → triangulated point ends up inconsistent /
        // behind one camera, must be rejected (no panic, None).
        let r = triangulate(
            &ta,
            &tb,
            obs(&tb, &x),
            obs(&ta, &x),
            TriangulateOptions::default(),
        );
        assert!(r.is_none());
    }

    #[test]
    fn rejects_gross_mismatch() {
        // A *wrong* correspondence: obs_a is the true projection but
        // obs_b points somewhere unrelated. Two views always admit a
        // best-fit point for small noise (it just moves), so the gate
        // that matters is rejecting gross mismatches — the rays are
        // inconsistent → cheirality / reprojection fails.
        let (ta, tb) = (pose(0.0), pose(0.6));
        let x = Vector3::new(0.0, 0.0, 5.0);
        let bad = Vector2::new(0.6, 0.5);
        assert!(triangulate(&ta, &tb, obs(&ta, &x), bad, TriangulateOptions::default()).is_none());
    }
}
