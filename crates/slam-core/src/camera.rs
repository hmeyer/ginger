//! Pinhole camera with Brown–Conrady (radial-tangential) distortion.
//!
//! `project` maps a point in the camera frame to a distorted pixel;
//! `unproject` maps a pixel back to a unit bearing; `undistort_point`
//! maps a distorted pixel to the ideal pinhole pixel it would have had
//! with zero distortion (what M3+ geometry consumes). The distortion
//! inverse has no closed form, so the two backward maps iterate
//! (Newton-free fixed point — the standard OpenCV scheme); it converges
//! in a handful of steps for sane lens coefficients.

use nalgebra::{Vector2, Vector3};

/// Pinhole intrinsics + 5 Brown–Conrady coefficients. Camera-frame
/// convention: `+z` forward, points with `z <= 0` do not project.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
    pub width: u32,
    pub height: u32,
}

impl CameraModel {
    /// Apply Brown–Conrady to a normalized image point `(x, y) = (X/Z, Y/Z)`.
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
        let xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (xd, yd)
    }

    /// Iteratively recover the undistorted normalized point from a
    /// distorted one (OpenCV's `undistortPoints` fixed point).
    fn undistort_normalized(&self, xd: f64, yd: f64) -> (f64, f64) {
        let (mut x, mut y) = (xd, yd);
        for _ in 0..20 {
            let r2 = x * x + y * y;
            let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
            let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            x = (xd - dx) / radial;
            y = (yd - dy) / radial;
        }
        (x, y)
    }

    /// Camera-frame point → distorted pixel. `None` if behind the camera.
    pub fn project(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p.z <= 1e-9 {
            return None;
        }
        let (xd, yd) = self.distort(p.x / p.z, p.y / p.z);
        Some(Vector2::new(self.fx * xd + self.cx, self.fy * yd + self.cy))
    }

    /// Distorted pixel → unit bearing in the camera frame.
    pub fn unproject(&self, px: &Vector2<f64>) -> Vector3<f64> {
        let xd = (px.x - self.cx) / self.fx;
        let yd = (px.y - self.cy) / self.fy;
        let (x, y) = self.undistort_normalized(xd, yd);
        Vector3::new(x, y, 1.0).normalize()
    }

    /// Distorted pixel → the ideal pinhole pixel (zero-distortion `K`).
    pub fn undistort_point(&self, px: &Vector2<f64>) -> Vector2<f64> {
        let xd = (px.x - self.cx) / self.fx;
        let yd = (px.y - self.cy) / self.fy;
        let (x, y) = self.undistort_normalized(xd, yd);
        Vector2::new(self.fx * x + self.cx, self.fy * y + self.cy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinhole() -> CameraModel {
        CameraModel {
            fx: 794.0,
            fy: 794.0,
            cx: 400.0,
            cy: 300.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
            width: 800,
            height: 600,
        }
    }

    #[test]
    fn principal_axis_hits_principal_point() {
        let c = pinhole();
        let px = c.project(&Vector3::new(0.0, 0.0, 5.0)).unwrap();
        assert!((px - Vector2::new(400.0, 300.0)).norm() < 1e-9);
    }

    #[test]
    fn behind_camera_does_not_project() {
        assert!(pinhole().project(&Vector3::new(0.1, 0.1, 0.0)).is_none());
        assert!(pinhole().project(&Vector3::new(0.1, 0.1, -2.0)).is_none());
    }

    #[test]
    fn zero_distortion_unproject_is_exact_inverse() {
        let c = pinhole();
        for p in [
            Vector3::new(0.2, -0.1, 1.0),
            Vector3::new(-0.4, 0.3, 2.0),
            Vector3::new(0.0, 0.0, 1.0),
        ] {
            let px = c.project(&p).unwrap();
            let bearing = c.unproject(&px);
            // Same direction as the original ray.
            assert!((bearing - p.normalize()).norm() < 1e-9);
        }
    }

    #[test]
    fn distortion_roundtrips_through_undistort() {
        let mut c = pinhole();
        c.k1 = -0.28;
        c.k2 = 0.10;
        c.p1 = 0.001;
        c.p2 = -0.0007;
        c.k3 = 0.004;
        // A point well off-axis where distortion is non-trivial.
        let p = Vector3::new(0.35, -0.25, 1.0);
        let px = c.project(&p).unwrap();
        // unproject must recover the original bearing despite distortion.
        let bearing = c.unproject(&px);
        assert!((bearing - p.normalize()).norm() < 1e-7);
        // undistort_point then re-distort returns the original pixel.
        let ideal = c.undistort_point(&px);
        let xi = (ideal.x - c.cx) / c.fx;
        let yi = (ideal.y - c.cy) / c.fy;
        let (xd, yd) = c.distort(xi, yi);
        let redist = Vector2::new(c.fx * xd + c.cx, c.fy * yd + c.cy);
        assert!((redist - px).norm() < 1e-6);
    }

    #[test]
    fn undistort_is_identity_without_distortion() {
        let c = pinhole();
        let px = Vector2::new(123.0, 456.0);
        assert!((c.undistort_point(&px) - px).norm() < 1e-9);
    }
}
