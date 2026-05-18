//! Intrinsics config: a serde/TOML record plus the rev 1.3 FOV-derived
//! prior used until a real calibration exists.
//!
//! The prior is derived from the standard Pi Camera "rev 1.3" lens
//! angular field of view, so it is **resolution-agnostic** for the
//! libcamera `ViewFinder` full-FOV mode (the stream resolution is read
//! at runtime, not fixed). It is explicitly flagged `verified = false`
//! so nothing mistakes the placeholder for a calibration; a ChArUco
//! calibration later overwrites the file with `verified = true`.

use serde::{Deserialize, Serialize};

use crate::camera::CameraModel;

/// Standard Pi Camera v1 / rev 1.3 lens, widely-cited angular FOV.
const REV1_3_HFOV_DEG: f64 = 53.5;
const REV1_3_VFOV_DEG: f64 = 41.4;

/// Serializable intrinsics record (the on-disk `slam.toml`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
    #[serde(default)]
    pub k3: f64,
    pub width: u32,
    pub height: u32,
    /// Free-text source, e.g. the prior tag or a calibration date.
    pub model: String,
    /// `false` for the derived prior, `true` only after a real
    /// calibration. The WebUI surfaces this as an `UNVERIFIED` badge.
    pub verified: bool,
}

impl Intrinsics {
    /// Derive the rev 1.3 standard-lens prior for a given stream
    /// resolution (pinhole, zero distortion, `verified = false`).
    pub fn rev1_3_prior(width: u32, height: u32) -> Self {
        let half = |fov_deg: f64| (fov_deg.to_radians() * 0.5).tan();
        let fx = (width as f64 * 0.5) / half(REV1_3_HFOV_DEG);
        let fy = (height as f64 * 0.5) / half(REV1_3_VFOV_DEG);
        Self {
            fx,
            fy,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
            width,
            height,
            model: "pi-cam-rev1.3 (FOV-derived prior, UNVERIFIED)".into(),
            verified: false,
        }
    }

    /// Parse from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Serialize to a TOML string.
    pub fn to_toml_string(&self) -> String {
        toml::to_string_pretty(self).expect("intrinsics serialize")
    }

    /// Horizontal field of view (degrees) implied by `fx`/`width`.
    pub fn hfov_deg(&self) -> f64 {
        2.0 * ((self.width as f64 * 0.5) / self.fx).atan().to_degrees()
    }

    /// Build the runtime [`CameraModel`].
    pub fn to_camera_model(&self) -> CameraModel {
        CameraModel {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            k1: self.k1,
            k2: self.k2,
            p1: self.p1,
            p2: self.p2,
            k3: self.k3,
            width: self.width,
            height: self.height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rev1_3_prior_matches_plan_numbers_at_800x600() {
        let k = Intrinsics::rev1_3_prior(800, 600);
        // Plan: fx ≈ fy ≈ 794, cx = 400, cy = 300.
        assert!((k.fx - 794.0).abs() < 3.0, "fx={}", k.fx);
        assert!((k.fy - 794.0).abs() < 3.0, "fy={}", k.fy);
        assert_eq!((k.cx, k.cy), (400.0, 300.0));
        assert!(!k.verified);
        // FOV is recovered (resolution-agnostic property).
        assert!((k.hfov_deg() - REV1_3_HFOV_DEG).abs() < 1e-9);
    }

    #[test]
    fn prior_fov_is_resolution_agnostic() {
        let a = Intrinsics::rev1_3_prior(800, 600);
        let b = Intrinsics::rev1_3_prior(1640, 1232);
        assert!((a.hfov_deg() - b.hfov_deg()).abs() < 1e-9);
    }

    #[test]
    fn toml_roundtrips() {
        let k = Intrinsics::rev1_3_prior(800, 600);
        let parsed = Intrinsics::from_toml_str(&k.to_toml_string()).unwrap();
        assert_eq!(k, parsed);
    }

    #[test]
    fn distortion_fields_default_when_absent() {
        let toml = r#"
            fx = 800.0
            fy = 800.0
            cx = 400.0
            cy = 300.0
            width = 800
            height = 600
            model = "calib-2026"
            verified = true
        "#;
        let k = Intrinsics::from_toml_str(toml).unwrap();
        assert_eq!((k.k1, k.k2, k.p1, k.p2, k.k3), (0.0, 0.0, 0.0, 0.0, 0.0));
        assert!(k.verified);
    }
}
