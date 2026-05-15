//! Software auto-exposure controller.
//!
//! Pure logic — no camera I/O — so it can be unit-tested in isolation.
//!
//! Brightness axis: a single 1D control variable in stops above the darkest
//! possible setting. 1 stop = 2× the light hitting the sensor.
//!   brightness 0    → gain=1×,  exp=0.5 ms       (darkest)
//!   brightness 4    → gain=16×, exp=0.5 ms       (gain saturated)
//!   brightness 11.6 → gain=16×, exp=100 ms       (brightest)
//! The mapping enforces gain-first ramping: gain saturates before exposure
//! extends, keeping motion blur bounded for as long as possible.

const AE_TARGET_LUMA: u8 = 128;
const AE_DEADZONE: f32 = 8.0; // luma counts; below this, no step
const AE_MIN_EXPOSURE_US: i32 = 500;
const AE_MAX_EXPOSURE_US: i32 = 100_000;
const AE_MIN_GAIN: f32 = 1.0;
const AE_MAX_GAIN: f32 = 16.0;
const AE_GAMMA: f32 = 2.2; // approximate luma → linear-light gamma
const AE_STEP_PER_LUMA: f32 = 1.0 / 80.0; // stops of correction per luma count of error
const AE_MAX_BRIGHTNESS: f32 = 11.64; // log2(100000/500) + log2(16/1)

/// Map controller brightness (stops) → (exposure_us, gain) with gain-first ramp.
pub fn brightness_to_settings(b: f32) -> (i32, f32) {
    let b = b.clamp(0.0, AE_MAX_BRIGHTNESS);
    if b <= 4.0 {
        (AE_MIN_EXPOSURE_US, 2.0_f32.powf(b))
    } else {
        let exp = AE_MIN_EXPOSURE_US as f32 * 2.0_f32.powf(b - 4.0);
        ((exp as i32).min(AE_MAX_EXPOSURE_US), AE_MAX_GAIN)
    }
}

/// Inverse of `brightness_to_settings`: read back brightness from sensor metadata.
pub fn settings_to_brightness(exp_us: i32, gain: f32) -> f32 {
    (exp_us.max(AE_MIN_EXPOSURE_US) as f32 / AE_MIN_EXPOSURE_US as f32).log2()
        + (gain.max(AE_MIN_GAIN) / AE_MIN_GAIN).log2()
}

/// Live AE readback. Camera thread writes; UI/SSE reads. No knobs — AE runs
/// unconditionally with hardcoded constants above.
pub struct ExposureConfig {
    /// Controller's intended brightness (stops). Steps each frame.
    brightness: f32,
    /// Live readback derived from frame metadata each frame.
    pub current_exposure_us: i32,
    pub current_gain: f32,
    pub current_brightness: f32,
    pub current_luma: u8,
    /// Damping safety net.
    step_scale: f32,
    last_step_dir: i8,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        let init_b = settings_to_brightness(8_000, 8.0);
        Self {
            brightness: init_b,
            current_exposure_us: 8_000,
            current_gain: 8.0,
            current_brightness: init_b,
            current_luma: 0,
            step_scale: 1.0,
            last_step_dir: 0,
        }
    }
}

/// Sample mean luma from YUYV data. Y bytes are at even indices; stepping by
/// 128 always lands on a Y byte and yields ~7 500 samples for 800×600.
pub fn mean_luma(data: &[u8]) -> u8 {
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    let mut i = 0;
    while i < data.len() {
        sum += data[i] as u64;
        count += 1;
        i += 128;
    }
    sum.checked_div(count).map(|v| v as u8).unwrap_or(128)
}

/// Run one AE step using the brightness axis + Smith-Predictor luma estimate.
///
/// The luma we measure reflects `applied_brightness` (from frame metadata),
/// not our controller's current target. We predict what luma we'd see if
/// our latest brightness target were already applied, then step against
/// that predicted error. This handles the ~3-frame libcamera pipeline delay
/// without waiting and without overshoot.
pub fn ae_step(cfg: &mut ExposureConfig, luma: u8, applied_brightness: f32) -> (i32, f32) {
    cfg.current_luma = luma;
    cfg.current_brightness = applied_brightness;

    let in_flight = cfg.brightness - applied_brightness;
    let predicted_luma = luma as f32 * 2.0_f32.powf(in_flight / AE_GAMMA);
    let error = predicted_luma - AE_TARGET_LUMA as f32;

    let dir: i8 = if error < -AE_DEADZONE {
        1
    } else if error > AE_DEADZONE {
        -1
    } else {
        0
    };

    if dir != 0 {
        if dir == cfg.last_step_dir {
            cfg.step_scale = (cfg.step_scale * 1.1).min(1.0);
        } else if cfg.last_step_dir != 0 {
            cfg.step_scale = (cfg.step_scale * 0.5).max(0.1);
        }
        cfg.last_step_dir = dir;

        let step_stops = -error * AE_STEP_PER_LUMA * cfg.step_scale;
        cfg.brightness = (cfg.brightness + step_stops).clamp(0.0, AE_MAX_BRIGHTNESS);
    } else {
        cfg.last_step_dir = 0;
    }

    brightness_to_settings(cfg.brightness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brightness_round_trips_through_settings() {
        // settings_to_brightness ∘ brightness_to_settings ≈ identity
        for &b in &[0.0_f32, 1.0, 3.5, 4.0, 6.0, 10.0, 11.6] {
            let (exp, gain) = brightness_to_settings(b);
            let back = settings_to_brightness(exp, gain);
            assert!((back - b).abs() < 0.05, "b={b} round-tripped to {back}");
        }
    }

    #[test]
    fn gain_ramps_before_exposure() {
        // Below 4 stops: exposure pinned at minimum, gain climbs.
        let (exp_lo, gain_lo) = brightness_to_settings(2.0);
        assert_eq!(exp_lo, 500);
        assert!((gain_lo - 4.0).abs() < 0.01);
        // Above 4 stops: gain pinned at max, exposure climbs.
        let (exp_hi, gain_hi) = brightness_to_settings(6.0);
        assert!((gain_hi - 16.0).abs() < 0.01);
        assert!(exp_hi > 500);
    }

    #[test]
    fn settings_clamp_to_envelope() {
        // Far above max brightness: gain saturated, exposure at the ceiling
        // (AE_MAX_BRIGHTNESS rounds just under the 100 ms hard cap).
        let (exp, gain) = brightness_to_settings(1000.0);
        assert!(exp <= AE_MAX_EXPOSURE_US, "exp {exp} exceeds envelope");
        assert!(exp > 90_000, "exp {exp} not near the ceiling");
        assert!((gain - AE_MAX_GAIN).abs() < 0.01);
        // Below min brightness: darkest possible setting.
        let (exp, gain) = brightness_to_settings(-5.0);
        assert_eq!(exp, AE_MIN_EXPOSURE_US);
        assert!((gain - AE_MIN_GAIN).abs() < 0.01);
    }

    #[test]
    fn ae_step_holds_inside_deadzone() {
        let mut cfg = ExposureConfig::default();
        let b0 = cfg.current_brightness;
        // luma already at target, nothing in flight → no correction.
        ae_step(&mut cfg, 128, b0);
        assert_eq!(cfg.last_step_dir, 0);
    }

    #[test]
    fn ae_step_brightens_a_dark_frame() {
        let mut cfg = ExposureConfig::default();
        let applied = cfg.current_brightness;
        let (exp_before, gain_before) = brightness_to_settings(applied);
        let (exp_after, gain_after) = ae_step(&mut cfg, 20, applied);
        // Dark frame (luma 20 ≪ 128) must raise the light gathered.
        assert!(
            (exp_after, gain_after as i32) > (exp_before, gain_before as i32)
                || gain_after > gain_before,
            "expected brighter settings, got exp {exp_before}->{exp_after} gain {gain_before}->{gain_after}"
        );
        assert_eq!(cfg.last_step_dir, 1);
    }

    #[test]
    fn mean_luma_of_uniform_buffer() {
        let buf = vec![100u8; 4096];
        assert_eq!(mean_luma(&buf), 100);
    }

    #[test]
    fn mean_luma_empty_defaults_to_mid() {
        assert_eq!(mean_luma(&[]), 128);
    }
}
