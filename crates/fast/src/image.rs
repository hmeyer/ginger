//! Grayscale image + scale pyramid for the SLAM frontend.
//!
//! YUYV→luma extraction lives in the main crate (it needs the camera
//! `Frame` type); this crate keeps only the camera-independent compute so
//! it cross-compiles for aarch64 without the hardware stack.

/// A single-channel 8-bit image, row-major, no padding.
#[derive(Clone)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// Bilinear downscale to `(w, h)`. Used to build the pyramid; the
    /// source is larger so this is a reduction (mild low-pass + sample).
    ///
    /// Dispatches to the NEON path on aarch64 (pyramid build is per-frame
    /// hot); every other target uses [`Self::resized_scalar`]. Both produce
    /// bit-identical output — see the `resize_parity` differential test.
    #[inline]
    pub fn resized(&self, w: usize, h: usize) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            self.resized_neon(w, h)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.resized_scalar(w, h)
        }
    }

    /// Scalar bilinear downscale. The reference implementation: the NEON
    /// path is verified bit-identical to this.
    #[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
    fn resized_scalar(&self, w: usize, h: usize) -> Self {
        let mut out = GrayImage::new(w, h);
        if w == 0 || h == 0 {
            return out;
        }
        // Map output centre back to input space.
        let sx = self.width as f32 / w as f32;
        let sy = self.height as f32 / h as f32;
        for oy in 0..h {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let wy = fy - y0 as f32;
            for ox in 0..w {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let wx = fx - x0 as f32;
                let p00 = self.at(x0, y0) as f32;
                let p10 = self.at(x1, y0) as f32;
                let p01 = self.at(x0, y1) as f32;
                let p11 = self.at(x1, y1) as f32;
                let top = p00 + (p10 - p00) * wx;
                let bot = p01 + (p11 - p01) * wx;
                out.data[oy * w + ox] = (top + (bot - top) * wy).round() as u8;
            }
        }
        out
    }

    /// NEON bilinear downscale: 4 output columns per step.
    ///
    /// Bilinear needs four scattered taps per pixel, and the source x of
    /// adjacent outputs is non-uniform for a non-integer scale (≈1.2 for
    /// the pyramid), so the gather stays scalar. The interpolation itself
    /// — three lerps and the round-to-`u8` — is the bulk of the work and
    /// runs 4-wide in `f32`. It is bit-identical to [`Self::resized_scalar`]:
    /// the lerp uses the same separate (non-FMA) `sub`/`mul`/`add` ops in
    /// the same order at IEEE-754 single precision, and `vcvtaq_u32_f32`
    /// (round to nearest, ties **away** from zero) reproduces Rust's
    /// `f32::round()` exactly for the result, which is always in `[0, 255]`.
    #[cfg(target_arch = "aarch64")]
    fn resized_neon(&self, w: usize, h: usize) -> Self {
        use core::arch::aarch64::*;

        let mut out = GrayImage::new(w, h);
        if w == 0 || h == 0 {
            return out;
        }
        let (sw, sh) = (self.width, self.height);
        let sx = sw as f32 / w as f32;
        let sy = sh as f32 / h as f32;
        let src = self.data.as_ptr();

        for oy in 0..h {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let wy = fy - y0 as f32;
            let row0 = y0 * sw;
            let row1 = y1 * sw;

            let mut ox = 0;
            // Full 4-wide blocks; the sub-4 remainder falls back below.
            while ox + 4 <= w {
                // Per-lane source x/weight (scalar — the gather is
                // non-uniform). Identical maths to the scalar reference.
                let mut a00 = [0.0f32; 4];
                let mut a10 = [0.0f32; 4];
                let mut a01 = [0.0f32; 4];
                let mut a11 = [0.0f32; 4];
                let mut wxa = [0.0f32; 4];
                for l in 0..4 {
                    let fx = (((ox + l) as f32 + 0.5) * sx - 0.5).max(0.0);
                    let x0 = fx.floor() as usize;
                    let x1 = (x0 + 1).min(sw - 1);
                    wxa[l] = fx - x0 as f32;
                    // SAFETY: x0,x1 ∈ [0, sw-1] and y0,y1 ∈ [0, sh-1], so
                    // row{0,1}+x{0,1} < sw*sh == self.data.len().
                    unsafe {
                        a00[l] = *src.add(row0 + x0) as f32;
                        a10[l] = *src.add(row0 + x1) as f32;
                        a01[l] = *src.add(row1 + x0) as f32;
                        a11[l] = *src.add(row1 + x1) as f32;
                    }
                }

                // SAFETY: NEON is baseline on aarch64 (no runtime gate);
                // all loads/stores below are on 4-element stack arrays or
                // `out.data[oy*w+ox .. +4]` with `ox+4 <= w`, `oy < h`.
                unsafe {
                    let wxv = vld1q_f32(wxa.as_ptr());
                    let wyv = vdupq_n_f32(wy);
                    let p00 = vld1q_f32(a00.as_ptr());
                    let p10 = vld1q_f32(a10.as_ptr());
                    let p01 = vld1q_f32(a01.as_ptr());
                    let p11 = vld1q_f32(a11.as_ptr());

                    // top = p00 + (p10-p00)*wx ; bot = p01 + (p11-p01)*wx
                    let top = vaddq_f32(p00, vmulq_f32(vsubq_f32(p10, p00), wxv));
                    let bot = vaddq_f32(p01, vmulq_f32(vsubq_f32(p11, p01), wxv));
                    // out = round(top + (bot-top)*wy), ties away from zero.
                    let res = vaddq_f32(top, vmulq_f32(vsubq_f32(bot, top), wyv));
                    let ri = vcvtaq_u32_f32(res);

                    let n16 = vmovn_u32(ri);
                    let n8 = vmovn_u16(vcombine_u16(n16, n16));
                    let mut buf = [0u8; 8];
                    vst1_u8(buf.as_mut_ptr(), n8);
                    let d = oy * w + ox;
                    out.data[d..d + 4].copy_from_slice(&buf[..4]);
                }
                ox += 4;
            }

            // Right-edge remainder (< 4 columns).
            for ox in ox..w {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let wx = fx - x0 as f32;
                let p00 = self.at(x0, y0) as f32;
                let p10 = self.at(x1, y0) as f32;
                let p01 = self.at(x0, y1) as f32;
                let p11 = self.at(x1, y1) as f32;
                let top = p00 + (p10 - p00) * wx;
                let bot = p01 + (p11 - p01) * wx;
                out.data[oy * w + ox] = (top + (bot - top) * wy).round() as u8;
            }
        }
        out
    }

    /// Separable box blur (window `2*radius+1`), edges use a shrinking
    /// window. Cheap low-pass so BRIEF samples single pixels instead of
    /// averaging a neighbourhood per test.
    pub fn box_blur(&self, radius: usize) -> GrayImage {
        if radius == 0 {
            return self.clone();
        }
        let (w, h) = (self.width, self.height);
        let mut tmp = vec![0u8; w * h];
        for y in 0..h {
            let row = &self.data[y * w..y * w + w];
            for x in 0..w {
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(w - 1);
                let acc: u32 = row[x0..=x1].iter().map(|&p| p as u32).sum();
                tmp[y * w + x] = (acc / (x1 - x0 + 1) as u32) as u8;
            }
        }
        let mut out = GrayImage::new(w, h);
        for x in 0..w {
            for y in 0..h {
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius).min(h - 1);
                let mut acc = 0u32;
                for yy in y0..=y1 {
                    acc += tmp[yy * w + x] as u32;
                }
                out.data[y * w + x] = (acc / (y1 - y0 + 1) as u32) as u8;
            }
        }
        out
    }
}

/// One pyramid level: the image plus the factor mapping a coordinate at
/// this level back to level 0 (`scale_factor ** level`).
pub struct PyramidLevel {
    pub image: GrayImage,
    pub scale: f32,
}

/// ORB-SLAM-style scale pyramid: `n_levels` images, each `scale_factor`×
/// smaller than the previous (level 0 is the original).
pub fn build_pyramid(base: GrayImage, n_levels: usize, scale_factor: f32) -> Vec<PyramidLevel> {
    let mut levels = Vec::with_capacity(n_levels);
    let mut scale = 1.0f32;
    for lvl in 0..n_levels {
        let image = if lvl == 0 {
            base.clone()
        } else {
            let w = (base.width as f32 / scale).round() as usize;
            let h = (base.height as f32 / scale).round() as usize;
            if w < 8 || h < 8 {
                break;
            }
            base.resized(w, h)
        };
        levels.push(PyramidLevel { image, scale });
        scale *= scale_factor;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_shrinks_and_stops() {
        let base = GrayImage::new(640, 480);
        let levels = build_pyramid(base, 8, 1.2);
        assert_eq!(levels.len(), 8);
        assert_eq!((levels[0].image.width, levels[0].image.height), (640, 480));
        assert!((levels[0].scale - 1.0).abs() < 1e-6);
        // Each level is strictly smaller and scale grows by ~1.2.
        for w in levels.windows(2) {
            assert!(w[1].image.width < w[0].image.width);
            assert!((w[1].scale / w[0].scale - 1.2).abs() < 1e-3);
        }
    }

    #[test]
    fn resize_preserves_constant_image() {
        let mut g = GrayImage::new(20, 20);
        g.data.iter_mut().for_each(|p| *p = 123);
        let r = g.resized(7, 9);
        assert_eq!((r.width, r.height), (7, 9));
        assert!(r.data.iter().all(|&p| p == 123));
    }

    /// Differential test: the dispatched `resized` (NEON on aarch64) must
    /// be bit-identical to the pure scalar reference. On non-aarch64 both
    /// sides are scalar, so this still exercises the harness and the
    /// 4-wide block / remainder tiling math.
    mod resize_parity {
        use super::*;

        struct Xorshift(u64);
        impl Xorshift {
            fn new(seed: u64) -> Self {
                Self(seed | 1)
            }
            fn next_u32(&mut self) -> u32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                (x >> 32) as u32
            }
        }

        fn rand_img(rng: &mut Xorshift, w: usize, h: usize) -> GrayImage {
            let mut g = GrayImage::new(w, h);
            for p in g.data.iter_mut() {
                *p = (rng.next_u32() & 0xff) as u8;
            }
            g
        }

        #[test]
        fn neon_matches_scalar() {
            let mut rng = Xorshift::new(0x0DEF_ACED_DEAD_BEEF);
            // Source sizes (some not multiples of 4) crossed with target
            // sizes: pyramid-like 1.2× steps plus tiny / odd targets to
            // stress the NEON tail and the gather bounds.
            for &(sw, sh) in &[(13, 11), (40, 30), (64, 48), (129, 77), (200, 95)] {
                let img = rand_img(&mut rng, sw, sh);
                let mut tw = sw as f32;
                let mut th = sh as f32;
                loop {
                    let (w, h) = (tw.round() as usize, th.round() as usize);
                    if w < 8 || h < 8 {
                        break;
                    }
                    let got = img.resized(w, h);
                    let want = img.resized_scalar(w, h);
                    assert_eq!(
                        (got.width, got.height),
                        (want.width, want.height),
                        "size mismatch sw={sw} sh={sh} -> w={w} h={h}"
                    );
                    assert_eq!(
                        got.data, want.data,
                        "pixel mismatch sw={sw} sh={sh} -> w={w} h={h}"
                    );
                    tw /= 1.2;
                    th /= 1.2;
                }
                // Also a few non-proportional / tiny targets.
                for &(w, h) in &[(1, 1), (3, 7), (7, 3), (5, sh), (sw, 5), (33, 33)] {
                    assert_eq!(img.resized(w, h).data, img.resized_scalar(w, h).data);
                }
            }
        }
    }

    #[test]
    fn box_blur_preserves_size_and_constant_and_smooths() {
        let mut g = GrayImage::new(16, 12);
        g.data.iter_mut().for_each(|p| *p = 90);
        let b = g.box_blur(2);
        assert_eq!((b.width, b.height), (16, 12));
        assert!(b.data.iter().all(|&p| p == 90));

        // A lone bright pixel must spread (centre drops, a neighbour rises).
        let mut g = GrayImage::new(16, 16);
        g.data[8 * 16 + 8] = 255;
        let b = g.box_blur(1);
        assert!(b.at(8, 8) < 255);
        assert!(b.at(8, 9) > 0);
    }
}
