//! FAST-9 corner detector (Rosten & Drummond) with SAD scoring and
//! non-maximal suppression.
//!
//! Detection is the dominant cost of the SLAM frontend, so the per-row
//! scan is parallelized with `rayon` across the Pi 4's four cores. On
//! aarch64 each row is additionally vectorized with NEON, processing 16
//! candidate columns per step from contiguous loads ([`detect_row_neon`]);
//! every other target uses the scalar [`corner_score`] path. Both produce
//! bit-identical results — see the `neon_parity` differential test.

use rayon::prelude::*;

use super::image::GrayImage;

/// A detected corner in some image's pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Corner {
    pub x: u16,
    pub y: u16,
    pub score: u16,
}

/// Bresenham circle of radius 3 (16 px), clockwise from the top.
const CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];
const BORDER: usize = 3; // circle radius
const ARC: usize = 9; // FAST-9: ≥9 contiguous ring pixels

/// If `(x, y)` is a FAST-9 corner at threshold `t`, return its score
/// (sum of |ring − centre| over the 16 ring pixels), else `None`.
#[inline]
fn corner_score(img: &GrayImage, x: usize, y: usize, t: i32) -> Option<u16> {
    let w = img.width;
    let ip = img.data[y * w + x] as i32;
    let hi = ip + t;
    let lo = ip - t;

    // Cheap reject on the 4 compass pixels (idx 0,4,8,12): any 9-arc
    // covers ≥2 of them, so <2 bright and <2 dark ⇒ not a corner.
    let mut cb = 0;
    let mut cd = 0;
    for k in [0usize, 4, 8, 12] {
        let (dx, dy) = CIRCLE[k];
        let p = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32;
        if p >= hi {
            cb += 1;
        } else if p <= lo {
            cd += 1;
        }
    }
    if cb < 2 && cd < 2 {
        return None;
    }

    // Classify the full ring and find the longest circular run.
    let mut cls = [0i8; 16];
    for (i, &(dx, dy)) in CIRCLE.iter().enumerate() {
        let p = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32;
        cls[i] = if p >= hi {
            1
        } else if p <= lo {
            -1
        } else {
            0
        };
    }
    if !has_arc(&cls, 1) && !has_arc(&cls, -1) {
        return None;
    }

    // Score: SAD of ring vs centre (monotone-ish; good for NMS ranking).
    let mut s = 0i32;
    for &(dx, dy) in &CIRCLE {
        let p = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as i32;
        s += (p - ip).abs();
    }
    Some(s.min(u16::MAX as i32) as u16)
}

/// Is there a circular run of ≥[`ARC`] pixels all equal to `want`?
#[inline]
fn has_arc(cls: &[i8; 16], want: i8) -> bool {
    let mut run = 0usize;
    // Scan 16+ARC-1 to cover wrap-around runs.
    for i in 0..16 + ARC - 1 {
        if cls[i % 16] == want {
            run += 1;
            if run >= ARC {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Scan one row scalar, pixel by pixel. The reference implementation:
/// the NEON path is verified bit-identical to this.
#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
fn detect_row_scalar(img: &GrayImage, y: usize, t: i32) -> Vec<Corner> {
    let w = img.width;
    let mut out = Vec::new();
    for x in BORDER..w - BORDER {
        if let Some(score) = corner_score(img, x, y, t) {
            out.push(Corner {
                x: x as u16,
                y: y as u16,
                score,
            });
        }
    }
    out
}

/// Scan one row, dispatching to the NEON path on aarch64.
#[inline]
fn detect_row(img: &GrayImage, y: usize, t: i32) -> Vec<Corner> {
    #[cfg(target_arch = "aarch64")]
    {
        detect_row_neon(img, y, t)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        detect_row_scalar(img, y, t)
    }
}

/// NEON FAST-9 over one row: 16 candidate columns per step.
///
/// For a fixed row the 16 ring samples at a given offset are contiguous
/// in memory, so each is a single `vld1q_u8`. Classification, the
/// circular ≥9-run arc test and the SAD score are all done 16-lanes-wide;
/// the row's right edge (a sub-16 remainder) falls back to
/// [`corner_score`]. Results are bit-identical to the scalar path —
/// the threshold compares are widened to 16-bit so they never saturate,
/// and `dark` is masked by `!bright` to mirror the scalar
/// `if p>=hi {..} else if p<=lo {..}` precedence at `t == 0`.
#[cfg(target_arch = "aarch64")]
fn detect_row_neon(img: &GrayImage, y: usize, t: i32) -> Vec<Corner> {
    use core::arch::aarch64::*;

    let w = img.width;
    let base = img.data.as_ptr();
    let tv = t as u16;
    let mut out = Vec::new();

    // Full 16-wide blocks. For a block starting at x0 the ring touches
    // columns [x0-3, x0+18]; emitting only x in [BORDER, w-BORDER) is
    // then automatic, and with BORDER==3 the condition below also keeps
    // x0+18 <= w-1, so every load stays in bounds.
    let mut x0 = BORDER;
    while x0 + 16 <= w - BORDER {
        // SAFETY: x0 >= BORDER==3 and x0+16 <= w-3, so for every ring
        // offset (dx,dy) with |dx|,|dy| <= 3 the 16-byte load at
        // (y+dy, x0+dx) lies fully inside `img.data`; `y` is always in
        // BORDER..h-BORDER. NEON is baseline on aarch64 (no runtime gate).
        unsafe {
            let c = vld1q_u8(base.add(y * w + x0));
            let c_lo = vmovl_u8(vget_low_u8(c));
            let c_hi = vmovl_u8(vget_high_u8(c));
            let tdup = vdupq_n_u16(tv);
            // bright threshold hi = c + t (u16, ≤510 — never saturates).
            let hi_lo = vaddq_u16(c_lo, tdup);
            let hi_hi = vaddq_u16(c_hi, tdup);
            // dark threshold lo = c - t as signed 16 (range [-255,255]).
            let tds = vreinterpretq_s16_u16(tdup);
            let lo_lo = vsubq_s16(vreinterpretq_s16_u16(c_lo), tds);
            let lo_hi = vsubq_s16(vreinterpretq_s16_u16(c_hi), tds);

            let ones = vdupq_n_u8(1);
            let mut bmask = [vdupq_n_u8(0); 16];
            let mut dmask = [vdupq_n_u8(0); 16];
            let mut sum_lo = vdupq_n_u16(0);
            let mut sum_hi = vdupq_n_u16(0);

            for (k, &(dx, dy)) in CIRCLE.iter().enumerate() {
                let p =
                    vld1q_u8(base.add(((y as i32 + dy) as usize) * w + (x0 as i32 + dx) as usize));
                let p_lo = vmovl_u8(vget_low_u8(p));
                let p_hi = vmovl_u8(vget_high_u8(p));

                // bright: p >= c + t  (unsigned 16-bit compare)
                let b = vcombine_u8(
                    vmovn_u16(vcgeq_u16(p_lo, hi_lo)),
                    vmovn_u16(vcgeq_u16(p_hi, hi_hi)),
                );
                // dark: p <= c - t  (signed 16-bit; p ≥ 0)
                let d_raw = vcombine_u8(
                    vmovn_u16(vcleq_s16(vreinterpretq_s16_u16(p_lo), lo_lo)),
                    vmovn_u16(vcleq_s16(vreinterpretq_s16_u16(p_hi), lo_hi)),
                );
                // Scalar classifies bright first; a pixel can only be
                // dark when it is not bright (matters at t==0).
                let d = vandq_u8(d_raw, vmvnq_u8(b));

                bmask[k] = b;
                dmask[k] = d;

                let ad = vabdq_u8(p, c);
                sum_lo = vaddq_u16(sum_lo, vmovl_u8(vget_low_u8(ad)));
                sum_hi = vaddq_u16(sum_hi, vmovl_u8(vget_high_u8(ad)));
            }

            // Per-lane circular ≥ARC run test, mirroring `has_arc`:
            // run = (run + 1) if eq else 0; corner if any run ≥ ARC.
            let arc = vdupq_n_u8(ARC as u8);
            let zero = vdupq_n_u8(0);
            let mut found_b = zero;
            let mut found_d = zero;
            let mut run_b = zero;
            let mut run_d = zero;
            for i in 0..16 + ARC - 1 {
                let m = bmask[i % 16];
                run_b = vandq_u8(vaddq_u8(run_b, vandq_u8(m, ones)), m);
                found_b = vorrq_u8(found_b, vcgeq_u8(run_b, arc));

                let md = dmask[i % 16];
                run_d = vandq_u8(vaddq_u8(run_d, vandq_u8(md, ones)), md);
                found_d = vorrq_u8(found_d, vcgeq_u8(run_d, arc));
            }
            let corner = vorrq_u8(found_b, found_d);

            let mut cmask = [0u8; 16];
            let mut scores = [0u16; 16];
            vst1q_u8(cmask.as_mut_ptr(), corner);
            vst1q_u16(scores.as_mut_ptr(), sum_lo);
            vst1q_u16(scores.as_mut_ptr().add(8), sum_hi);

            for (lane, &m) in cmask.iter().enumerate() {
                if m != 0 {
                    out.push(Corner {
                        x: (x0 + lane) as u16,
                        y: y as u16,
                        // ≤ 16*255 = 4080, so .min() matches the scalar
                        // `s.min(u16::MAX)` (a no-op here).
                        score: scores[lane].min(u16::MAX),
                    });
                }
            }
        }
        x0 += 16;
    }

    // Right-edge remainder (< 16 columns).
    for x in x0..w - BORDER {
        if let Some(score) = corner_score(img, x, y, t) {
            out.push(Corner {
                x: x as u16,
                y: y as u16,
                score,
            });
        }
    }
    out
}

/// Detect FAST-9 corners over the whole image at threshold `t`.
/// Rows are scanned in parallel.
pub fn detect(img: &GrayImage, t: u8) -> Vec<Corner> {
    let (w, h) = (img.width, img.height);
    if w <= 2 * BORDER || h <= 2 * BORDER {
        return Vec::new();
    }
    let t = t as i32;
    (BORDER..h - BORDER)
        .into_par_iter()
        .flat_map_iter(move |y| detect_row(img, y, t).into_iter())
        .collect()
}

/// 3×3 non-maximal suppression: keep a corner only if its score is
/// strictly greater than every 8-neighbour corner's score.
pub fn non_max_suppress(corners: &[Corner], w: usize, h: usize) -> Vec<Corner> {
    if corners.is_empty() {
        return Vec::new();
    }
    let mut score = vec![0u16; w * h];
    for c in corners {
        score[c.y as usize * w + c.x as usize] = c.score;
    }
    corners
        .iter()
        .copied()
        .filter(|c| {
            let (cx, cy, s) = (c.x as usize, c.y as usize, c.score);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    if score[ny as usize * w + nx as usize] > s {
                        return false;
                    }
                }
            }
            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Image with a bright filled disc on a dark field — its rim is a
    /// ring of corners; the flat interior/background are not.
    fn disc(w: usize, h: usize) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        let (cx, cy, r2) = (
            w as f32 / 2.0,
            h as f32 / 2.0,
            (w.min(h) as f32 / 4.0).powi(2),
        );
        for y in 0..h {
            for x in 0..w {
                let d = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                img.data[y * w + x] = if d < r2 { 240 } else { 15 };
            }
        }
        img
    }

    #[test]
    fn flat_image_has_no_corners() {
        let mut img = GrayImage::new(40, 40);
        img.data.iter_mut().for_each(|p| *p = 100);
        assert!(detect(&img, 20).is_empty());
    }

    #[test]
    fn detects_corners_on_a_disc_rim() {
        let img = disc(60, 60);
        let corners = detect(&img, 30);
        assert!(
            corners.len() > 8,
            "expected many rim corners, got {}",
            corners.len()
        );
        // All near the rim radius (~15 px from centre), none at centre.
        let (cx, cy) = (30.0f32, 30.0f32);
        for c in &corners {
            let d = ((c.x as f32 - cx).powi(2) + (c.y as f32 - cy).powi(2)).sqrt();
            assert!(d > 6.0, "corner too close to flat centre: {c:?}");
        }
    }

    #[test]
    fn nms_thins_clusters() {
        let img = disc(60, 60);
        let raw = detect(&img, 30);
        let kept = non_max_suppress(&raw, img.width, img.height);
        assert!(!kept.is_empty());
        assert!(kept.len() <= raw.len());
    }

    #[test]
    fn higher_threshold_detects_fewer() {
        let img = disc(80, 80);
        let lo = detect(&img, 20).len();
        let hi = detect(&img, 80).len();
        assert!(hi <= lo, "stricter threshold should not add corners");
    }

    /// Differential test: the dispatched `detect` (NEON on aarch64) must
    /// be bit-identical to the pure scalar reference. On non-aarch64 both
    /// sides are scalar, so this still exercises the harness and the
    /// row-tiling math.
    mod neon_parity {
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

        fn scalar_detect(img: &GrayImage, t: u8) -> Vec<Corner> {
            let (w, h) = (img.width, img.height);
            if w <= 2 * BORDER || h <= 2 * BORDER {
                return Vec::new();
            }
            let mut v = Vec::new();
            for y in BORDER..h - BORDER {
                v.extend(detect_row_scalar(img, y, t as i32));
            }
            v
        }

        fn sorted(mut v: Vec<Corner>) -> Vec<Corner> {
            v.sort_unstable_by_key(|c| (c.y, c.x, c.score));
            v
        }

        #[test]
        fn neon_matches_scalar() {
            let mut rng = Xorshift::new(0x00C0_FFEE_1234_5678);
            // Mix of widths that are / aren't multiples of 16, plus a
            // tiny image, to stress the NEON tail and bounds math.
            for &(w, h) in &[
                (13, 13),
                (16, 9),
                (40, 30),
                (64, 48),
                (70, 33),
                (129, 17),
                (200, 77),
            ] {
                for &t in &[0u8, 1, 5, 20, 40, 100, 200, 255] {
                    let img = rand_img(&mut rng, w, h);
                    let neon = sorted(detect(&img, t));
                    let scalar = sorted(scalar_detect(&img, t));
                    assert_eq!(
                        neon,
                        scalar,
                        "mismatch w={w} h={h} t={t}: dispatched={} scalar={}",
                        neon.len(),
                        scalar.len()
                    );
                }
            }
        }
    }
}
