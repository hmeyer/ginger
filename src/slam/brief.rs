//! Oriented BRIEF descriptor (the "rBRIEF" half of ORB) plus a
//! brute-force Hamming matcher.
//!
//! FAST says *where* a keypoint is; BRIEF says *what the patch around it
//! looks like* as a 256-bit binary string built from intensity
//! comparisons of point pairs. To make it rotation-invariant (ORB) we
//! first estimate the keypoint orientation by the intensity centroid and
//! steer the sampling pattern by that angle.
//!
//! The sampling pattern is the original BRIEF "G_II" distribution
//! (isotropic Gaussian pairs within the patch), generated once from a
//! fixed seed so it is deterministic and reproducible without carrying a
//! 1024-entry magic table. This is a valid rotation-aware BRIEF; the
//! learned ORB table is a matching-quality optimization we can swap in
//! later without touching callers.

use std::sync::OnceLock;

use ginger_rand::Rng32;
use rayon::prelude::*;

use super::image::GrayImage;

/// Patch is `(2*PATCH_RADIUS+1)` square; orientation centroid and every
/// (post-rotation) sample point live inside the disc of this radius.
const PATCH_RADIUS: i32 = 15;
/// A keypoint must be at least this far from the level-image edge for the
/// patch (and any steered sample) to be fully in-bounds.
pub const BORDER: i32 = PATCH_RADIUS + 1;
/// Descriptor length: 256 bits = 32 bytes.
pub const DESC_BYTES: usize = 32;
const N_TESTS: usize = DESC_BYTES * 8;

/// One 256-bit oriented-BRIEF descriptor.
pub type Descriptor = [u8; DESC_BYTES];

// ── Sampling pattern ──────────────────────────────────────────────────────────

/// 256 point pairs `[ax, ay, bx, by]`, both points inside the patch disc.
type Pattern = [[i32; 4]; N_TESTS];

fn build_pattern() -> Pattern {
    let mut rng = Rng32::new(0x9E37_79B9);
    let mut pat = [[0i32; 4]; N_TESTS];
    // G_II: first point ~N(0, s1²), second ~N(first, s2²).
    let s1 = PATCH_RADIUS as f32 / 3.0;
    let s2 = PATCH_RADIUS as f32 / 6.0;
    let r2 = (PATCH_RADIUS * PATCH_RADIUS) as f32;
    let draw = |rng: &mut Rng32, cx: i32, cy: i32, sigma: f32| {
        // Reject outside the disc; clamp as a safety net so this can't spin.
        for _ in 0..32 {
            let x = cx + (rng.gauss() * sigma).round() as i32;
            let y = cy + (rng.gauss() * sigma).round() as i32;
            if (x * x + y * y) as f32 <= r2 {
                return (x, y);
            }
        }
        (
            cx.clamp(-PATCH_RADIUS, PATCH_RADIUS),
            cy.clamp(-PATCH_RADIUS, PATCH_RADIUS),
        )
    };
    for p in pat.iter_mut() {
        let (ax, ay) = draw(&mut rng, 0, 0, s1);
        let (bx, by) = draw(&mut rng, ax, ay, s2);
        *p = [ax, ay, bx, by];
    }
    pat
}

fn pattern() -> &'static Pattern {
    static PATTERN: OnceLock<Pattern> = OnceLock::new();
    PATTERN.get_or_init(build_pattern)
}

// ── Geometry / description ────────────────────────────────────────────────────

/// True if a `(x, y)` keypoint at this level is far enough from the edge
/// to compute orientation and a (steered) descriptor.
#[inline]
pub fn in_bounds(w: usize, h: usize, x: i32, y: i32) -> bool {
    x >= BORDER && y >= BORDER && x < w as i32 - BORDER && y < h as i32 - BORDER
}

/// Intensity-centroid orientation (radians) over the patch disc. Caller
/// must guarantee [`in_bounds`].
pub fn orientation(img: &GrayImage, x: i32, y: i32) -> f32 {
    let r = PATCH_RADIUS;
    let r2 = r * r;
    let (mut m10, mut m01) = (0i64, 0i64);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r2 {
                let p = img.at((x + dx) as usize, (y + dy) as usize) as i64;
                m10 += dx as i64 * p;
                m01 += dy as i64 * p;
            }
        }
    }
    (m01 as f32).atan2(m10 as f32)
}

/// Steered 256-bit BRIEF on `img` (which should be a smoothed level
/// image). Caller must guarantee [`in_bounds`].
pub fn describe(img: &GrayImage, x: i32, y: i32, angle: f32) -> Descriptor {
    let (s, c) = angle.sin_cos();
    let mut d = [0u8; DESC_BYTES];
    for (i, &[ax, ay, bx, by]) in pattern().iter().enumerate() {
        let (ax, ay, bx, by) = (ax as f32, ay as f32, bx as f32, by as f32);
        let pax = (c * ax - s * ay).round() as i32;
        let pay = (s * ax + c * ay).round() as i32;
        let pbx = (c * bx - s * by).round() as i32;
        let pby = (s * bx + c * by).round() as i32;
        let va = img.at((x + pax) as usize, (y + pay) as usize);
        let vb = img.at((x + pbx) as usize, (y + pby) as usize);
        if va < vb {
            d[i >> 3] |= 1 << (i & 7);
        }
    }
    d
}

// ── Matching ──────────────────────────────────────────────────────────────────

/// Hamming distance (number of differing bits) between two descriptors.
#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Reject a match whose best distance exceeds this many bits.
const MAX_DIST: u32 = 64;
/// Lowe ratio: best must beat `RATIO * second_best`.
const RATIO: f32 = 0.8;

/// Brute-force mutual-nearest-neighbour matcher with the ratio test.
/// Returns `(i, j)` index pairs into `a` and `b`.
///
/// The full `a × b` Hamming matrix is computed **once** (row `i` = all
/// distances from `a[i]`); the forward best-two/ratio pass reads it
/// row-wise and the backward best pass reads it column-wise. This is
/// output-identical to scoring each direction independently — same scan
/// order, so ties break the same way — but halves the Hamming work, which
/// previously recomputed every distance a second time for `bwd`.
pub fn match_descriptors(a: &[Descriptor], b: &[Descriptor]) -> Vec<(u32, u32)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (na, nb) = (a.len(), b.len());

    // Distance matrix, one row per `a`, computed in parallel over rows.
    // Max distance is 256 bits, so `u16` holds it exactly.
    let mut dist = vec![0u16; na * nb];
    dist.par_chunks_mut(nb)
        .zip(a.par_iter())
        .for_each(|(row, da)| {
            for (slot, db) in row.iter_mut().zip(b) {
                *slot = hamming(da, db) as u16;
            }
        });

    // Forward: best b for each a (best two + ratio test), reading row i.
    let fwd: Vec<Option<usize>> = dist
        .par_chunks(nb)
        .map(|row| {
            let (mut bi, mut d1, mut d2) = (usize::MAX, u32::MAX, u32::MAX);
            for (j, &d) in row.iter().enumerate() {
                let d = d as u32;
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    bi = j;
                } else if d < d2 {
                    d2 = d;
                }
            }
            let ratio_ok = d2 == u32::MAX || (d1 as f32) < RATIO * d2 as f32;
            if d1 <= MAX_DIST && ratio_ok {
                Some(bi)
            } else {
                None
            }
        })
        .collect();

    // Backward: best a for each b — same matrix, read column-wise. Used
    // only for the mutual-consistency check.
    let bwd: Vec<usize> = (0..nb)
        .into_par_iter()
        .map(|j| {
            let (mut ai, mut best) = (usize::MAX, u32::MAX);
            for i in 0..na {
                let d = dist[i * nb + j] as u32;
                if d < best {
                    best = d;
                    ai = i;
                }
            }
            ai
        })
        .collect();

    fwd.iter()
        .enumerate()
        .filter_map(|(i, &mj)| {
            let j = mj?;
            (bwd[j] == i).then_some((i as u32, j as u32))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise_image(w: usize, h: usize, seed: u32) -> GrayImage {
        let mut g = GrayImage::new(w, h);
        let mut r = Rng32::new(seed | 1);
        for p in g.data.iter_mut() {
            *p = (r.next_u32() & 0xff) as u8;
        }
        g
    }

    #[test]
    fn pattern_deterministic_and_inside_disc() {
        let a = pattern();
        let b = build_pattern();
        assert_eq!(a, &b, "pattern must be reproducible");
        let r2 = (PATCH_RADIUS * PATCH_RADIUS) as f32;
        for &[ax, ay, bx, by] in a.iter() {
            assert!((ax * ax + ay * ay) as f32 <= r2);
            assert!((bx * bx + by * by) as f32 <= r2);
        }
    }

    #[test]
    fn descriptor_is_deterministic_and_hamming_extremes() {
        let g = noise_image(64, 64, 7);
        let d1 = describe(&g, 32, 32, 0.3);
        let d2 = describe(&g, 32, 32, 0.3);
        assert_eq!(d1, d2);
        assert_eq!(hamming(&d1, &d2), 0);
        let mut inv = d1;
        for b in inv.iter_mut() {
            *b = !*b;
        }
        assert_eq!(hamming(&d1, &inv), (DESC_BYTES * 8) as u32);
    }

    #[test]
    fn orientation_points_to_the_bright_side() {
        // Bright right half → centroid pulls +x → angle ≈ 0.
        let mut g = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                g.data[y * 64 + x] = if x >= 32 { 255 } else { 0 };
            }
        }
        let a = orientation(&g, 32, 32);
        assert!(a.abs() < 0.4, "expected ~0, got {a}");

        // Bright bottom half → angle ≈ +π/2.
        let mut g = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                g.data[y * 64 + x] = if y >= 32 { 255 } else { 0 };
            }
        }
        let a = orientation(&g, 32, 32);
        assert!(
            (a - std::f32::consts::FRAC_PI_2).abs() < 0.4,
            "expected ~π/2, got {a}"
        );
    }

    #[test]
    fn border_guard() {
        assert!(!in_bounds(64, 64, 0, 0));
        assert!(!in_bounds(64, 64, BORDER - 1, 32));
        assert!(in_bounds(64, 64, BORDER, BORDER));
        assert!(in_bounds(64, 64, 32, 32));
        assert!(!in_bounds(64, 64, 64 - BORDER, 32));
    }

    #[test]
    fn matcher_recovers_noisy_correspondence() {
        // Build distinct descriptors, perturb a few bits, expect i↔i.
        let n = 40;
        let mut rng = Rng32::new(12345);
        let a: Vec<Descriptor> = (0..n)
            .map(|_| {
                let mut d = [0u8; DESC_BYTES];
                for b in d.iter_mut() {
                    *b = (rng.next_u32() & 0xff) as u8;
                }
                d
            })
            .collect();
        let b: Vec<Descriptor> = a
            .iter()
            .map(|d| {
                let mut d = *d;
                d[0] ^= 0b0000_0011; // flip 2 bits — well under MAX_DIST
                d
            })
            .collect();
        let m = match_descriptors(&a, &b);
        assert!(m.len() >= n - 2, "recovered only {} of {n}", m.len());
        assert!(m.iter().all(|&(i, j)| i == j));
    }

    #[test]
    fn empty_inputs_match_nothing() {
        assert!(match_descriptors(&[], &[[0u8; DESC_BYTES]]).is_empty());
        assert!(match_descriptors(&[[0u8; DESC_BYTES]], &[]).is_empty());
    }
}
