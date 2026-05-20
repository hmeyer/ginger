//! Deterministic seeded PRNGs — the single source of truth for every
//! *seeded* random stream in the workspace, with [`rand_core::RngCore`]
//! adapters so consumers can layer the `rand` ecosystem's
//! `Rng` / `rand_distr` ergonomics on top.
//!
//! The bit sequences these types produce are load-bearing: RANSAC
//! model selection and inlier sets, the k-means++ BoW vocabulary, and
//! the fixed BRIEF sampling pattern are pure functions of
//! `(inputs, seed)`. The headless test suite gates on *invariants*
//! (counts, tolerances, structural shape) rather than literal bytes,
//! but still relies on those streams being intra-build reproducible.
//! Changing an algorithm/constant here is a deliberate behavior step,
//! never an incidental cleanup.
//!
//! Three streams exist because three call sites genuinely need
//! different shapes — do not "unify" them:
//!
//! * [`Rng64`] — xorshift64\* (Vigna): the core stream for RANSAC /
//!   k-means / numeric test fixtures.
//! * [`Rng32`] — xorshift32: only builds the fixed BRIEF pattern.
//! * [`Xs64`] — plain xorshift64 (13/7/17): the buzzer/LED show and a
//!   few image-fuzz test fixtures.
//!
//! Plus [`noise_u8`], the integer value-noise hash behind the mock
//! camera's synthetic frames.
//!
//! # `rand_core` adapter layer
//!
//! Each of [`Rng64`] / [`Rng32`] / [`Xs64`] implements
//! [`rand_core::TryRng`] with `Error = Infallible`, which gives them
//! blanket impls of [`rand_core::Rng`] (and the deprecated
//! [`rand_core::RngCore`] alias) for free. Inherent methods take
//! precedence in Rust's method resolution, so existing call sites
//! (`rng.next_u64()`, `rng.f()`, `rng.upto(n)`, etc.) stay
//! byte-identical; the trait impl is purely additive and lets new
//! call sites pass `&mut rng` to any function bounded by `R: Rng`
//! (or, via the `rand` crate, `R: rand::RngExt`).

/// Vigna's xorshift64\* output multiplier.
const MUL: u64 = 0x2545_F491_4F6C_DD1D;

/// xorshift64\* (Marsaglia 12/25/27 + Vigna output scramble).
///
/// State advances by the three shifts only; the multiply is an
/// output-only scramble (it does not feed back into the state), so
/// [`Rng64::next_u64`] and [`Rng64::f`] each advance the stream by
/// exactly one step.
pub struct Rng64(pub u64);

impl Rng64 {
    /// A fresh stream seeded with `seed` (used verbatim).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next scrambled `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(MUL)
    }

    /// Uniform `f64` in `[0, 1)` with a 53-bit mantissa.
    #[inline]
    pub fn f(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(MUL) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform index in `[0, n)` via the scrambled integer (`n > 0`).
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform index in `[0, n)` via the scrambled integer (`n > 0`).
    ///
    /// Identical to [`below`](Self::below); kept as a distinct name
    /// because the RANSAC call sites read as `upto`.
    #[inline]
    pub fn upto(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform index in `[0, n)` via the unit float, clamped to `n - 1`.
    /// Returns `0` for `n == 0`. This is the k-means++ index draw and is
    /// *not* the same sequence as [`upto`](Self::upto).
    #[inline]
    pub fn upto_unit(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.f() * n as f64) as usize).min(n - 1)
    }

    /// A pseudo-random byte (`f() * 256`).
    #[inline]
    pub fn byte(&mut self) -> u8 {
        (self.f() * 256.0) as u8
    }
}

impl rand_core::TryRng for Rng64 {
    type Error = core::convert::Infallible;
    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(Rng64::next_u64(self))
    }
    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok((Rng64::next_u64(self) >> 32) as u32)
    }
    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        rand_core::utils::fill_bytes_via_next_word(dst, || Ok(Rng64::next_u64(self)))
    }
}

/// xorshift32 (Marsaglia 13/17/5). Only used to build the fixed BRIEF
/// sampling pattern, so its sequence pins every descriptor in the map.
pub struct Rng32(pub u32);

impl Rng32 {
    #[inline]
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal via Box–Muller.
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

impl rand_core::TryRng for Rng32 {
    type Error = core::convert::Infallible;
    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(Rng32::next_u32(self))
    }
    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        rand_core::utils::next_u64_via_u32(self)
    }
    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        rand_core::utils::fill_bytes_via_next_word(dst, || Ok(Rng32::next_u32(self)))
    }
}

/// Plain xorshift64 (13/7/17), raw state as output. Used where the
/// stream quality is irrelevant (the buzzer/LED show) or only seeds
/// image-fuzz test fixtures.
pub struct Xs64(pub u64);

impl Xs64 {
    /// Seed forced odd (a zero state would be a fixed point).
    #[inline]
    pub fn seeded(seed: u64) -> Self {
        Self(seed | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// High 32 bits of the next state.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform integer in `[lo, hi]`.
    #[inline]
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }

    /// Uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl rand_core::TryRng for Xs64 {
    type Error = core::convert::Infallible;
    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(Xs64::next_u64(self))
    }
    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(Xs64::next_u32(self))
    }
    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        rand_core::utils::fill_bytes_via_next_word(dst, || Ok(Xs64::next_u64(self)))
    }
}

/// Integer value-noise hash → `[0, 1]`. Deterministic per `(x, y)`;
/// drives the mock camera's synthetic multi-octave noise frames.
#[inline]
pub fn noise_u8(x: i32, y: i32) -> f32 {
    let mut n = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263)) as u32;
    n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
    ((n ^ (n >> 16)) & 0xff) as f32 / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng64_is_deterministic_and_advances() {
        let mut a = Rng64::new(0xDEAD_BEEF);
        let mut b = Rng64::new(0xDEAD_BEEF);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(xs, ys);
        assert!(xs.windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn rng64_f_in_unit_interval() {
        let mut r = Rng64::new(1);
        for _ in 0..10_000 {
            let v = r.f();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn rng64_upto_variants_stay_in_range() {
        let mut r = Rng64::new(42);
        for _ in 0..1000 {
            assert!(r.below(7) < 7);
            assert!(r.upto(7) < 7);
            assert!(r.upto_unit(7) < 7);
        }
        assert_eq!(Rng64::new(0).upto_unit(0), 0);
    }

    #[test]
    fn rng32_unit_in_interval_and_deterministic() {
        let mut a = Rng32::new(0x9E37_79B9);
        let mut b = Rng32::new(0x9E37_79B9);
        for _ in 0..10_000 {
            let v = a.unit();
            assert!((0.0..1.0).contains(&v));
            assert_eq!(v, b.unit());
        }
    }

    #[test]
    fn xs64_seed_forced_odd_and_ranges() {
        assert_eq!(Xs64::seeded(0).0 & 1, 1);
        assert_eq!(Xs64::seeded(2).0 & 1, 1);
        let mut r = Xs64::seeded(123);
        for _ in 0..1000 {
            let v = r.range(4, 7);
            assert!((4..=7).contains(&v));
            assert!((0.0..1.0).contains(&r.unit()));
        }
    }

    #[test]
    fn rng_trait_impls_match_inherent_methods() {
        // The `rand_core::Rng` (auto-impl'd from our `TryRng` impl)
        // must delegate to the inherent methods so existing call sites
        // stay byte-identical. Inherent method resolution beats the
        // trait, so this test guards the *trait* path against silent
        // divergence.
        use rand_core::Rng;

        let mut a = Rng64::new(0xABCD_EF01_2345_6789);
        let mut b = Rng64::new(0xABCD_EF01_2345_6789);
        for _ in 0..256 {
            assert_eq!(Rng64::next_u64(&mut a), Rng::next_u64(&mut b));
        }

        let mut a = Rng32::new(0xCAFE_BABE);
        let mut b = Rng32::new(0xCAFE_BABE);
        for _ in 0..256 {
            assert_eq!(Rng32::next_u32(&mut a), Rng::next_u32(&mut b));
        }

        let mut a = Xs64::seeded(0xFEED_FACE_DEAD_BEEF);
        let mut b = Xs64::seeded(0xFEED_FACE_DEAD_BEEF);
        for _ in 0..256 {
            assert_eq!(Xs64::next_u64(&mut a), Rng::next_u64(&mut b));
        }
    }

    #[test]
    fn noise_u8_deterministic_and_bounded() {
        for x in -5..5 {
            for y in -5..5 {
                let v = noise_u8(x, y);
                assert!((0.0..=1.0).contains(&v));
                assert_eq!(v, noise_u8(x, y));
            }
        }
    }
}
