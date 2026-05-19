//! The robot's buzzer + LED "personality": one tap plays a short,
//! randomized, synchronized audiovisual emote. A random *mood* fixes the
//! pitch band, tempo, rise/fall contour and a matching colour palette; a
//! random *viz* fixes how the strip tracks the pitch. Every tone is
//! preceded by the matching light frame so the show is synchronized by
//! construction. Self-contained and hardware-facing — kept out of the
//! supervisor so the control loop stays purely about safety + teleop.

use std::time::SystemTime;

use crate::robot::car::Car;

/// Tiny dependency-free xorshift PRNG, seeded from the wall clock. Good
/// enough to make the buzzer/LED show feel different every press.
struct Rng(u64);
impl Rng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Rng(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
    /// Uniform float in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// HSV → RGB. `h` in degrees, `s`/`v` in `[0, 1]`.
fn hsv(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let q = |f: f32| ((f + m) * 255.0).round() as u8;
    (q(r), q(g), q(b))
}

/// A pitch near the phrase's (drifting) centre, jittered within the band.
fn band_freq(rng: &mut Rng, centre: f32, lo: i32, hi: i32) -> i32 {
    let j = (rng.unit() - 0.5) * (hi - lo) as f32 * 0.6;
    (centre + j).clamp(lo as f32, hi as f32) as i32
}

/// A colour palette: `at(t, v)` maps a normalized pitch `t` to an RGB at
/// brightness `v`. Each mood carries its own so the lights match the
/// sound's vibe (warm & bright vs cool & dim, etc.).
#[derive(Clone, Copy)]
struct Pal {
    h0: f32,
    span: f32,
    sat: f32,
}
impl Pal {
    fn at(&self, t: f32, v: f32) -> (u8, u8, u8) {
        hsv(self.h0 + t * self.span, self.sat, v.clamp(0.0, 1.0))
    }
}

/// A buzzer/LED personality. `kinds` lists the segment types this mood
/// favours (0 gliss · 1 warble · 2 sustained · 3 blips · 4 tight trill).
#[derive(Clone, Copy)]
struct Mood {
    lo: i32,
    hi: i32,
    tempo: u64,
    contour: i32,
    kinds: &'static [u8],
    pal: Pal,
    base_v: f32,
}

/// Paint the strip for a beat: the lights track pitch. `t` is the
/// normalized pitch (0 = band low, 1 = band high); `viz` selects the
/// look. Always followed immediately by the matching `buzzer.tone`, so
/// audio and light stay in lock-step.
fn light(car: &mut Car, pal: &Pal, viz: u64, t: f32, v: f32) {
    use crate::hal::led::LED_COUNT as N;
    let t = t.clamp(0.0, 1.0);
    car.leds.set_all(0, 0, 0);
    match viz {
        // Scanner: a dot whose position *and* colour ride the pitch.
        0 => {
            let pos = (t * (N as f32 - 1.0)).round() as usize;
            let (r, g, b) = pal.at(t, v);
            car.leds.set(pos, r, g, b);
            let (r2, g2, b2) = pal.at(t, v * 0.25);
            if pos > 0 {
                car.leds.set(pos - 1, r2, g2, b2);
            }
            if pos + 1 < N {
                car.leds.set(pos + 1, r2, g2, b2);
            }
        }
        // VU meter: higher pitch fills more of the ring.
        1 => {
            let lvl = ((t * N as f32).ceil() as usize).clamp(1, N);
            for i in 0..lvl {
                let (r, g, b) = pal.at(i as f32 / N as f32, v);
                car.leds.set(i, r, g, b);
            }
        }
        // Flood: whole strip, colour from pitch.
        _ => {
            let (r, g, b) = pal.at(t, v);
            car.leds.set_all(r, g, b);
        }
    }
    car.leds.show().ok();
}

/// One unified, randomized emote: LEDs and buzzer play together.
/// A random *mood* fixes the pitch band, tempo, rise/fall contour and a
/// matching colour palette; a random *viz* fixes how the lights track
/// the pitch. Every tone is preceded by the matching `light()` frame, so
/// the show is synchronized by construction. Blocks the supervisor loop
/// briefly (< ~2.5 s), like the old scan did.
pub(crate) fn play_emote(car: &mut Car) {
    let mut rng = Rng::new();

    let p = |h0: f32, span: f32, sat: f32| Pal { h0, span, sat };
    let moods: [Mood; 5] = [
        // excited: high, fast, rising — warm→bright rainbow
        Mood {
            lo: 900,
            hi: 3000,
            tempo: 75,
            contour: 1,
            kinds: &[3, 3, 0, 4],
            pal: p(30.0, 300.0, 1.0),
            base_v: 1.0,
        },
        // curious: mid, questioning — teal/green
        Mood {
            lo: 450,
            hi: 1800,
            tempo: 115,
            contour: 1,
            kinds: &[3, 2, 0],
            pal: p(160.0, 90.0, 1.0),
            base_v: 1.0,
        },
        // grumpy: low, slow, falling — deep blue/purple, dim
        Mood {
            lo: 150,
            hi: 850,
            tempo: 155,
            contour: -1,
            kinds: &[2, 0, 3],
            pal: p(225.0, 60.0, 0.9),
            base_v: 0.55,
        },
        // alarmed: harsh fast trills — red/orange
        Mood {
            lo: 700,
            hi: 2400,
            tempo: 60,
            contour: 0,
            kinds: &[4, 4, 3],
            pal: p(0.0, 32.0, 1.0),
            base_v: 1.0,
        },
        // chatty: wide, mixed — full rainbow
        Mood {
            lo: 250,
            hi: 2600,
            tempo: 100,
            contour: 0,
            kinds: &[0, 1, 2, 3],
            pal: p(0.0, 360.0, 1.0),
            base_v: 1.0,
        },
    ];
    let Mood {
        lo,
        hi,
        tempo,
        contour,
        kinds,
        pal,
        base_v,
    } = moods[rng.range(0, 4) as usize];
    let viz = rng.range(0, 2); // 0 scanner · 1 VU · 2 flood
    let span = (hi - lo) as f32;
    let norm = |f: i32| ((f - lo) as f32 / span).clamp(0.0, 1.0);

    let segments = rng.range(4, 7);
    let mut centre = match contour {
        c if c > 0 => lo as f32,
        c if c < 0 => hi as f32,
        _ => (lo + hi) as f32 / 2.0,
    };
    let drift = contour as f32 * span / segments as f32;
    let dur = |rng: &mut Rng, a: u64, b: u64| (rng.range(a, b) * tempo / 100).max(6);

    for _ in 0..segments {
        match kinds[rng.range(0, kinds.len() as u64 - 1) as usize] {
            // gliss: a pitch sweep — the scanner dot slides with it
            0 => {
                let (f0, f1) = (
                    band_freq(&mut rng, centre, lo, hi),
                    band_freq(&mut rng, centre, lo, hi),
                );
                let steps = 14;
                let total = dur(&mut rng, 110, 240);
                for s in 0..steps {
                    let fr = f0 + (f1 - f0) * s as i32 / steps as i32;
                    light(car, &pal, viz, norm(fr), base_v);
                    car.buzzer.tone(fr.max(60) as u32, (total / steps).max(4));
                }
            }
            // warble: alternate two pitches, lights jump with them
            1 => {
                let (a, b) = (
                    band_freq(&mut rng, centre, lo, hi),
                    band_freq(&mut rng, centre, lo, hi),
                );
                for i in 0..rng.range(4, 8) {
                    let fr = if i.is_multiple_of(2) { a } else { b }.max(60);
                    light(car, &pal, viz, norm(fr), base_v);
                    car.buzzer.tone(fr as u32, dur(&mut rng, 16, 32));
                }
            }
            // sustained chirp with a gentle brightness "breath"
            2 => {
                let fr = band_freq(&mut rng, centre, lo, hi).max(60);
                let total = dur(&mut rng, 110, 220);
                let steps = 8;
                for s in 0..steps {
                    let glow = 0.55 + 0.45 * (s as f32 / steps as f32 * std::f32::consts::PI).sin();
                    light(car, &pal, viz, norm(fr), base_v * glow);
                    car.buzzer.tone(fr as u32, (total / steps).max(4));
                }
            }
            // tight trill on two close pitches — strobes
            4 => {
                let a = band_freq(&mut rng, centre, lo, hi);
                let b = a + rng.range(120, 380) as i32;
                for i in 0..rng.range(6, 12) {
                    let fr = if i.is_multiple_of(2) { a } else { b }.max(60);
                    light(car, &pal, viz, norm(fr), base_v);
                    car.buzzer.tone(fr as u32, dur(&mut rng, 9, 18));
                }
            }
            // short blips with little dark gaps
            _ => {
                for _ in 0..rng.range(1, 3) {
                    let fr = band_freq(&mut rng, centre, lo, hi).max(60);
                    light(car, &pal, viz, norm(fr), base_v);
                    car.buzzer.tone(fr as u32, dur(&mut rng, 22, 75));
                    car.leds.set_all(0, 0, 0);
                    car.leds.show().ok();
                    car.buzzer.tone(0, dur(&mut rng, 15, 40));
                }
            }
        }
        centre = (centre + drift).clamp(lo as f32, hi as f32);
        // A short dark pause between most segments.
        if rng.range(0, 2) != 0 {
            car.leds.set_all(0, 0, 0);
            car.leds.show().ok();
            car.buzzer.tone(0, dur(&mut rng, 25, 110));
        }
    }
    car.buzzer.off();
    car.leds.clear().ok();
}
