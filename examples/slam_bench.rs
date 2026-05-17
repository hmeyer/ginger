//! Camera-free A/B harness for the SLAM frontend.
//!
//! Runs the exact `detect_features` + `match_descriptors` path on a
//! deterministic, realistically-textured synthetic frame so the per-stage
//! `StageMs` breakdown is reproducible across code changes (no camera, no
//! scene variance). Use it to measure a fix: run, change code, run again,
//! compare. The numbers track the live HUD's stage breakdown.
//!
//!   cargo run --release --example slam_bench -- [W] [H] [ITERS]
//!
//! Defaults: 640x480, 40 iters (first 5 discarded as warmup).

use std::time::Instant;

use ginger_rs::slam::brief::match_descriptors;
use ginger_rs::slam::detect_features;
use ginger_rs::slam::image::GrayImage;

/// Deterministic multi-octave value noise → a frame with realistic corner
/// density (smooth structure + finer detail), not a pathological
/// checkerboard. `shift` translates the pattern so frame B differs from A
/// like consecutive video frames (gives the matcher real work).
fn textured(w: usize, h: usize, shift: f32) -> GrayImage {
    let mut g = GrayImage::new(w, h);
    let hash = |x: i32, y: i32| {
        let mut n = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263)) as u32;
        n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
        ((n ^ (n >> 16)) & 0xff) as f32 / 255.0
    };
    let smooth = |fx: f32, fy: f32| {
        let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * (t * t * (3.0 - 2.0 * t));
        let top = lerp(hash(x0, y0), hash(x0 + 1, y0), tx);
        let bot = lerp(hash(x0, y0 + 1), hash(x0 + 1, y0 + 1), tx);
        lerp(top, bot, ty)
    };
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 + shift, y as f32 + shift);
            // Octaves: coarse structure + mid + fine detail (corners live
            // in the mid/fine bands, like a real textured scene).
            let v = 0.55 * smooth(fx / 48.0, fy / 48.0)
                + 0.30 * smooth(fx / 12.0, fy / 12.0)
                + 0.15 * smooth(fx / 4.0, fy / 4.0);
            g.data[y * w + x] = (v * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    g
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(640);
    let h: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(480);
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);
    let warmup = 5.min(iters / 4);

    // Two frames, one shifted, mimicking consecutive camera frames.
    let frame_a = textured(w, h, 0.0);
    let frame_b = textured(w, h, 2.5);
    let (_, descs_prev, _, _) = detect_features(&frame_a);

    let (mut acc, mut match_ms) = (StageAcc::default(), 0.0f64);
    let (mut n_total, mut n_kept) = (0u32, 0usize);
    for i in 0..iters {
        // Alternate frames so the work is steady-state, not cached.
        let frame = if i.is_multiple_of(2) {
            &frame_b
        } else {
            &frame_a
        };
        let (points, descs, nt, st) = detect_features(frame);

        let tm = Instant::now();
        let m = match_descriptors(&descs_prev, &descs);
        let mms = tm.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&m);

        if i >= warmup {
            acc.add(&st);
            match_ms += mms;
            n_total = nt;
            n_kept = points.len();
        }
    }

    let n = (iters - warmup) as f64;
    let total = acc.gray + acc.pyramid + acc.fast + acc.blur + acc.orient + acc.describe;
    let total = total / n + match_ms / n;
    println!("slam_bench  {w}x{h}  {} iters (avg, ms)", iters - warmup);
    println!("  gray      {:>7.2}", acc.gray / n);
    println!("  pyramid   {:>7.2}", acc.pyramid / n);
    println!("  fast      {:>7.2}", acc.fast / n);
    println!("  blur      {:>7.2}", acc.blur / n);
    println!(
        "  orient    {:>7.2}   <- pre-cap corner set",
        acc.orient / n
    );
    println!(
        "  describe  {:>7.2}   <- pre-cap corner set",
        acc.describe / n
    );
    println!("  match     {:>7.2}", match_ms / n);
    println!("  ----------------");
    println!("  TOTAL     {total:>7.2}   ({:.1} fps)", 1000.0 / total);
    println!("  corners   {n_kept}/{n_total} kept/total");
}

#[derive(Default)]
struct StageAcc {
    gray: f64,
    pyramid: f64,
    fast: f64,
    blur: f64,
    orient: f64,
    describe: f64,
}
impl StageAcc {
    fn add(&mut self, s: &ginger_rs::slam::StageMs) {
        self.gray += s.gray as f64;
        self.pyramid += s.pyramid as f64;
        self.fast += s.fast as f64;
        self.blur += s.blur as f64;
        self.orient += s.orient as f64;
        self.describe += s.describe as f64;
    }
}
