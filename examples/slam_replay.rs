//! Deterministic offline replay of the SLAM frontend over a recorded /
//! dataset frame sequence — the M2 regression harness.
//!
//! Unlike `slam_bench` (timing on a synthetic frame), this runs the real
//! `detect_features` + `match_descriptors` over an ordered `*.pgm`
//! directory and prints a **stable, timing-free** summary so a code
//! change can be diffed run-to-run (and gated in CI). No camera, no
//! libcamera — build/run headless:
//!
//!   cargo run --no-default-features --example slam_replay -- <frames-dir>
//!
//! `<frames-dir>` is a directory of `*.pgm` (lexicographically ordered),
//! optionally with a sibling `slam.toml` (intrinsics, surfaced here;
//! consumed for real by M3+).

use ginger_rs::slam::brief::{Descriptor, match_descriptors};
use ginger_rs::slam::detect_features;
use ginger_rs::slam::image::GrayImage;
use ginger_slam_core::dataset::{FrameSequence, GrayFrame};

fn to_gray(f: &GrayFrame) -> GrayImage {
    let mut g = GrayImage::new(f.width as usize, f.height as usize);
    g.data.copy_from_slice(&f.data);
    g
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: slam_replay <frames-dir>");
        std::process::exit(2);
    });
    let seq = FrameSequence::open(&dir).unwrap_or_else(|e| {
        eprintln!("cannot open {dir}: {e}");
        std::process::exit(1);
    });
    if seq.is_empty() {
        eprintln!("{dir}: no *.pgm frames");
        std::process::exit(1);
    }

    println!("slam_replay  {dir}  ({} frames)", seq.len());
    match &seq.intrinsics {
        Some(i) => println!(
            "  intrinsics: model={:?} fx={:.1} fov={:.1}° verified={}",
            i.model,
            i.fx,
            i.hfov_deg(),
            i.verified
        ),
        None => println!("  intrinsics: none (M3+ will require slam.toml)"),
    }

    let mut prev: Option<Vec<Descriptor>> = None;
    let (mut tot_kept, mut tot_total, mut tot_matches) = (0u64, 0u64, 0u64);
    for (idx, frame) in seq.iter_frames().enumerate() {
        let frame = frame.unwrap_or_else(|e| {
            eprintln!("frame {idx}: {e}");
            std::process::exit(1);
        });
        let gray = to_gray(&frame);
        let (points, descs, n_total, _) = detect_features(&gray);
        let n_matches = match &prev {
            Some(pd) => match_descriptors(pd, &descs).len(),
            None => 0,
        };
        println!(
            "  f{idx:04}  {:>4}x{:<4}  kept {:>4}  total {:>5}  matches {:>4}",
            frame.width,
            frame.height,
            points.len(),
            n_total,
            n_matches
        );
        tot_kept += points.len() as u64;
        tot_total += n_total as u64;
        tot_matches += n_matches as u64;
        prev = Some(descs);
    }
    let n = seq.len() as u64;
    println!(
        "  ----\n  avg/frame  kept {}  total {}  matches {}",
        tot_kept / n,
        tot_total / n,
        tot_matches / n.max(1)
    );
}
