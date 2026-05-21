//! Headless mock camera (no libcamera) — used in CI, on dev machines,
//! and as the frame source for the offline replay harness.
//!
//! It mirrors the real backend's [`camera_loop`](super::capture) exactly:
//! same signature, same `Shared` frame/generation + condvar protocol, so
//! [`Camera`](super::capture::Camera) and everything above it (the SLAM
//! frontend thread, the server, `/api/slam/*`) run unchanged.
//!
//! Frame source:
//! - `GINGER_MOCK_FRAMES=/dir` — replay a directory of `*.pgm` in order,
//!   looped (the offline replay path; deterministic).
//! - unset — a deterministic shifting value-noise scene so frame-to-frame
//!   matching has real work (smoke / CI default).
//!
//! Frames are emitted as YUYV (luma in the even bytes, neutral 128
//! chroma) so `gray_from_yuyv` recovers the luma exactly.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ginger_slam_core::dataset::{FrameSequence, GrayFrame};
use log::{info, warn};

use super::auto_exposure::ExposureConfig;
use super::capture::{Frame, Shared};

/// Synthetic fallback resolution (matches the OV5647 ViewFinder default).
const SYNTH_W: u32 = 800;
const SYNTH_H: u32 = 600;
/// ~10 fps, matching the real frontend's steady-state cadence.
const FRAME_PERIOD: Duration = Duration::from_millis(100);

/// Pack a grayscale buffer into a YUYV [`Frame`] (even=luma, odd=128).
fn gray_to_yuyv(width: u32, height: u32, luma: &[u8]) -> Frame {
    let n = (width as usize) * (height as usize);
    let mut data = vec![128u8; n * 2];
    for (k, &y) in luma.iter().take(n).enumerate() {
        data[2 * k] = y;
    }
    Frame {
        width,
        height,
        data,
    }
}

/// Integer value-noise hash, `(x, y) → [0, 1]`. Pure function (not an
/// RNG); kept local to the mock camera because it is the only synthetic
/// texture source in production code.
#[inline]
fn noise_u8(x: i32, y: i32) -> f32 {
    let mut n = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263)) as u32;
    n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
    ((n ^ (n >> 16)) & 0xff) as f32 / 255.0
}

/// Deterministic multi-octave value noise, translated by `shift` so
/// consecutive frames differ like a slowly panning camera.
fn synthetic(shift: f32) -> Frame {
    let (w, h) = (SYNTH_W as usize, SYNTH_H as usize);
    let hash = noise_u8;
    let smooth = |fx: f32, fy: f32| {
        let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * (t * t * (3.0 - 2.0 * t));
        let top = lerp(hash(x0, y0), hash(x0 + 1, y0), tx);
        let bot = lerp(hash(x0, y0 + 1), hash(x0 + 1, y0 + 1), tx);
        lerp(top, bot, ty)
    };
    let mut luma = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 + shift, y as f32 + shift);
            let v = 0.55 * smooth(fx / 48.0, fy / 48.0)
                + 0.30 * smooth(fx / 12.0, fy / 12.0)
                + 0.15 * smooth(fx / 4.0, fy / 4.0);
            luma[y * w + x] = (v * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    gray_to_yuyv(SYNTH_W, SYNTH_H, &luma)
}

enum Source {
    /// Pre-decoded recorded frames, replayed in order and looped.
    Replay(Vec<GrayFrame>),
    Synthetic,
}

impl Source {
    fn open() -> Self {
        match std::env::var("GINGER_MOCK_FRAMES") {
            Ok(dir) => match FrameSequence::open(&dir) {
                Ok(seq) if !seq.is_empty() => {
                    let frames: Result<Vec<_>, _> = seq.iter_frames().collect();
                    match frames {
                        Ok(f) => {
                            info!("mock camera: replaying {} frames from {dir}", f.len());
                            Source::Replay(f)
                        }
                        Err(e) => {
                            warn!("mock camera: {dir} unreadable ({e}); synthetic scene");
                            Source::Synthetic
                        }
                    }
                }
                Ok(_) => {
                    warn!("mock camera: {dir} has no *.pgm; synthetic scene");
                    Source::Synthetic
                }
                Err(e) => {
                    warn!("mock camera: cannot open {dir} ({e}); synthetic scene");
                    Source::Synthetic
                }
            },
            Err(_) => {
                info!("mock camera: no GINGER_MOCK_FRAMES; synthetic scene");
                Source::Synthetic
            }
        }
    }

    fn frame(&self, i: usize) -> Frame {
        match self {
            Source::Replay(frames) => {
                let g = &frames[i % frames.len()];
                gray_to_yuyv(g.width, g.height, &g.data)
            }
            Source::Synthetic => synthetic((i as f32) * 1.5),
        }
    }
}

/// Drop-in replacement for the libcamera `camera_loop`: publish frames
/// into `shared` with a monotonically increasing generation + condvar
/// notify, exactly like the real backend. `exposure_cfg` is unused (no
/// sensor) but kept for signature parity.
pub(crate) fn camera_loop(
    shared: Shared,
    fps_x10: Arc<AtomicU32>,
    _exposure_cfg: Arc<Mutex<ExposureConfig>>,
    setup_tx: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) {
    let source = Source::open();
    let _ = setup_tx.send(Ok(()));
    fps_x10.store(100, Ordering::Relaxed); // nominal 10.0 fps

    let mut i: usize = 0;
    loop {
        let frame = Arc::new(source.frame(i));
        {
            let (lock, cvar) = &*shared;
            let mut st = lock.lock().unwrap();
            st.frame = Some(frame);
            st.generation += 1;
            cvar.notify_all();
        }
        i += 1;
        std::thread::sleep(FRAME_PERIOD);
    }
}
