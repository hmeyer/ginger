//! OV5647 CSI camera via libcamera.
//!
//! Runs a background thread that continuously captures YUYV frames and
//! publishes them behind an Arc so callers can either poll (`get_frame`)
//! or block until a new frame arrives (`wait_frame`).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    controls::{AeEnable, AnalogueGain, ExposureTime},
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    request::ReuseFlag,
    stream::StreamRole,
};

use crate::{Error, Result};

const YUYV: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);
const WARMUP_FRAMES: usize = 5;
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);

// ── Exposure control ──────────────────────────────────────────────────────────
//
// Brightness axis: a single 1D control variable in stops above the darkest
// possible setting. 1 stop = 2× the light hitting the sensor.
//   brightness 0  → gain=1×,  exp=0.5 ms       (darkest)
//   brightness 4  → gain=16×, exp=0.5 ms       (gain saturated)
//   brightness 11.6 → gain=16×, exp=100 ms     (brightest)
// The mapping enforces gain-first ramping: gain saturates before exposure
// extends, keeping motion blur bounded for as long as possible.

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
fn brightness_to_settings(b: f32) -> (i32, f32) {
    let b = b.clamp(0.0, AE_MAX_BRIGHTNESS);
    if b <= 4.0 {
        (AE_MIN_EXPOSURE_US, 2.0_f32.powf(b))
    } else {
        let exp = AE_MIN_EXPOSURE_US as f32 * 2.0_f32.powf(b - 4.0);
        ((exp as i32).min(AE_MAX_EXPOSURE_US), AE_MAX_GAIN)
    }
}

/// Inverse of brightness_to_settings: read back brightness from sensor metadata.
fn settings_to_brightness(exp_us: i32, gain: f32) -> f32 {
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
fn mean_luma(data: &[u8]) -> u8 {
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
fn ae_step(cfg: &mut ExposureConfig, luma: u8, applied_brightness: f32) -> (i32, f32) {
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

// ── Frame ─────────────────────────────────────────────────────────────────────

pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Raw YUYV bytes: 2 bytes per pixel, [Y0 U Y1 V] per 4-byte group.
    pub data: Vec<u8>,
}

impl Frame {
    /// Convert YUYV to packed RGB (3 bytes per pixel, row-major).
    pub fn to_rgb(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut rgb = vec![0u8; w * h * 3];
        for (i, chunk) in self.data[..w * h * 2].chunks_exact(4).enumerate() {
            let (y0, u, y1, v) = (
                chunk[0] as i32,
                chunk[1] as i32,
                chunk[2] as i32,
                chunk[3] as i32,
            );
            for (j, &y) in [y0, y1].iter().enumerate() {
                let base = (i * 2 + j) * 3;
                rgb[base] = (y + 1402 * (v - 128) / 1000).clamp(0, 255) as u8;
                rgb[base + 1] =
                    (y - 344 * (u - 128) / 1000 - 714 * (v - 128) / 1000).clamp(0, 255) as u8;
                rgb[base + 2] = (y + 1772 * (u - 128) / 1000).clamp(0, 255) as u8;
            }
        }
        rgb
    }

    /// Write a binary PPM file (viewable without any extra library).
    pub fn save_ppm(&self, path: &str) -> std::io::Result<()> {
        let header = format!("P6\n{} {}\n255\n", self.width, self.height);
        let mut out = header.into_bytes();
        out.extend_from_slice(&self.to_rgb());
        std::fs::write(path, out)
    }
}

// ── Internal shared state ─────────────────────────────────────────────────────

struct FrameState {
    frame: Option<Arc<Frame>>,
    generation: u64,
}

type Shared = Arc<(Mutex<FrameState>, Condvar)>;

// ── Camera ────────────────────────────────────────────────────────────────────

pub struct Camera {
    shared: Shared,
    /// fps × 10 stored as integer for lock-free reads (e.g. 75 = 7.5 fps).
    fps_x10: Arc<AtomicU32>,
    pub exposure_cfg: Arc<Mutex<ExposureConfig>>,
    _thread: JoinHandle<()>,
}

impl Camera {
    /// Current capture rate as measured between the last two frames delivered to callers.
    pub fn fps(&self) -> f32 {
        self.fps_x10.load(Ordering::Relaxed) as f32 / 10.0
    }

    /// Open the first available camera and start streaming in a background thread.
    /// Blocks until the first real frame is ready (warmup included, ≤10 s).
    pub fn new() -> Result<Self> {
        let (setup_tx, setup_rx) =
            std::sync::mpsc::sync_channel::<std::result::Result<(), String>>(1);

        let fps_x10 = Arc::new(AtomicU32::new(0));
        let shared: Shared = Arc::new((
            Mutex::new(FrameState {
                frame: None,
                generation: 0,
            }),
            Condvar::new(),
        ));
        let exposure_cfg = Arc::new(Mutex::new(ExposureConfig::default()));

        let shared_thread = shared.clone();
        let fps_x10_thread = fps_x10.clone();
        let exposure_thread = exposure_cfg.clone();

        let thread = thread::Builder::new()
            .name("camera".into())
            .spawn(move || camera_loop(shared_thread, fps_x10_thread, exposure_thread, setup_tx))
            .map_err(|e| Error::Camera(e.to_string()))?;

        match setup_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => return Err(Error::Camera(msg)),
            Err(_) => return Err(Error::Camera("camera thread did not start in time".into())),
        }

        {
            let (lock, cvar) = shared.as_ref();
            let deadline = std::time::Instant::now() + FIRST_FRAME_TIMEOUT;
            let mut guard = lock.lock().unwrap();
            while guard.frame.is_none() {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return Err(Error::Camera("timeout waiting for first frame".into()));
                }
                let (g, _) = cvar.wait_timeout(guard, remaining).unwrap();
                guard = g;
            }
        }

        Ok(Self {
            shared,
            fps_x10,
            exposure_cfg,
            _thread: thread,
        })
    }

    /// Most recent frame, returned immediately (may be the same as last call).
    pub fn get_frame(&self) -> Arc<Frame> {
        self.shared.0.lock().unwrap().frame.clone().unwrap()
    }

    /// Block until the next new frame arrives, then return it.
    pub fn wait_frame(&self) -> Arc<Frame> {
        let (lock, cvar) = &*self.shared;
        let prev_gen = lock.lock().unwrap().generation;
        let guard = cvar
            .wait_while(lock.lock().unwrap(), |s| s.generation == prev_gen)
            .unwrap();
        guard.frame.clone().unwrap()
    }
}

// ── Background capture loop ───────────────────────────────────────────────────

fn camera_loop(
    shared: Shared,
    fps_x10: Arc<AtomicU32>,
    exposure_cfg: Arc<Mutex<ExposureConfig>>,
    setup_tx: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) {
    if let Err(e) = run_camera(shared, fps_x10, exposure_cfg, setup_tx.clone()) {
        let _ = setup_tx.try_send(Err(e.to_string()));
    }
}

fn run_camera(
    shared: Shared,
    fps_x10: Arc<AtomicU32>,
    exposure_cfg: Arc<Mutex<ExposureConfig>>,
    setup_tx: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mgr = CameraManager::new()?;
    let cameras = mgr.cameras();
    let cam = cameras.get(0).ok_or("no camera found")?;
    let mut cam = cam.acquire()?;

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::ViewFinder])
        .ok_or("could not generate camera configuration")?;
    cfgs.get_mut(0).unwrap().set_pixel_format(YUYV);

    if let CameraConfigurationStatus::Invalid = cfgs.validate() {
        return Err("camera config invalid".into());
    }

    let width = cfgs.get(0).unwrap().get_size().width;
    let height = cfgs.get(0).unwrap().get_size().height;

    cam.configure(&mut cfgs)?;

    let mut alloc = FrameBufferAllocator::new(&cam);
    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream)?;

    let buffers: Vec<_> = buffers
        .into_iter()
        .map(|b| MemoryMappedFrameBuffer::new(b).unwrap())
        .collect();

    let mut reqs: Vec<_> = buffers
        .into_iter()
        .map(|buf| {
            let mut req = cam.create_request(None).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect();

    let (frame_tx, frame_rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        let _ = frame_tx.send(req);
    });

    cam.start(None)?;

    let (init_exp, init_gain) = {
        let cfg = exposure_cfg.lock().unwrap();
        (cfg.current_exposure_us, cfg.current_gain)
    };
    log::info!(
        "camera: AE off, exposure={}µs ({:.0}ms), gain={:.1}× (initial)",
        init_exp,
        init_exp as f32 / 1000.0,
        init_gain
    );
    for mut req in reqs.drain(..) {
        req.controls_mut().set(AeEnable(false)).ok();
        req.controls_mut().set(ExposureTime(init_exp)).ok();
        req.controls_mut().set(AnalogueGain(init_gain)).ok();
        cam.queue_request(req).map_err(|(_, e)| e)?;
    }

    let _ = setup_tx.send(Ok(()));

    let mut warmup = 0usize;
    let mut last_frame_instant = Instant::now();

    loop {
        let mut req = frame_rx.recv()?;

        let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
        let bytes_used = fb
            .metadata()
            .and_then(|m| m.planes().get(0).map(|p| p.bytes_used as usize))
            .unwrap_or(width as usize * height as usize * 2);

        let (next_exp, next_gain) = if warmup >= WARMUP_FRAMES {
            let data = fb.data()[0][..bytes_used].to_vec();

            // Read the exposure/gain the sensor actually applied to this frame.
            let actual_exp = req
                .metadata()
                .get::<ExposureTime>()
                .ok()
                .map(|v| *v)
                .unwrap_or(init_exp);
            let actual_gain = req
                .metadata()
                .get::<AnalogueGain>()
                .ok()
                .map(|v| *v)
                .unwrap_or(init_gain);
            let applied_brightness = settings_to_brightness(actual_exp, actual_gain);

            let luma = mean_luma(&data);
            let (exp, gain) = {
                let mut cfg = exposure_cfg.lock().unwrap();
                cfg.current_exposure_us = actual_exp;
                cfg.current_gain = actual_gain;
                ae_step(&mut cfg, luma, applied_brightness)
            };

            let frame = Arc::new(Frame {
                width,
                height,
                data,
            });

            let now = Instant::now();
            let dt = now.duration_since(last_frame_instant).as_secs_f32();
            last_frame_instant = now;
            if dt > 0.0 {
                let prev = fps_x10.load(Ordering::Relaxed) as f32 / 10.0;
                let raw = 1.0 / dt;
                let smoothed = if prev == 0.0 {
                    raw
                } else {
                    0.2 * raw + 0.8 * prev
                };
                fps_x10.store((smoothed * 10.0) as u32, Ordering::Relaxed);
            }

            let (lock, cvar) = &*shared;
            let mut st = lock.lock().unwrap();
            st.frame = Some(frame);
            st.generation += 1;
            cvar.notify_all();

            (exp, gain)
        } else {
            warmup += 1;
            let cfg = exposure_cfg.lock().unwrap();
            (cfg.current_exposure_us, cfg.current_gain)
        };

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        req.controls_mut().set(AeEnable(false)).ok();
        req.controls_mut().set(ExposureTime(next_exp)).ok();
        req.controls_mut().set(AnalogueGain(next_gain)).ok();
        if cam.queue_request(req).map_err(|(_, e)| e).is_err() {
            break;
        }
    }

    cam.stop()?;
    Ok(())
}
