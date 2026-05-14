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

const AE_DEADZONE: i32 = 10;
const AE_GAIN_STEP: f32 = 0.25;
const AE_EXPOSURE_STEP_US: i32 = 200;
const AE_MIN_EXPOSURE_US: i32 = 500;
const AE_MIN_GAIN: f32 = 1.0;
const AE_LUMA_ALPHA: f32 = 0.15; // EMA smoothing; ~7-frame time constant

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExposureMode {
    Auto,
    Manual,
}

pub struct ExposureConfig {
    pub mode: ExposureMode,
    /// AE target brightness (0–255).
    pub target_luma: u8,
    /// AE ceiling for exposure (µs). Keeps motion blur bounded.
    pub max_exposure_us: i32,
    /// AE ceiling for gain.
    pub max_gain: f32,
    pub manual_exposure_us: i32,
    pub manual_gain: f32,
    /// Live readback written by the camera thread.
    pub current_exposure_us: i32,
    pub current_gain: f32,
    pub current_luma: u8,
    luma_ema: f32, // internal smoothed luma, not exposed to UI
}

impl Default for ExposureConfig {
    fn default() -> Self {
        Self {
            mode: ExposureMode::Auto,
            target_luma: 128,
            max_exposure_us: 100_000,
            max_gain: 16.0,
            manual_exposure_us: 8_000,
            manual_gain: 8.0,
            current_exposure_us: 8_000,
            current_gain: 8.0,
            current_luma: 0,
            luma_ema: 0.0,
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

/// Run one AE step: gain-first when darkening (keeps exposure short for SLAM),
/// exposure-first when brightening. Returns the (exposure_us, gain) to apply.
fn ae_step(cfg: &mut ExposureConfig, luma: u8) -> (i32, f32) {
    // Smooth luma with EMA to filter transient bright spots (windows, reflections).
    cfg.luma_ema = if cfg.luma_ema == 0.0 {
        luma as f32
    } else {
        AE_LUMA_ALPHA * luma as f32 + (1.0 - AE_LUMA_ALPHA) * cfg.luma_ema
    };
    let smoothed = cfg.luma_ema as u8;
    cfg.current_luma = smoothed;
    let error = smoothed as i32 - cfg.target_luma as i32;
    if error < -AE_DEADZONE {
        // Too dark: raise gain first, then exposure.
        if cfg.current_gain < cfg.max_gain {
            cfg.current_gain = (cfg.current_gain + AE_GAIN_STEP).min(cfg.max_gain);
        } else {
            cfg.current_exposure_us =
                (cfg.current_exposure_us + AE_EXPOSURE_STEP_US).min(cfg.max_exposure_us);
        }
    } else if error > AE_DEADZONE {
        // Too bright: lower exposure first, then gain.
        if cfg.current_exposure_us > AE_MIN_EXPOSURE_US {
            cfg.current_exposure_us =
                (cfg.current_exposure_us - AE_EXPOSURE_STEP_US).max(AE_MIN_EXPOSURE_US);
        } else {
            cfg.current_gain = (cfg.current_gain - AE_GAIN_STEP).max(AE_MIN_GAIN);
        }
    }
    (cfg.current_exposure_us, cfg.current_gain)
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

    /// Convert YUYV to packed RGB at a reduced resolution by sampling every Nth pixel.
    /// This avoids converting pixels that will be discarded, making it proportionally faster.
    /// Returns `(rgb_bytes, out_width, out_height)`.
    pub fn to_rgb_scaled(&self, scale: f32) -> (Vec<u8>, u32, u32) {
        let sw = self.width as usize;
        let sh = self.height as usize;
        let dw = ((sw as f32 * scale) as usize).max(1);
        let dh = ((sh as f32 * scale) as usize).max(1);
        let x_stride = sw as f32 / dw as f32;
        let y_stride = sh as f32 / dh as f32;

        let mut rgb = vec![0u8; dw * dh * 3];
        for dy in 0..dh {
            let sy = (dy as f32 * y_stride) as usize;
            let row_base = sy * sw;
            for dx in 0..dw {
                let sx = (dx as f32 * x_stride) as usize;
                // YUYV: 4 bytes represent 2 pixels [Y0 U Y1 V].
                // Pixel at column sx lives in chunk (sx/2); Y is at byte 0 or 2.
                let chunk = &self.data[(row_base + sx) / 2 * 4..];
                let y = if sx.is_multiple_of(2) {
                    chunk[0] as i32
                } else {
                    chunk[2] as i32
                };
                let u = chunk[1] as i32;
                let v = chunk[3] as i32;

                let base = (dy * dw + dx) * 3;
                rgb[base] = (y + 1402 * (v - 128) / 1000).clamp(0, 255) as u8;
                rgb[base + 1] =
                    (y - 344 * (u - 128) / 1000 - 714 * (v - 128) / 1000).clamp(0, 255) as u8;
                rgb[base + 2] = (y + 1772 * (u - 128) / 1000).clamp(0, 255) as u8;
            }
        }
        (rgb, dw as u32, dh as u32)
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
    let mut verified = false;
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

            // Verify sensor is applying our settings (once, post-warmup).
            if !verified {
                let actual_exp = req.metadata().get::<ExposureTime>().ok().map(|v| *v);
                let actual_gain = req.metadata().get::<AnalogueGain>().ok().map(|v| *v);
                log::info!(
                    "camera: sensor reports exposure={:?}µs  gain={:?}×",
                    actual_exp,
                    actual_gain
                );
                verified = true;
            }

            let luma = mean_luma(&data);
            let (exp, gain) = {
                let mut cfg = exposure_cfg.lock().unwrap();
                match cfg.mode {
                    ExposureMode::Auto => ae_step(&mut cfg, luma),
                    ExposureMode::Manual => {
                        cfg.luma_ema = if cfg.luma_ema == 0.0 {
                            luma as f32
                        } else {
                            AE_LUMA_ALPHA * luma as f32 + (1.0 - AE_LUMA_ALPHA) * cfg.luma_ema
                        };
                        cfg.current_luma = cfg.luma_ema as u8;
                        cfg.current_exposure_us = cfg.manual_exposure_us;
                        cfg.current_gain = cfg.manual_gain;
                        (cfg.manual_exposure_us, cfg.manual_gain)
                    }
                }
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
