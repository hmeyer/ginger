//! OV5647 CSI camera via libcamera.
//!
//! Runs a background thread that continuously captures YUYV frames and
//! publishes them behind an Arc so callers can either poll (`get_frame`)
//! or block until a new frame arrives (`wait_frame`). Exposure is driven
//! by the [`super::auto_exposure`] controller.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
#[cfg(feature = "libcamera")]
use std::time::Instant;

#[cfg(feature = "libcamera")]
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

use crate::camera::auto_exposure::ExposureConfig;
#[cfg(feature = "libcamera")]
use crate::camera::auto_exposure::{ae_step, mean_luma, settings_to_brightness};
use crate::{Error, Result};

// Headless replacement for the libcamera capture loop. Same signature,
// selected when the `libcamera` feature is off (CI / dev machines).
#[cfg(not(feature = "libcamera"))]
use super::mock::camera_loop;

#[cfg(feature = "libcamera")]
const YUYV: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);
#[cfg(feature = "libcamera")]
const WARMUP_FRAMES: usize = 5;
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);

// ── Frame ─────────────────────────────────────────────────────────────────────

pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Raw YUYV bytes: 2 bytes per pixel, [Y0 U Y1 V] per 4-byte group.
    pub data: Vec<u8>,
}

// ── Internal shared state ─────────────────────────────────────────────────────

pub(crate) struct FrameState {
    pub(crate) frame: Option<Arc<Frame>>,
    pub(crate) generation: u64,
}

pub(crate) type Shared = Arc<(Mutex<FrameState>, Condvar)>;

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

#[cfg(feature = "libcamera")]
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

#[cfg(feature = "libcamera")]
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

    // The SLAM intrinsics prior assumes this ViewFinder stream is a
    // full-FOV downscale (FOV-derivation is then resolution-agnostic),
    // not a center crop. Log the negotiated mode so that assumption is
    // visible/auditable; a crop would need fx/fy from the full-res
    // pixel focal instead. (Querying libcamera Model/PixelArraySize for
    // hardware confirmation is a deferred follow-up.)
    log::info!("camera: ViewFinder stream negotiated {width}x{height} YUYV (assumed full-FOV)");

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
