//! OV5647 CSI camera via libcamera.
//!
//! Runs a background thread that continuously captures YUYV frames and
//! publishes them behind an Arc so callers can either poll (`get_frame`)
//! or block until a new frame arrives (`wait_frame`).

use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    control::ControlList,
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

// Fixed exposure for motion-blur-free SLAM images.
// AE is disabled; raise ANALOGUE_GAIN if images are too dark.
const EXPOSURE_US: i32 = 8_000; // 8 ms
const ANALOGUE_GAIN: f32 = 8.0; // 8× — tunable for ambient light

// ── Frame ────────────────────────────────────────────────────────────────────

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

// ── Internal shared state ────────────────────────────────────────────────────

struct FrameState {
    frame: Option<Arc<Frame>>,
    generation: u64,
}

type Shared = Arc<(Mutex<FrameState>, Condvar)>;

// ── Camera ───────────────────────────────────────────────────────────────────

pub struct Camera {
    shared: Shared,
    _thread: JoinHandle<()>,
}

impl Camera {
    /// Open the first available camera and start streaming in a background thread.
    /// Blocks until the first real frame is ready (warmup included, ≤10 s).
    pub fn new() -> Result<Self> {
        // Setup handshake: thread sends Ok(()) or Err(msg) once the camera is configured.
        let (setup_tx, setup_rx) =
            std::sync::mpsc::sync_channel::<std::result::Result<(), String>>(1);

        let shared: Shared = Arc::new((
            Mutex::new(FrameState {
                frame: None,
                generation: 0,
            }),
            Condvar::new(),
        ));
        let shared_thread = shared.clone();

        let thread = thread::Builder::new()
            .name("camera".into())
            .spawn(move || camera_loop(shared_thread, setup_tx))
            .map_err(|e| Error::Camera(e.to_string()))?;

        // Wait for setup confirmation
        match setup_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => return Err(Error::Camera(msg)),
            Err(_) => return Err(Error::Camera("camera thread did not start in time".into())),
        }

        // Wait for first real frame (post-warmup)
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
    setup_tx: std::sync::mpsc::SyncSender<std::result::Result<(), String>>,
) {
    if let Err(e) = run_camera(shared, setup_tx.clone()) {
        let _ = setup_tx.try_send(Err(e.to_string()));
    }
}

fn run_camera(
    shared: Shared,
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

    let mut ctrl = ControlList::new();
    ctrl.set(AeEnable(false)).ok();
    ctrl.set(ExposureTime(EXPOSURE_US)).ok();
    ctrl.set(AnalogueGain(ANALOGUE_GAIN)).ok();
    cam.start(Some(&*ctrl))?;
    log::info!(
        "camera: AE off, exposure={}µs ({:.0}ms), gain={:.1}×",
        EXPOSURE_US,
        EXPOSURE_US as f32 / 1000.0,
        ANALOGUE_GAIN
    );
    for req in reqs.drain(..) {
        cam.queue_request(req).map_err(|(_, e)| e)?;
    }

    // Signal successful setup to Camera::new()
    let _ = setup_tx.send(Ok(()));

    let mut warmup = 0usize;

    loop {
        let mut req = frame_rx.recv()?;

        let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
        let bytes_used = fb
            .metadata()
            .and_then(|m| m.planes().get(0).map(|p| p.bytes_used as usize))
            .unwrap_or(width as usize * height as usize * 2);

        if warmup >= WARMUP_FRAMES {
            let data = fb.data()[0][..bytes_used].to_vec();
            let frame = Arc::new(Frame {
                width,
                height,
                data,
            });
            let (lock, cvar) = &*shared;
            let mut st = lock.lock().unwrap();
            st.frame = Some(frame);
            st.generation += 1;
            cvar.notify_all();
        } else {
            warmup += 1;
        }

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        if cam.queue_request(req).map_err(|(_, e)| e).is_err() {
            break;
        }
    }

    cam.stop()?;
    Ok(())
}
