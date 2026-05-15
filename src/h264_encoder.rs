//! Direct V4L2 stateful M2M H.264 encoder for the Pi's bcm2835-codec
//! (`/dev/video11`).
//!
//! ffmpeg's `h264_v4l2m2m` works fine but gives no runtime control over
//! bitrate or keyframes. WebRTC needs both: drop the bitrate when the
//! receiver's bandwidth estimate falls, and emit an immediate IDR when the
//! browser sends a Picture Loss Indication. Talking to the codec directly
//! gives us `VIDIOC_S_CTRL` on a live stream with no re-encode glitch.
//!
//! The device is multiplanar M2M: an OUTPUT_MPLANE queue we feed planar
//! YUV420 into, and a CAPTURE_MPLANE queue that yields Annex-B H.264 access
//! units (one per buffer, SPS/PPS repeated before each IDR).
//!
//! All control changes are funnelled through the single encoder thread via
//! atomics — no V4L2 ioctl is ever issued from two threads at once.

use std::io;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use v4l::v4l_sys::*;
use v4l::v4l2;

// ── Constants not surfaced as plain consts by bindgen ─────────────────────────

const V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE: u32 = 9;
const V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE: u32 = 10;
const V4L2_MEMORY_MMAP: u32 = 1;
const V4L2_FIELD_NONE: u32 = 1;

const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}
const V4L2_PIX_FMT_YUV420: u32 = fourcc(b'Y', b'U', b'1', b'2');
const V4L2_PIX_FMT_H264: u32 = fourcc(b'H', b'2', b'6', b'4');

// Codec control IDs (V4L2_CID_CODEC_BASE = 0x00990000 | 0x900).
const V4L2_CID_CODEC_BASE: u32 = 0x0099_0900;
const V4L2_CID_MPEG_VIDEO_GOP_SIZE: u32 = V4L2_CID_CODEC_BASE + 203;
const V4L2_CID_MPEG_VIDEO_BITRATE_MODE: u32 = V4L2_CID_CODEC_BASE + 206;
const V4L2_CID_MPEG_VIDEO_BITRATE: u32 = V4L2_CID_CODEC_BASE + 207;
const V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER: u32 = V4L2_CID_CODEC_BASE + 226;
const V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME: u32 = V4L2_CID_CODEC_BASE + 229;
const V4L2_MPEG_VIDEO_BITRATE_MODE_VBR: i32 = 0;

const DEVICE: &str = "/dev/video11";
const NUM_OUT_BUFFERS: usize = 4;
const NUM_CAP_BUFFERS: usize = 4;
// Worst-case compressed access-unit size the driver should allocate.
const CAP_BUFFER_SIZE: u32 = 1024 * 1024;

// ── Shared control surface ────────────────────────────────────────────────────

/// Cheap-to-clone handle other threads use to steer the encoder. The encoder
/// thread reads these once per frame; no ioctl crosses a thread boundary.
#[derive(Clone)]
pub struct EncoderControl {
    desired_bitrate: Arc<AtomicU32>,
    force_keyframe: Arc<AtomicBool>,
}

impl EncoderControl {
    /// Request a new target bitrate (bits/sec). Applied before the next frame.
    pub fn set_bitrate(&self, bps: u32) {
        self.desired_bitrate.store(bps, Ordering::Relaxed);
    }

    /// Ask the encoder to emit an IDR on the next frame (e.g. on RTCP PLI).
    pub fn request_keyframe(&self) {
        self.force_keyframe.store(true, Ordering::Relaxed);
    }
}

pub struct Encoded {
    pub data: Vec<u8>,
    pub keyframe: bool,
}

struct MappedBuffer {
    ptr: *mut c_void,
    len: usize,
}

// SAFETY: the mmap pointers are only ever touched by the single encoder
// thread that owns the `H264Encoder`. The struct is moved onto that thread
// and never shared.
unsafe impl Send for H264Encoder {}

pub struct H264Encoder {
    fd: i32,
    width: u32,
    height: u32,
    out_buffers: Vec<MappedBuffer>,
    cap_buffers: Vec<MappedBuffer>,
    /// Y-plane stride the driver chose for the OUTPUT buffer. bcm2835 pads
    /// it (e.g. 800 → 832); chroma stride is half this. Writing packed
    /// (stride == width) produces a sheared, garbled image.
    out_y_stride: usize,
    /// Total OUTPUT plane size the driver expects (Y + U + V, strided).
    out_sizeimage: usize,
    /// Index of the next OUTPUT buffer to use (round-robin; lock-step so it
    /// is always free by the time we come back to it).
    next_out: usize,
    applied_bitrate: u32,
    control: EncoderControl,
}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        unsafe {
            let mut t = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE as i32;
            let _ = v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_STREAMOFF,
                &mut t as *mut _ as *mut c_void,
            );
            let mut t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE as i32;
            let _ = v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_STREAMOFF,
                &mut t as *mut _ as *mut c_void,
            );
            for b in self.out_buffers.iter().chain(self.cap_buffers.iter()) {
                let _ = v4l2::munmap(b.ptr, b.len);
            }
            let _ = v4l2::close(self.fd);
        }
    }
}

impl H264Encoder {
    pub fn control(&self) -> EncoderControl {
        self.control.clone()
    }

    pub fn new(width: u32, height: u32, fps: u32, bitrate: u32) -> io::Result<Self> {
        let fd = v4l2::open(DEVICE, libc::O_RDWR)?;
        let mut enc = Self {
            fd,
            width,
            height,
            out_buffers: Vec::new(),
            cap_buffers: Vec::new(),
            out_y_stride: width as usize,
            out_sizeimage: (width * height * 3 / 2) as usize,
            next_out: 0,
            applied_bitrate: bitrate,
            control: EncoderControl {
                desired_bitrate: Arc::new(AtomicU32::new(bitrate)),
                force_keyframe: Arc::new(AtomicBool::new(false)),
            },
        };

        if let Err(e) = enc.configure(fps, bitrate) {
            // Prevent Drop from running a half-initialised teardown, then
            // close the fd ourselves.
            std::mem::forget(enc);
            let _ = v4l2::close(fd);
            return Err(e);
        }
        Ok(enc)
    }

    fn configure(&mut self, fps: u32, bitrate: u32) -> io::Result<()> {
        // OUTPUT (we feed raw YUV420 here). Honour the driver's padded
        // stride / size — bcm2835 rounds 800 up to 832.
        let (out_bpl, out_size) = self.set_format(
            V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            V4L2_PIX_FMT_YUV420,
            self.width * self.height * 3 / 2,
            self.width,
        )?;
        self.out_y_stride = out_bpl as usize;
        self.out_sizeimage = out_size as usize;
        // CAPTURE (encoded H.264 comes out here).
        self.set_format(
            V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
            V4L2_PIX_FMT_H264,
            CAP_BUFFER_SIZE,
            0,
        )?;

        // VBR so the encoder can spend fewer bits on static scenes; the
        // bitrate ceiling is what we retarget for congestion control.
        self.set_ctrl(
            V4L2_CID_MPEG_VIDEO_BITRATE_MODE,
            V4L2_MPEG_VIDEO_BITRATE_MODE_VBR,
        )?;
        self.set_ctrl(V4L2_CID_MPEG_VIDEO_BITRATE, bitrate as i32)?;
        // One IDR per second; PLI can still force extra ones on demand.
        self.set_ctrl(V4L2_CID_MPEG_VIDEO_GOP_SIZE, fps as i32)?;
        // Repeat SPS/PPS before every IDR so a WebRTC receiver that joins
        // late (or recovers via PLI) can decode without a separate config.
        self.set_ctrl(V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER, 1)?;

        self.request_buffers(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, NUM_OUT_BUFFERS as u32)?;
        self.request_buffers(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, NUM_CAP_BUFFERS as u32)?;

        self.out_buffers = self.map_buffers(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, NUM_OUT_BUFFERS)?;
        self.cap_buffers = self.map_buffers(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, NUM_CAP_BUFFERS)?;

        // Pre-queue all CAPTURE buffers so the encoder has somewhere to write.
        for i in 0..NUM_CAP_BUFFERS {
            self.queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, 0)?;
        }

        self.stream_on(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)?;
        self.stream_on(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)?;
        Ok(())
    }

    /// Set a queue format and return the geometry the driver actually chose
    /// `(bytesperline, sizeimage)` — bcm2835 pads the stride, so callers must
    /// honour the returned values rather than their requested ones.
    fn set_format(
        &self,
        buf_type: u32,
        pixelformat: u32,
        sizeimage: u32,
        bytesperline: u32,
    ) -> io::Result<(u32, u32)> {
        unsafe {
            let mut fmt: v4l2_format = std::mem::zeroed();
            fmt.type_ = buf_type;
            {
                let pix = &mut fmt.fmt.pix_mp;
                pix.width = self.width;
                pix.height = self.height;
                pix.pixelformat = pixelformat;
                pix.field = V4L2_FIELD_NONE;
                pix.num_planes = 1;
                pix.plane_fmt[0].sizeimage = sizeimage;
                pix.plane_fmt[0].bytesperline = bytesperline;
            }
            v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_S_FMT,
                &mut fmt as *mut _ as *mut c_void,
            )?;
            // pix_mp is #[repr(C, packed)]; read fields unaligned.
            let pix = std::ptr::addr_of!(fmt.fmt.pix_mp);
            let bpl =
                std::ptr::read_unaligned(std::ptr::addr_of!((*pix).plane_fmt[0].bytesperline));
            let si = std::ptr::read_unaligned(std::ptr::addr_of!((*pix).plane_fmt[0].sizeimage));
            Ok((bpl, si))
        }
    }

    fn set_ctrl(&self, id: u32, value: i32) -> io::Result<()> {
        unsafe {
            let mut ctrl: v4l2_control = std::mem::zeroed();
            ctrl.id = id;
            ctrl.value = value;
            v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_S_CTRL,
                &mut ctrl as *mut _ as *mut c_void,
            )
        }
    }

    fn request_buffers(&self, buf_type: u32, count: u32) -> io::Result<()> {
        unsafe {
            let mut req: v4l2_requestbuffers = std::mem::zeroed();
            req.count = count;
            req.type_ = buf_type;
            req.memory = V4L2_MEMORY_MMAP;
            v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_REQBUFS,
                &mut req as *mut _ as *mut c_void,
            )
        }
    }

    fn map_buffers(&self, buf_type: u32, count: usize) -> io::Result<Vec<MappedBuffer>> {
        let mut mapped = Vec::with_capacity(count);
        for i in 0..count {
            unsafe {
                let mut planes: [v4l2_plane; 1] = std::mem::zeroed();
                let mut buf: v4l2_buffer = std::mem::zeroed();
                buf.index = i as u32;
                buf.type_ = buf_type;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.length = 1;
                buf.m.planes = planes.as_mut_ptr();
                v4l2::ioctl(
                    self.fd,
                    v4l2::vidioc::VIDIOC_QUERYBUF,
                    &mut buf as *mut _ as *mut c_void,
                )?;
                let len = planes[0].length as usize;
                let offset = planes[0].m.mem_offset as libc::off_t;
                let ptr = v4l2::mmap(
                    ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd,
                    offset,
                )?;
                mapped.push(MappedBuffer { ptr, len });
            }
        }
        Ok(mapped)
    }

    fn queue_buffer(&self, buf_type: u32, index: usize, bytesused: u32) -> io::Result<()> {
        unsafe {
            let mut planes: [v4l2_plane; 1] = std::mem::zeroed();
            planes[0].bytesused = bytesused;
            planes[0].length = if buf_type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE {
                self.out_buffers[index].len as u32
            } else {
                self.cap_buffers[index].len as u32
            };
            let mut buf: v4l2_buffer = std::mem::zeroed();
            buf.index = index as u32;
            buf.type_ = buf_type;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.length = 1;
            buf.m.planes = planes.as_mut_ptr();
            v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_QBUF,
                &mut buf as *mut _ as *mut c_void,
            )
        }
    }

    /// Dequeue one buffer of `buf_type`. Returns `(index, bytesused, flags)`.
    fn dequeue_buffer(&self, buf_type: u32) -> io::Result<(usize, u32, u32)> {
        unsafe {
            let mut planes: [v4l2_plane; 1] = std::mem::zeroed();
            let mut buf: v4l2_buffer = std::mem::zeroed();
            buf.type_ = buf_type;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.length = 1;
            buf.m.planes = planes.as_mut_ptr();
            v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_DQBUF,
                &mut buf as *mut _ as *mut c_void,
            )?;
            Ok((buf.index as usize, planes[0].bytesused, buf.flags))
        }
    }

    fn stream_on(&self, buf_type: u32) -> io::Result<()> {
        unsafe {
            let mut t = buf_type as i32;
            v4l2::ioctl(
                self.fd,
                v4l2::vidioc::VIDIOC_STREAMON,
                &mut t as *mut _ as *mut c_void,
            )
        }
    }

    fn apply_pending_controls(&mut self) {
        let want = self.control.desired_bitrate.load(Ordering::Relaxed);
        if want != self.applied_bitrate
            && want > 0
            && self
                .set_ctrl(V4L2_CID_MPEG_VIDEO_BITRATE, want as i32)
                .is_ok()
        {
            log::debug!("h264: bitrate {} → {} bps", self.applied_bitrate, want);
            self.applied_bitrate = want;
        }
        if self.control.force_keyframe.swap(false, Ordering::Relaxed) {
            log::info!("h264: forcing keyframe (RTCP PLI/FIR)");
            let _ = self.set_ctrl(V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME, 1);
        }
    }

    /// Convert packed YUYV → planar I420 straight into OUTPUT buffer `idx`,
    /// honouring the driver's Y stride (`out_y_stride`) and chroma stride
    /// (half that). Chroma is vertically subsampled by taking even rows.
    fn write_i420(&self, idx: usize, yuyv: &[u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let ys = self.out_y_stride; // Y stride (padded, e.g. 832)
        let cs = ys / 2; // chroma stride
        let ch = h / 2; // chroma rows
        let dst = self.out_buffers[idx].ptr as *mut u8;
        let u_off = ys * h;
        let v_off = u_off + cs * ch;
        let src_row = w * 2; // YUYV bytes per row

        unsafe {
            for row in 0..h {
                let s = &yuyv[row * src_row..row * src_row + src_row];
                let y_dst = dst.add(row * ys);
                for (px, g) in s.chunks_exact(4).enumerate() {
                    *y_dst.add(px * 2) = g[0];
                    *y_dst.add(px * 2 + 1) = g[2];
                }
                if row % 2 == 0 {
                    let crow = row / 2;
                    let u_dst = dst.add(u_off + crow * cs);
                    let v_dst = dst.add(v_off + crow * cs);
                    for (cx, g) in s.chunks_exact(4).enumerate() {
                        *u_dst.add(cx) = g[1];
                        *v_dst.add(cx) = g[3];
                    }
                }
            }
        }
    }

    /// Encode one packed-YUYV (4:2:2) camera frame and return the resulting
    /// H.264 access unit. The frame is converted to planar I420 *directly
    /// into the mmap'd OUTPUT buffer* using the driver's padded stride.
    /// Lock-step (1 in → 1 out); the bcm2835 encoder never emits B-frames so
    /// there is no reordering delay.
    pub fn encode(&mut self, yuyv: &[u8]) -> io::Result<Encoded> {
        self.apply_pending_controls();

        let idx = self.next_out;
        self.next_out = (self.next_out + 1) % NUM_OUT_BUFFERS;

        let w = self.width as usize;
        let h = self.height as usize;
        if yuyv.len() < w * h * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame smaller than expected YUYV size",
            ));
        }
        self.write_i420(idx, yuyv);
        self.queue_buffer(
            V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            idx,
            self.out_sizeimage as u32,
        )?;

        // Reclaim the OUTPUT buffer the encoder is done reading.
        let _ = self.dequeue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)?;

        // Pull the encoded access unit.
        let (cap_idx, bytesused, flags) =
            self.dequeue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)?;
        let src = &self.cap_buffers[cap_idx];
        let n = (bytesused as usize).min(src.len);
        let mut data = vec![0u8; n];
        unsafe {
            ptr::copy_nonoverlapping(src.ptr as *const u8, data.as_mut_ptr(), n);
        }
        let keyframe = flags & V4L2_BUF_FLAG_KEYFRAME != 0;

        // Hand the CAPTURE buffer back to the encoder.
        self.queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, cap_idx, 0)?;

        Ok(Encoded { data, keyframe })
    }
}
