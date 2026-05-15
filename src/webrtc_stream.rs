//! WebRTC publisher for the camera feed.
//!
//! Pipeline: camera YUYV frames → ffmpeg (`h264_v4l2m2m`, Annex B output) →
//! NAL parser → `TrackLocalStaticSample` → browser `<video>` via
//! `RTCPeerConnection`. Signalling is a minimal WHEP-style POST/answer:
//! the client posts its SDP offer, the server returns its SDP answer.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use log::{info, warn};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use webrtc::api::APIBuilder;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MIME_TYPE_H264, MediaEngine};
use webrtc::ice_transport::ice_connection_state::RTCIceConnectionState;
use webrtc::interceptor::registry::Registry;
use webrtc::media::Sample;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use crate::camera::Camera;

const FFMPEG_BITRATE: &str = "2500000";
const FFMPEG_GOP: &str = "30";
const FRAME_DURATION: Duration = Duration::from_millis(33); // ~30 fps

/// Accept a client SDP offer, set up a PeerConnection that publishes the
/// camera's H.264 feed, and return our SDP answer.
pub async fn whep_handle(
    camera: Arc<Camera>,
    offer_sdp: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut m = MediaEngine::default();
    m.register_default_codecs()?;

    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut m)?;

    let api = APIBuilder::new()
        .with_media_engine(m)
        .with_interceptor_registry(registry)
        .build();

    let pc = Arc::new(api.new_peer_connection(RTCConfiguration::default()).await?);

    // Outbound H.264 track.
    let track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_H264.to_owned(),
            clock_rate: 90_000,
            ..Default::default()
        },
        "video".to_owned(),
        "ginger".to_owned(),
    ));
    let rtp_sender = pc
        .add_track(track.clone() as Arc<dyn webrtc::track::track_local::TrackLocal + Send + Sync>)
        .await?;

    // Drain RTCP from the sender (otherwise it back-pressures and stalls).
    tokio::spawn(async move {
        let mut buf = vec![0u8; 1500];
        while rtp_sender.read(&mut buf).await.is_ok() {}
    });

    // Frame dimensions for the encoder.
    let frame = camera.get_frame();
    let width = frame.width;
    let height = frame.height;
    drop(frame);

    // Spawn ffmpeg + NAL feeder. Tied to PC lifetime via state callback below.
    let cam_for_feeder = camera.clone();
    let track_for_feeder = track.clone();
    let feeder = tokio::spawn(async move {
        if let Err(e) = run_feeder(cam_for_feeder, track_for_feeder, width, height).await {
            warn!("webrtc: feeder ended: {e}");
        }
    });

    // Aborter that lives inside the on_state_change closure: when the PC
    // dies, kill the feeder task (which also drops the ffmpeg Child →
    // kill_on_drop terminates it).
    let feeder_abort = Arc::new(std::sync::Mutex::new(Some(feeder)));
    let feeder_abort_cb = feeder_abort.clone();
    let pc_for_cb = Arc::downgrade(&pc);
    pc.on_peer_connection_state_change(Box::new(move |s| {
        let feeder_abort = feeder_abort_cb.clone();
        let pc_weak = pc_for_cb.clone();
        Box::pin(async move {
            info!("webrtc: pc state = {s}");
            match s {
                RTCPeerConnectionState::Disconnected
                | RTCPeerConnectionState::Failed
                | RTCPeerConnectionState::Closed => {
                    if let Some(h) = feeder_abort.lock().unwrap().take() {
                        h.abort();
                    }
                    if let Some(pc) = pc_weak.upgrade() {
                        let _ = pc.close().await;
                    }
                }
                _ => {}
            }
        })
    }));

    pc.on_ice_connection_state_change(Box::new(|s: RTCIceConnectionState| {
        Box::pin(async move {
            info!("webrtc: ice state = {s}");
        })
    }));

    // Apply offer, build answer, wait for ICE gathering, return SDP.
    let offer = RTCSessionDescription::offer(offer_sdp)?;
    pc.set_remote_description(offer).await?;

    let answer = pc.create_answer(None).await?;
    let mut gather_complete = pc.gathering_complete_promise().await;
    pc.set_local_description(answer).await?;
    let _ = gather_complete.recv().await;

    let local = pc.local_description().await.ok_or("no local description")?;

    // Keep the PeerConnection alive for the session. The state-change closure
    // owns the abort handle; ICE failure / close drops everything.
    leak_pc(pc);

    Ok(local.sdp)
}

/// Intentional leak — the PeerConnection is kept alive by an internal task
/// graph; we lose the explicit handle once we've returned the SDP answer,
/// but its destructors will fire when the on_peer_connection_state_change
/// handler closes it on ICE failure.
fn leak_pc(pc: Arc<RTCPeerConnection>) {
    tokio::spawn(async move {
        // Hold a strong reference until the connection truly closes. Polling
        // is fine here — events drive the work; we just need this future to
        // hang on to `pc` until the connection ends.
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let s = pc.connection_state();
            if matches!(
                s,
                RTCPeerConnectionState::Closed | RTCPeerConnectionState::Failed
            ) {
                break;
            }
        }
        info!("webrtc: session ended");
    });
}

async fn run_feeder(
    camera: Arc<Camera>,
    track: Arc<TrackLocalStaticSample>,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-fflags",
            "+genpts+nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuyv422",
            "-s",
            &format!("{width}x{height}"),
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-c:v",
            "h264_v4l2m2m",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            FFMPEG_BITRATE,
            "-g",
            FFMPEG_GOP,
            // AUD NAL between access units → unambiguous frame splitter.
            "-bsf:v",
            "h264_metadata=aud=insert",
            "-f",
            "h264",
            "-flush_packets",
            "1",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("ffmpeg stdin")?;
    let stdout = child.stdout.take().ok_or("ffmpeg stdout")?;

    // Writer: pulls camera frames, writes YUYV to ffmpeg stdin.
    let cam_for_writer = camera.clone();
    let writer = tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        loop {
            let cam = cam_for_writer.clone();
            let frame = match tokio::task::spawn_blocking(move || cam.wait_frame()).await {
                Ok(f) => f,
                Err(_) => break,
            };
            if stdin.write_all(&frame.data).await.is_err() {
                break;
            }
        }
    });

    // Reader: parses Annex B stream, emits one AU per Sample.
    let mut splitter = AccessUnitSplitter::new();
    let mut reader = stdout;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        for au in splitter.feed(&buf[..n]) {
            track
                .write_sample(&Sample {
                    data: au,
                    duration: FRAME_DURATION,
                    ..Default::default()
                })
                .await?;
        }
    }

    writer.abort();
    Ok(())
}

/// Split an H.264 Annex B byte stream into access units. Each AU starts at
/// an AUD NAL (type 9), which we ask ffmpeg to insert. The first chunk before
/// any AUD is treated as part of the first AU.
struct AccessUnitSplitter {
    pending: Vec<u8>,
}

impl AccessUnitSplitter {
    fn new() -> Self {
        Self {
            pending: Vec::with_capacity(64 * 1024),
        }
    }

    fn feed(&mut self, chunk: &[u8]) -> Vec<Bytes> {
        self.pending.extend_from_slice(chunk);
        let mut out = Vec::new();

        // Search for AUD start sequences in self.pending, leaving incomplete
        // tail bytes (up to 3) plus the current AU prefix.
        // Boundary pattern: <start_code> <AUD nal byte = 0x09>
        // where <start_code> is `00 00 01` or `00 00 00 01`.
        let mut last_cut = 0usize;
        let mut i = 0usize;
        let buf = &self.pending;
        while i + 4 <= buf.len() {
            // Look for 3-byte or 4-byte start code followed by AUD (0x09).
            let three = buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1;
            let four = i + 5 <= buf.len()
                && buf[i] == 0
                && buf[i + 1] == 0
                && buf[i + 2] == 0
                && buf[i + 3] == 1;
            let (sc_len, ok) = if four {
                (4, buf[i + 4] & 0x1f == 0x09)
            } else if three {
                (3, buf[i + 3] & 0x1f == 0x09)
            } else {
                (0, false)
            };
            if ok {
                if i > last_cut {
                    let au = buf[last_cut..i].to_vec();
                    // Skip the very first slice if it's empty (no NALs yet).
                    if au.iter().any(|&b| b != 0) {
                        out.push(Bytes::from(au));
                    }
                }
                last_cut = i;
                i += sc_len + 1;
            } else {
                i += 1;
            }
        }

        // Keep an unsearched tail of up to 4 bytes so we don't miss a start
        // code straddling chunk boundaries.
        let drain_end = last_cut;
        if drain_end > 0 {
            self.pending.drain(..drain_end);
        }
        out
    }
}
