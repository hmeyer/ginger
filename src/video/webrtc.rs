//! WebRTC publisher for the camera feed.
//!
//! Pipeline: camera YUYV frames → `to_i420()` → in-process V4L2 H.264
//! encoder (the Pi's bcm2835 hardware codec) → `TrackLocalStaticSample` →
//! browser `<video>` via `RTCPeerConnection`. Signalling is a minimal
//! WHEP-style POST/answer.
//!
//! Adaptive: RTCP feedback from the receiver drives the encoder. A Picture
//! Loss Indication / Full Intra Request forces an immediate IDR; the
//! Receiver-Estimated Max Bitrate (and packet-loss in Receiver Reports)
//! retargets the encoder bitrate with no re-encode glitch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use log::{info, warn};

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
use webrtc::rtcp::payload_feedbacks::full_intra_request::FullIntraRequest;
use webrtc::rtcp::payload_feedbacks::picture_loss_indication::PictureLossIndication;
use webrtc::rtcp::receiver_report::ReceiverReport;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use crate::camera::Camera;
use crate::video::h264::{EncoderControl, H264Encoder};

const FPS: u32 = 30;
const FRAME_DURATION: Duration = Duration::from_millis(33);

// Bitrate envelope for adaptation (bits/sec).
const INITIAL_BITRATE: u32 = 2_500_000;
const MIN_BITRATE: u32 = 300_000;
const MAX_BITRATE: u32 = 4_000_000;
// AIMD on Receiver-Report loss. fraction_lost is 0..=255 (fraction × 256);
// 13 ≈ 5 % loss. Multiplicative decrease on loss, additive recovery when
// clean. We deliberately do NOT chase REMB: without a sender-side bandwidth
// estimator REMB just tracks the (low) send rate of a static VBR scene and
// spirals the bitrate to the floor even on an idle network.
const LOSS_BACKOFF_THRESHOLD: u8 = 13;
const LOSS_BACKOFF_FACTOR: f64 = 0.85;
const RECOVERY_STEP: f64 = 150_000.0;

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

    // Frame dimensions for the encoder.
    let frame = camera.get_frame();
    let width = frame.width;
    let height = frame.height;
    drop(frame);

    // Encoder runs on its own blocking thread (V4L2 DQBUF blocks). It owns
    // the codec; other threads steer it only via the atomics in
    // EncoderControl. A stop flag lets us tear it down on disconnect.
    let stop = Arc::new(AtomicBool::new(false));
    let (au_tx, mut au_rx) = tokio::sync::mpsc::channel::<crate::video::h264::Encoded>(8);
    let (ctl_tx, ctl_rx) = tokio::sync::oneshot::channel::<EncoderControl>();

    let cam_for_enc = camera.clone();
    let stop_for_enc = stop.clone();
    let encoder_thread = std::thread::Builder::new()
        .name("h264-enc".into())
        .spawn(move || {
            let mut enc = match H264Encoder::new(width, height, FPS, INITIAL_BITRATE) {
                Ok(e) => e,
                Err(e) => {
                    warn!("webrtc: encoder init failed: {e}");
                    return;
                }
            };
            let _ = ctl_tx.send(enc.control());
            while !stop_for_enc.load(Ordering::Relaxed) {
                let frame = cam_for_enc.wait_frame();
                match enc.encode(&frame.data) {
                    Ok(au) => {
                        if au_tx.blocking_send(au).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("webrtc: encode error: {e}");
                        break;
                    }
                }
            }
        })?;

    let control = match ctl_rx.await {
        Ok(c) => c,
        Err(_) => return Err("encoder failed to start".into()),
    };

    // Writer: encoded access units → WebRTC track.
    let track_w = track.clone();
    let writer = tokio::spawn(async move {
        while let Some(au) = au_rx.recv().await {
            let sample = Sample {
                data: Bytes::from(au.data),
                duration: FRAME_DURATION,
                ..Default::default()
            };
            if track_w.write_sample(&sample).await.is_err() {
                break;
            }
        }
    });

    // RTCP feedback. PLI/FIR → immediate keyframe. Receiver-Report loss →
    // AIMD: multiplicative cut on loss, gradual additive recovery when the
    // link is clean. REMB is intentionally ignored (see constant docs).
    let ctl_rtcp = control.clone();
    let rtcp_task = tokio::spawn(async move {
        let mut target = INITIAL_BITRATE as f64;
        loop {
            let pkts = match rtp_sender.read_rtcp().await {
                Ok((p, _)) => p,
                Err(_) => break,
            };
            for p in pkts {
                let any = p.as_any();
                if any.is::<PictureLossIndication>() || any.is::<FullIntraRequest>() {
                    ctl_rtcp.request_keyframe();
                } else if let Some(rr) = any.downcast_ref::<ReceiverReport>() {
                    let worst = rr
                        .reports
                        .iter()
                        .map(|r| r.fraction_lost)
                        .max()
                        .unwrap_or(0);
                    let prev = target;
                    if worst >= LOSS_BACKOFF_THRESHOLD {
                        target = (target * LOSS_BACKOFF_FACTOR).max(MIN_BITRATE as f64);
                        info!("webrtc: loss {worst}/256 → bitrate {} bps", target as u32);
                    } else {
                        target = (target + RECOVERY_STEP).min(MAX_BITRATE as f64);
                    }
                    if (target - prev).abs() >= 1.0 {
                        ctl_rtcp.set_bitrate(target as u32);
                    }
                }
            }
        }
    });

    // Lifecycle: on disconnect, stop the encoder thread and abort tasks.
    let teardown = Arc::new(std::sync::Mutex::new(Some((
        writer,
        rtcp_task,
        encoder_thread,
        stop.clone(),
    ))));
    let teardown_cb = teardown.clone();
    let pc_for_cb = Arc::downgrade(&pc);
    pc.on_peer_connection_state_change(Box::new(move |s| {
        let teardown = teardown_cb.clone();
        let pc_weak = pc_for_cb.clone();
        Box::pin(async move {
            info!("webrtc: pc state = {s}");
            match s {
                RTCPeerConnectionState::Disconnected
                | RTCPeerConnectionState::Failed
                | RTCPeerConnectionState::Closed => {
                    if let Some((writer, rtcp_task, enc_thread, stop)) =
                        teardown.lock().unwrap().take()
                    {
                        stop.store(true, Ordering::Relaxed);
                        writer.abort();
                        rtcp_task.abort();
                        // Encoder thread observes `stop` within one frame
                        // (~33 ms) and exits, running its Drop (STREAMOFF +
                        // munmap + close).
                        tokio::task::spawn_blocking(move || {
                            let _ = enc_thread.join();
                        });
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

    let offer = RTCSessionDescription::offer(offer_sdp)?;
    pc.set_remote_description(offer).await?;

    let answer = pc.create_answer(None).await?;
    let mut gather_complete = pc.gathering_complete_promise().await;
    pc.set_local_description(answer).await?;
    let _ = gather_complete.recv().await;

    let local = pc.local_description().await.ok_or("no local description")?;

    keep_alive(pc);

    Ok(local.sdp)
}

/// Hold a strong reference to the PeerConnection until the session ends.
/// Events drive the work; this future just owns `pc` for its lifetime.
fn keep_alive(pc: Arc<RTCPeerConnection>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            if matches!(
                pc.connection_state(),
                RTCPeerConnectionState::Closed | RTCPeerConnectionState::Failed
            ) {
                break;
            }
        }
        info!("webrtc: session ended");
    });
}
