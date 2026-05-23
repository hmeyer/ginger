//! HTTP server: axum router, shared state, and route handlers.
//!
//! Pure transport — every request is translated into a [`Command`] for
//! the supervisor or a read of shared telemetry.

use std::{
    convert::Infallible,
    sync::{Arc, RwLock},
    time::Duration,
};

use axum::{
    Json, Router,
    body::Body,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::Stream;
use log::warn;
use serde::Deserialize;
use tokio::sync::mpsc;

use crate::{
    api::{AngleBody, Command, DriveBody, SensorConfig, SensorSnapshot},
    camera::Camera,
    slam::{MapSnapshot, SlamSnapshot},
    video::webrtc,
};

const HTML_TEMPLATE: &str = include_str!("bin/web/index.html");
pub const BUILD_TIME: &str = env!("BUILD_TIME");

/// Shared application state handed to every request handler.
#[derive(Clone)]
pub struct AppState {
    pub cmd_tx: mpsc::Sender<Command>,
    pub sensors: Arc<RwLock<SensorSnapshot>>,
    pub camera: Arc<Camera>,
    pub slam: Arc<RwLock<SlamSnapshot>>,
    pub map: Arc<RwLock<MapSnapshot>>,
}

/// Build the axum router (all routes + shared state). Split out from
/// [`serve`] so the handlers can be exercised in-process by tests
/// (`tower`'s `oneshot`) with no socket or camera hardware.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(serve_html))
        .route("/api/sensors/stream", get(sensor_stream))
        .route("/api/slam/stream", get(slam_stream))
        .route("/api/slam/map", get(slam_map))
        .route("/api/camera/frame", get(camera_frame))
        .route("/api/webrtc/whep", post(webrtc_whep))
        .route("/api/drive", post(drive))
        .route("/api/stop", post(stop_car))
        .route("/api/pan", post(pan))
        .route("/api/tilt", post(tilt))
        .route("/api/emote", post(emote))
        .route("/api/sensors/config", post(sensor_config))
        .with_state(state)
}

/// Build the router and serve forever on `0.0.0.0:8080`.
pub async fn serve(state: AppState) {
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Listening on http://0.0.0.0:8080");
    axum::serve(listener, router(state)).await.unwrap();
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn serve_html() -> impl IntoResponse {
    let html = HTML_TEMPLATE.replace("{{BUILD_TIME}}", BUILD_TIME);
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], html)
}

async fn sensor_stream(
    State(st): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(Duration::from_millis(200));
        loop {
            interval.tick().await;
            let mut snap = st.sensors.read().unwrap().clone();
            snap.camera_fps = st.camera.fps();
            {
                let exp = st.camera.exposure_cfg.lock().unwrap();
                snap.exposure_us = exp.current_exposure_us;
                snap.gain = exp.current_gain;
                snap.brightness = exp.current_brightness;
                snap.luma = exp.current_luma;
            }
            let json = serde_json::to_string(&snap).unwrap();
            yield Ok::<Event, Infallible>(Event::default().data(json));
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// SLAM frontend stream: latest detected features for the live overlay.
/// ~15 Hz, decoupled from the 5 Hz telemetry stream so neither stalls
/// the other. Only emits when a new frame has been processed.
async fn slam_stream(
    State(st): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(Duration::from_millis(66));
        let mut last_sent = u32::MAX;
        loop {
            interval.tick().await;
            let snap = st.slam.read().unwrap().clone();
            // Skip resends when the frontend hasn't produced a new frame.
            let stamp = snap.n_total ^ ((snap.detect_ms * 100.0) as u32);
            if stamp == last_sent {
                continue;
            }
            last_sent = stamp;
            let json = serde_json::to_string(&snap).unwrap();
            yield Ok::<Event, Infallible>(Event::default().data(json));
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Top-down map for the WebUI canvas: keyframe poses + map points (or
/// an init-status string while two-view bootstrap parallax accumulates).
async fn slam_map(State(st): State<AppState>) -> impl IntoResponse {
    Json(st.map.read().unwrap().clone())
}

// ── Camera frame still ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct CameraFrameParams {
    /// Output width in pixels (height follows source aspect ratio).
    #[serde(default = "default_frame_w")]
    w: u32,
    /// JPEG quality 1..=100.
    #[serde(default = "default_frame_q")]
    q: u8,
}
fn default_frame_w() -> u32 {
    320
}
fn default_frame_q() -> u8 {
    70
}

/// Single grayscale JPEG of the most recent camera frame. Sized for
/// debug/diagnostic use (a few KB at default 320 px) — answers "is the
/// path clear?" without negotiating WebRTC. Pulls the Y plane straight
/// out of YUYV with nearest-neighbor downscale; no extra camera load.
async fn camera_frame(State(st): State<AppState>, Query(p): Query<CameraFrameParams>) -> Response {
    let Some(frame) = st.camera.try_frame() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "camera frame not ready",
        )
            .into_response();
    };
    let w_out = p.w.clamp(16, frame.width);
    let h_out = ((w_out as u64 * frame.height as u64) / frame.width as u64).max(1) as u32;
    let q = p.q.clamp(10, 100);

    // Y plane lives at even bytes within the YUYV row (Y0 U Y1 V repeats).
    let stride = (frame.width as usize) * 2;
    let mut gray = vec![0u8; (w_out as usize) * (h_out as usize)];
    for y in 0..h_out {
        let y_src = ((y as u64 * frame.height as u64) / h_out as u64) as usize;
        let row_in = y_src * stride;
        let row_out = (y as usize) * (w_out as usize);
        for x in 0..w_out {
            let x_src = ((x as u64 * frame.width as u64) / w_out as u64) as usize;
            gray[row_out + x as usize] = frame.data[row_in + x_src * 2];
        }
    }

    let mut buf = Vec::with_capacity(gray.len() / 2);
    if let Err(e) = jpeg_encoder::Encoder::new(&mut buf, q).encode(
        &gray,
        w_out as u16,
        h_out as u16,
        jpeg_encoder::ColorType::Luma,
    ) {
        warn!("camera_frame: jpeg encode failed: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("jpeg encode: {e}"),
        )
            .into_response();
    }

    ([(header::CONTENT_TYPE, "image/jpeg")], buf).into_response()
}

// ── WebRTC signalling ─────────────────────────────────────────────────────────

async fn webrtc_whep(State(st): State<AppState>, body: String) -> Response {
    match webrtc::whep_handle(st.camera.clone(), body).await {
        Ok(answer_sdp) => Response::builder()
            .status(StatusCode::CREATED)
            .header(header::CONTENT_TYPE, "application/sdp")
            .header(header::LOCATION, "/api/webrtc/whep")
            .body(Body::from(answer_sdp))
            .unwrap(),
        Err(e) => {
            warn!("webrtc: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response()
        }
    }
}

// ── Control endpoints ─────────────────────────────────────────────────────────

async fn drive(State(st): State<AppState>, Json(b): Json<DriveBody>) -> StatusCode {
    st.cmd_tx
        .send(Command::SetMotors {
            left: b.left,
            right: b.right,
        })
        .await
        .ok();
    StatusCode::OK
}

async fn stop_car(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(Command::Stop).await.ok();
    StatusCode::OK
}

async fn pan(State(st): State<AppState>, Json(b): Json<AngleBody>) -> StatusCode {
    st.cmd_tx.send(Command::SetPan(b.angle)).await.ok();
    StatusCode::OK
}

async fn tilt(State(st): State<AppState>, Json(b): Json<AngleBody>) -> StatusCode {
    st.cmd_tx.send(Command::SetTilt(b.angle)).await.ok();
    StatusCode::OK
}

async fn emote(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(Command::Emote).await.ok();
    StatusCode::OK
}

async fn sensor_config(State(st): State<AppState>, Json(b): Json<SensorConfig>) -> StatusCode {
    st.cmd_tx.send(Command::SetSensors(b)).await.ok();
    StatusCode::OK
}

/// In-process HTTP tests: drive the real router + handlers via `tower`'s
/// `oneshot` — no socket, no camera hardware. Gated to the mock-camera
/// build (`libcamera` off) so `AppState` is constructible headless;
/// that is the configuration `cargo test --no-default-features` uses.
#[cfg(all(test, not(feature = "libcamera")))]
mod tests {
    use super::*;
    use axum::http::Request;
    use tower::ServiceExt; // `oneshot`

    use crate::camera::Camera;
    use crate::slam::{MapSnapshot, SlamSnapshot};

    /// Exercise the read + control endpoints end-to-end: real router,
    /// real handlers, real JSON serialization.
    #[tokio::test]
    async fn api_endpoints_respond() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<Command>(32);
        let state = AppState {
            cmd_tx,
            sensors: Arc::new(RwLock::new(SensorSnapshot::initial())),
            camera: Arc::new(Camera::new().expect("mock camera")),
            slam: Arc::new(RwLock::new(SlamSnapshot::initial())),
            map: Arc::new(RwLock::new(MapSnapshot::initial())),
        };
        // Distinctive values so the assertions check real serialization,
        // not just the `initial()` defaults.
        *state.map.write().unwrap() = MapSnapshot {
            status: "tracking: 30/40 inliers".into(),
            n_keyframes: 9,
            loop_closures: 3,
            bow_ready: true,
            bow_words: 512,
            ..MapSnapshot::initial()
        };
        let app = router(state);

        // GET /api/slam/map — the snapshot the WebUI #map-stat HUD reads,
        // including the debug-HUD fields added for it.
        let resp = app
            .clone()
            .oneshot(Request::get("/api/slam/map").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "tracking: 30/40 inliers");
        assert_eq!(v["n_keyframes"], 9);
        assert_eq!(v["loop_closures"], 3);
        assert_eq!(v["bow_ready"], true);
        assert_eq!(v["bow_words"], 512);

        // POST /api/stop — 200, and a Stop command reached the supervisor.
        let resp = app
            .clone()
            .oneshot(Request::post("/api/stop").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(matches!(cmd_rx.try_recv(), Ok(Command::Stop)));

        // GET / — the WebUI document is served.
        let resp = app
            .clone()
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // GET /api/camera/frame — returns a tiny JPEG of the mock frame.
        // We assert content-type + SOI/EOI markers; not bit-exact bytes
        // because the encoder output can shift across versions, but the
        // JPEG container framing is stable.
        let resp = app
            .oneshot(
                Request::get("/api/camera/frame?w=64&q=70")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/jpeg"),
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(body.len() > 4, "jpeg body too short: {} bytes", body.len());
        assert_eq!(&body[..2], &[0xff, 0xd8], "missing JPEG SOI marker");
        assert_eq!(
            &body[body.len() - 2..],
            &[0xff, 0xd9],
            "missing JPEG EOI marker"
        );
    }
}
