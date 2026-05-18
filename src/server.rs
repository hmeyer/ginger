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
    extract::State,
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::Stream;
use log::warn;
use tokio::sync::mpsc;

use crate::{
    api::{AngleBody, Command, DriveBody, SensorConfig, SensorSnapshot},
    camera::Camera,
    slam::SlamSnapshot,
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
}

/// Build the router and serve forever on `0.0.0.0:8080`.
pub async fn serve(state: AppState) {
    let app = Router::new()
        .route("/", get(serve_html))
        .route("/api/sensors/stream", get(sensor_stream))
        .route("/api/slam/stream", get(slam_stream))
        .route("/api/slam/map", get(slam_map))
        .route("/api/webrtc/whep", post(webrtc_whep))
        .route("/api/drive", post(drive))
        .route("/api/stop", post(stop_car))
        .route("/api/pan", post(pan))
        .route("/api/tilt", post(tilt))
        .route("/api/express", post(express))
        .route("/api/sensors/config", post(sensor_config))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
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

/// Top-down map (poses + points) for the WebUI canvas. **M2 stub:**
/// returns an empty map so the transport + canvas exist now; M3's
/// two-view init is the first thing that fills it.
async fn slam_map() -> impl IntoResponse {
    Json(serde_json::json!({ "poses": [], "points": [] }))
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

async fn express(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(Command::Express).await.ok();
    StatusCode::OK
}

async fn sensor_config(State(st): State<AppState>, Json(b): Json<SensorConfig>) -> StatusCode {
    st.cmd_tx.send(Command::SetSensors(b)).await.ok();
    StatusCode::OK
}
