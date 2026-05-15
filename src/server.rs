//! HTTP server: axum router, shared state, and route handlers.
//!
//! Pure transport — every request is translated into a [`Command`] for
//! the supervisor or a read of shared telemetry/map state.

use std::{
    convert::Infallible,
    sync::{Arc, RwLock, atomic::AtomicBool, atomic::Ordering},
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
    api::{AngleBody, BuzzerBody, Command, DriveBody, LedBody, SensorConfig, SensorSnapshot},
    camera::Camera,
    robot::map::Map,
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
    pub map: Arc<RwLock<Map>>,
    pub explore_stop: Arc<AtomicBool>,
}

/// Build the router and serve forever on `0.0.0.0:8080`.
pub async fn serve(state: AppState) {
    let app = Router::new()
        .route("/", get(serve_html))
        .route("/api/sensors/stream", get(sensor_stream))
        .route("/api/webrtc/whep", post(webrtc_whep))
        .route("/api/drive", post(drive))
        .route("/api/stop", post(stop_car))
        .route("/api/pan", post(pan))
        .route("/api/tilt", post(tilt))
        .route("/api/led", post(led))
        .route("/api/led/off", post(led_off))
        .route("/api/buzzer", post(buzzer))
        .route("/api/sensors/config", post(sensor_config))
        .route("/api/scan", post(trigger_scan))
        .route("/api/explore/start", post(explore_start))
        .route("/api/explore/stop", post(explore_stop_handler))
        .route("/api/map", get(map_meta))
        .route("/api/map/png", get(map_png))
        .route("/api/map/ascii", get(map_ascii))
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

async fn led(State(st): State<AppState>, Json(b): Json<LedBody>) -> StatusCode {
    st.cmd_tx
        .send(Command::SetLed {
            r: b.r,
            g: b.g,
            b: b.b,
        })
        .await
        .ok();
    StatusCode::OK
}

async fn led_off(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(Command::LedOff).await.ok();
    StatusCode::OK
}

async fn buzzer(State(st): State<AppState>, Json(b): Json<BuzzerBody>) -> StatusCode {
    st.cmd_tx.send(Command::Buzzer(b.on)).await.ok();
    StatusCode::OK
}

async fn sensor_config(State(st): State<AppState>, Json(b): Json<SensorConfig>) -> StatusCode {
    st.cmd_tx.send(Command::SetSensors(b)).await.ok();
    StatusCode::OK
}

// ── Scan & exploration endpoints ──────────────────────────────────────────────

async fn trigger_scan(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(Command::Scan).await.ok();
    StatusCode::OK
}

async fn explore_start(State(st): State<AppState>) -> StatusCode {
    st.explore_stop.store(false, Ordering::Relaxed);
    st.cmd_tx.send(Command::ExploreStart).await.ok();
    StatusCode::OK
}

async fn explore_stop_handler(State(st): State<AppState>) -> StatusCode {
    st.explore_stop.store(true, Ordering::Relaxed);
    // Also queue Stop so motors are halted immediately after the current tick exits
    st.cmd_tx.send(Command::Stop).await.ok();
    StatusCode::OK
}

// ── Map endpoints ─────────────────────────────────────────────────────────────

async fn map_meta(State(st): State<AppState>) -> impl IntoResponse {
    Json(st.map.read().unwrap().meta())
}

async fn map_png(State(st): State<AppState>) -> Response {
    let png = tokio::task::spawn_blocking(move || st.map.read().unwrap().render_png())
        .await
        .unwrap();
    (
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        png,
    )
        .into_response()
}

async fn map_ascii(State(st): State<AppState>) -> impl IntoResponse {
    let ascii = st.map.read().unwrap().to_ascii();
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], ascii)
}
