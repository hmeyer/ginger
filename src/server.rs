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
    api::{AngleBody, Command, DriveBody, ImuSampleView, SensorConfig, SensorSnapshot},
    camera::Camera,
    imu::Imu,
    motion::{ExploreHandle, ModelInput, MotionTarget, MotorModel, PoseState, arcade_drive},
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
    /// `None` when the BNO055 wasn't reachable at boot (no hardware,
    /// flaky bus, headless dev / test). Endpoints that depend on it
    /// answer 503 in that case rather than panicking.
    pub imu: Option<Arc<Imu>>,
    /// Forward motor model — predicts (Δs, Δθ) over a 200 ms window
    /// from PWMs + chassis state. Trained continuously by the
    /// `motion::labels` worker. Read out at `/api/motion/model` and
    /// `/api/motion/model/predict`; **not** on the drive path (which
    /// uses `arcade_drive`).
    pub motor_model: Arc<RwLock<MotorModel>>,
    /// Telemetry from the Stage-2 label worker
    /// (`src/motion/labels.rs`). Counters + rejection breakdown surfaced
    /// at `/api/motion/labels`. Zeroed at boot, updated in-place by the
    /// worker. Always present even when the IMU isn't (in which case
    /// the counters never advance — the operator sees the silence).
    pub label_stats: Arc<RwLock<crate::motion::LabelStats>>,
    /// Stage 3: latest desired motion + the PWM the model predicted
    /// from it. Written by `/api/motion/drive`, read by the pose
    /// integrator and the WebUI residual display.
    pub motion_target: Arc<RwLock<MotionTarget>>,
    /// Stage 3: chassis pose `(x, y, θ)` + trail. Updated at 20 Hz by
    /// the `motion-pose` thread; read by `/api/motion/pose` and the
    /// WebUI Pose card.
    pub pose: Arc<RwLock<PoseState>>,
    /// Stage 4: exploration controller handle. The worker thread polls
    /// `handle.on()` and drives autonomously when set; the WebUI flips
    /// it via `POST /api/motion/explore?on=1|0`.
    pub explore: Arc<ExploreHandle>,
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
        .route("/api/imu/sample", get(imu_sample))
        .route("/api/imu/calib", get(imu_calib))
        .route("/api/motion/model", get(motion_model_info))
        .route("/api/motion/model/predict", get(motion_model_predict))
        .route("/api/motion/model/reset", post(motion_model_reset))
        .route("/api/motion/labels", get(motion_labels))
        .route("/api/motion/drive", post(motion_drive))
        .route("/api/motion/pose", get(motion_pose))
        .route("/api/motion/reset", post(motion_pose_reset))
        .route("/api/motion/explore", get(motion_explore_status))
        .route("/api/motion/explore", post(motion_explore_toggle))
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
            // IMU read-out + sync canary. Compute camera_age and
            // sample_age from the *same* `now` snapshot so the WebUI's
            // sync-gap number can't pick up jitter from two separate
            // Instant::now() calls. `latest()` already gates on the
            // chip's fusion warm-up — None during the first few seconds
            // after boot is the chip auto-zeroing its gyro.
            if let Some(imu) = st.imu.as_ref() {
                snap.imu_rate_hz = Some(imu.rate_hz());
                snap.imu_calib = Some(imu.calib_status());
                if let Some(s) = imu.latest() {
                    let (_, _, yaw) = s.orientation.euler_angles();
                    snap.imu_yaw_deg = Some(yaw.to_degrees());
                    snap.imu_linear_accel_mps2 = Some(s.linear_accel);
                    let now = std::time::Instant::now();
                    let sample_age = now.duration_since(s.t_read).as_secs_f32() * 1000.0;
                    if let Some(f) = st.camera.try_frame() {
                        let frame_age = now.duration_since(f.t_capture).as_secs_f32() * 1000.0;
                        snap.imu_frame_sync_ms = Some(frame_age - sample_age);
                    }
                }
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

// ── IMU sample (debug / sync-verification) ────────────────────────────────────

/// Latest BNO055 fusion snapshot + the latest camera frame's capture
/// time, both reported as "ago in ms" so the caller can read the
/// host-monotonic gap directly. Returns 503 if the IMU wasn't
/// initialized at boot (no chip on the bus) or if the chip's fusion
/// warm-up hasn't completed (`calib.gyr == 0` — `Imu::latest` returns
/// `None` until then; first few seconds after power-up).
async fn imu_sample(State(st): State<AppState>) -> Response {
    let Some(imu) = st.imu.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "imu not initialised",
        )
            .into_response();
    };
    let Some(s) = imu.latest() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "imu fusion warming up",
        )
            .into_response();
    };

    let now = std::time::Instant::now();
    let t_sample_ago_ms = now.duration_since(s.t_read).as_secs_f32() * 1000.0;
    let frame_t = st.camera.try_frame().map(|f| f.t_capture);
    let t_frame_capture_ago_ms = frame_t.map(|t| now.duration_since(t).as_secs_f32() * 1000.0);
    let frame_to_sample_ms = t_frame_capture_ago_ms.map(|f| f - t_sample_ago_ms);

    let q = s.orientation.into_inner();
    let (roll, pitch, yaw) = s.orientation.euler_angles();

    Json(ImuSampleView {
        orientation_quat: [q.w, q.i, q.j, q.k],
        yaw_deg: yaw.to_degrees(),
        pitch_deg: pitch.to_degrees(),
        roll_deg: roll.to_degrees(),
        linear_accel_mps2: s.linear_accel,
        calib: imu.calib_status(),
        rate_hz: imu.rate_hz(),
        sample_index: s.sample_index,
        t_sample_ago_ms,
        t_frame_capture_ago_ms,
        frame_to_sample_ms,
    })
    .into_response()
}

/// Per-subsystem fusion calibration status — drives the WebUI's
/// "fusion ready" badge. Always responds 200 when the IMU is present
/// (even during warm-up, since the calib byte itself is the warm-up
/// indicator); 503 only when the chip wasn't initialized at boot.
async fn imu_calib(State(st): State<AppState>) -> Response {
    let Some(imu) = st.imu.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            "imu not initialised",
        )
            .into_response();
    };
    Json(imu.calib_status()).into_response()
}

// ── Motor-model endpoints ─────────────────────────────────────────────────────
//
// The forward model predicts `(Δs, Δθ)` over a 200 ms window from PWMs
// + chassis state. It is **not on the drive path** — `/api/motion/drive`
// uses the pure-math `arcade_drive` mapping. The model is read out for
// diagnostics, planning, and the WebUI probe.

/// Health + telemetry for the `Motor model` WebUI card.
async fn motion_model_info(State(st): State<AppState>) -> Response {
    let m = st.motor_model.read().unwrap();
    Json(serde_json::json!({
        "trained_steps": m.trained_steps(),
        "last_battery_v": m.last_battery_v(),
        "last_updated_unix": m.last_updated_unix(),
        "residual_motion": m.residual_motion(),
    }))
    .into_response()
}

/// Forward pass: PWMs + state → predicted `(Δs_m, Δθ_rad)` over the
/// next 200 ms label window. All state inputs default to neutral
/// (zero history, zero previous motion, 7.8 V battery); `pwm_l` and
/// `pwm_r` are required.
async fn motion_model_predict(
    State(st): State<AppState>,
    Query(q): Query<MotionPredictQuery>,
) -> Response {
    let input = ModelInput {
        pwm_l: q.pwm_l,
        pwm_r: q.pwm_r,
        pwm_l_prev: q.pwm_l_prev.unwrap_or(0),
        pwm_r_prev: q.pwm_r_prev.unwrap_or(0),
        v_prev: q.v_prev.unwrap_or(0.0),
        omega_prev: q.omega_prev.unwrap_or(0.0),
        battery_v: q.battery_v.unwrap_or(7.8),
    };
    let pred = st.motor_model.read().unwrap().predict(input);
    Json(serde_json::json!({
        "ds_m": pred.ds_m,
        "dtheta_rad": pred.dtheta_rad,
    }))
    .into_response()
}

/// Re-bootstrap the forward model. Wired to a WebUI button.
async fn motion_model_reset(State(st): State<AppState>) -> Response {
    let battery_v = st.sensors.read().unwrap().battery_v;
    let fresh = MotorModel::default_bootstrap(battery_v);
    *st.motor_model.write().unwrap() = fresh;
    StatusCode::OK.into_response()
}

/// Label-worker telemetry. Counters of observed / Δs-labelled
/// windows and rejection breakdown — backs the WebUI "Labels" card.
async fn motion_labels(State(st): State<AppState>) -> Response {
    let stats = *st.label_stats.read().unwrap();
    Json(stats).into_response()
}

/// Drive in motion units. POST body `{ v_target, omega_target }`
/// (m/s, rad/s). The mapping is **pure math** (arcade-drive): no
/// learned model in the drive path, robust by construction. The
/// commanded intent + resulting PWMs are stored in `motion_target` so
/// the pose integrator and label worker can read them.
///
/// `POST /api/drive` (raw PWM) remains for diagnostics and the curl
/// recipes in `CLAUDE.md`.
async fn motion_drive(State(st): State<AppState>, Json(b): Json<MotionDriveBody>) -> StatusCode {
    let (pwm_l, pwm_r) = arcade_drive(b.v_target, b.omega_target);
    *st.motion_target.write().unwrap() = MotionTarget {
        v_target: b.v_target,
        omega_target: b.omega_target,
        pwm_l,
        pwm_r,
    };
    st.cmd_tx
        .send(Command::SetMotors {
            left: pwm_l,
            right: pwm_r,
        })
        .await
        .ok();
    StatusCode::OK
}

/// Stage 3: chassis pose + recent trail. The trail is the deque kept
/// inside [`PoseState`]; the JSON serialises it as an array.
async fn motion_pose(State(st): State<AppState>) -> Response {
    let p = st.pose.read().unwrap().clone();
    Json(p).into_response()
}

/// Stage 3: zero the pose integrator (back to `(0, 0, 0)`, empty trail).
async fn motion_pose_reset(State(st): State<AppState>) -> StatusCode {
    st.pose.write().unwrap().reset();
    StatusCode::OK
}

/// Stage 4: latest exploration state — current phase, last polar scan,
/// scan counter. Powers the WebUI "Explore" card.
async fn motion_explore_status(State(st): State<AppState>) -> Response {
    Json(st.explore.status()).into_response()
}

/// Stage 4: toggle the autonomous controller. `?on=1` starts it,
/// `?on=0` stops. Always returns 200; the operator can spam this
/// without ill effect.
async fn motion_explore_toggle(
    State(st): State<AppState>,
    Query(q): Query<ExploreToggle>,
) -> StatusCode {
    st.explore.set_on(q.on != 0);
    StatusCode::OK
}

#[derive(Deserialize)]
struct ExploreToggle {
    #[serde(default)]
    on: u32,
}

#[derive(Deserialize)]
struct MotionDriveBody {
    v_target: f32,
    omega_target: f32,
}

#[derive(Deserialize)]
struct MotionPredictQuery {
    pwm_l: i32,
    pwm_r: i32,
    pwm_l_prev: Option<i32>,
    pwm_r_prev: Option<i32>,
    v_prev: Option<f32>,
    omega_prev: Option<f32>,
    battery_v: Option<f32>,
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
    use crate::motion::{ExploreHandle, LabelStats, MotionTarget, MotorModel, PoseState};
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
            // Headless: no chip on the bus. /api/imu/sample returns 503,
            // which the next test asserts on.
            imu: None,
            motor_model: Arc::new(RwLock::new(MotorModel::default_bootstrap(7.8))),
            label_stats: Arc::new(RwLock::new(LabelStats::default())),
            motion_target: Arc::new(RwLock::new(MotionTarget::default())),
            pose: Arc::new(RwLock::new(PoseState::default())),
            explore: Arc::new(ExploreHandle::new()),
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
            .clone()
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

        // GET /api/imu/sample — IMU absent in this headless config, so
        // the handler must answer 503 rather than panic. Verifies the
        // Option<Imu> fallback path; the populated path is exercised by
        // `crate::imu::tests` and a manual on-Pi curl per PLAN.md.
        let resp = app
            .clone()
            .oneshot(Request::get("/api/imu/sample").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        // GET /api/motion/model — health endpoint, always responds 200
        // once the bootstrap has run.
        let resp = app
            .clone()
            .oneshot(
                Request::get("/api/motion/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // The bootstrap ran 2000 SGD steps; the endpoint should reflect that.
        assert!(v["trained_steps"].as_u64().unwrap() > 0);
        // Battery anchored to the value we passed at construction.
        assert!((v["last_battery_v"].as_f64().unwrap() - 7.8).abs() < 1e-3);

        // GET /api/motion/model/predict — forward direction: PWMs in,
        // (Δs, Δθ) out. PWM 1500 forward symmetric should predict
        // positive Δs and near-zero Δθ after bootstrap.
        let resp = app
            .oneshot(
                Request::get("/api/motion/model/predict?pwm_l=1500&pwm_r=1500")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ds = v["ds_m"].as_f64().unwrap();
        let dtheta = v["dtheta_rad"].as_f64().unwrap();
        assert!(
            ds > 0.0,
            "forward PWMs should predict positive Δs, got {ds}"
        );
        assert!(
            dtheta.abs() < 0.05,
            "forward PWMs should predict near-zero Δθ, got {dtheta}"
        );
    }
}
