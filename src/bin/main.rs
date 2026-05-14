use std::{
    convert::Infallible,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::State,
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::Stream;
use image::{DynamicImage, ImageFormat, RgbImage};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use log::{info, warn};

use ginger_rs::{camera::Camera, car::Car, explore, map::Map};

// ── Embedded web UI ───────────────────────────────────────────────────────────

const HTML_TEMPLATE: &str = include_str!("web/index.html");
const BUILD_TIME: &str = env!("BUILD_TIME");

// ── Shared types ──────────────────────────────────────────────────────────────

// 2S LiPo: 8.4 V full, 6.0 V cutoff. Log data to refine these constants.
const BAT_FULL_V: f32 = 8.4;
const BAT_EMPTY_V: f32 = 6.0;

fn battery_pct(v: f32) -> u8 {
    ((v - BAT_EMPTY_V) / (BAT_FULL_V - BAT_EMPTY_V) * 100.0).clamp(0.0, 100.0) as u8
}

#[derive(Clone, Serialize)]
struct SensorSnapshot {
    battery_v: f32,
    battery_pct: u8,
    light_left: Option<f32>,
    light_right: Option<f32>,
    ir: Option<[bool; 3]>,
    us_cm: Option<f32>,
    explore_state: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct SensorConfig {
    light: bool,
    ir: bool,
    us: bool,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            light: true,
            ir: true,
            us: true,
        }
    }
}

enum CarCmd {
    SetMotors { left: i32, right: i32 },
    Stop,
    SetPan(f32),
    SetTilt(f32),
    SetLed { r: u8, g: u8, b: u8 },
    LedOff,
    Buzzer(bool),
    SetSensors(SensorConfig),
    Scan,
    ExploreStart,
}

#[derive(Clone)]
struct AppState {
    cmd_tx: mpsc::Sender<CarCmd>,
    sensors: Arc<RwLock<SensorSnapshot>>,
    camera: Arc<Camera>,
    map: Arc<RwLock<Map>>,
    explore_stop: Arc<AtomicBool>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    info!("Ginger starting — built {BUILD_TIME}");

    let (cmd_tx, cmd_rx) = mpsc::channel::<CarCmd>(32);

    let sensors = Arc::new(RwLock::new(SensorSnapshot {
        battery_v: 0.0,
        battery_pct: 0,
        light_left: None,
        light_right: None,
        ir: None,
        us_cm: None,
        explore_state: "idle".into(),
    }));

    let map = Arc::new(RwLock::new(Map::new()));
    let explore_stop = Arc::new(AtomicBool::new(false));

    let sensors_hw = sensors.clone();
    let map_hw = map.clone();
    let explore_stop_hw = explore_stop.clone();
    thread::spawn(move || hardware_thread(cmd_rx, sensors_hw, map_hw, explore_stop_hw));

    println!("Initialising camera…");
    let camera = Arc::new(Camera::new().expect("camera init failed"));
    println!("Camera ready.");

    let state = AppState {
        cmd_tx,
        sensors,
        camera,
        map,
        explore_stop,
    };

    let app = Router::new()
        .route("/", get(serve_html))
        .route("/api/sensors/stream", get(sensor_stream))
        .route("/api/camera/frame", get(camera_frame))
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

// ── Hardware thread ───────────────────────────────────────────────────────────

fn hardware_thread(
    mut cmd_rx: mpsc::Receiver<CarCmd>,
    sensors: Arc<RwLock<SensorSnapshot>>,
    map: Arc<RwLock<Map>>,
    explore_stop: Arc<AtomicBool>,
) {
    let mut car = Car::new().expect("Car init failed");
    let mut config = SensorConfig::default();
    let mut last_drive = Instant::now();
    let mut is_driving = false;
    let mut explore_active = false;

    loop {
        // ── Command queue ──────────────────────────────────────────────────────
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                CarCmd::ExploreStart => {
                    info!("hw: exploration started");
                    explore_stop.store(false, Ordering::Relaxed);
                    explore_active = true;
                    is_driving = false;
                }
                CarCmd::Stop => {
                    info!("hw: stop command — cancelling exploration");
                    explore_active = false;
                    explore_stop.store(true, Ordering::Relaxed);
                    if let Err(e) = car.stop() {
                        warn!("hw: stop error: {e}");
                    }
                    is_driving = false;
                }
                CarCmd::SetMotors { left, right } => {
                    if explore_active {
                        info!("hw: manual drive — cancelling exploration");
                    }
                    explore_active = false;
                    explore_stop.store(true, Ordering::Relaxed);
                    if let Err(e) = car.motors().drive(left, right) {
                        warn!("hw: drive({left},{right}) error: {e}");
                    }
                    last_drive = Instant::now();
                    is_driving = left != 0 || right != 0;
                }
                CarCmd::SetPan(a) => {
                    car.pan_tilt().set_pan(a).ok();
                }
                CarCmd::SetTilt(a) => {
                    car.pan_tilt().set_tilt(a).ok();
                }
                CarCmd::SetLed { r, g, b } => {
                    car.leds.set_all(r, g, b);
                    car.leds.show().ok();
                }
                CarCmd::LedOff => {
                    car.leds.clear().ok();
                }
                CarCmd::Buzzer(on) => {
                    if on {
                        car.buzzer.on()
                    } else {
                        car.buzzer.off()
                    }
                }
                CarCmd::SetSensors(cfg) => {
                    config = cfg;
                }
                CarCmd::Scan => {
                    let noop = AtomicBool::new(false);
                    let rays = explore::do_scan(&mut car, &noop);
                    map.write().unwrap().integrate_scan(&rays);
                }
            }
        }

        // ── Exploration loop ───────────────────────────────────────────────────
        if explore_active {
            sensors.write().unwrap().explore_state = "scanning".into();
            let status = explore::tick(&mut car, &map, &explore_stop);
            info!("explore: tick → {status}");
            {
                let mut snap = sensors.write().unwrap();
                snap.explore_state = status.to_string();
                let v = car.battery_v().unwrap_or(snap.battery_v);
                snap.battery_v = v;
                snap.battery_pct = battery_pct(v);
            }
            if status == explore::Status::Complete || explore_stop.load(Ordering::Relaxed) {
                info!("explore: stopped (status={status})");
                explore_active = false;
                explore_stop.store(false, Ordering::Relaxed);
            }
            continue; // skip normal sensor poll + sleep
        }

        // ── Normal sensor poll ────────────────────────────────────────────────
        let battery_v = car.battery_v().unwrap_or(0.0);
        let (light_left, light_right) = if config.light {
            car.light()
                .map(|(l, r)| (Some(l), Some(r)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };
        let ir = if config.ir {
            let (l, c, r) = car.ir.read_all();
            Some([l, c, r])
        } else {
            None
        };
        let us_cm = if config.us {
            car.us().distance_cm()
        } else {
            None
        };

        let battery_pct = battery_pct(battery_v);
        info!("bat: {battery_v:.3} V  {battery_pct}%");
        *sensors.write().unwrap() = SensorSnapshot {
            battery_v,
            battery_pct,
            light_left,
            light_right,
            ir,
            us_cm,
            explore_state: "idle".into(),
        };

        // Safety stop if motors have been spinning with no command for 500 ms
        if is_driving && last_drive.elapsed() > Duration::from_millis(500) {
            car.stop().ok();
            is_driving = false;
        }

        thread::sleep(Duration::from_millis(80));
    }
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
            let snap = st.sensors.read().unwrap().clone();
            let json = serde_json::to_string(&snap).unwrap();
            yield Ok::<Event, Infallible>(Event::default().data(json));
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn camera_frame(State(st): State<AppState>) -> Response {
    let frame = st.camera.get_frame();
    let jpeg = tokio::task::spawn_blocking(move || -> Vec<u8> {
        let rgb = frame.to_rgb();
        let img: DynamicImage = RgbImage::from_raw(frame.width, frame.height, rgb)
            .unwrap()
            .into();
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Jpeg).unwrap();
        buf.into_inner()
    })
    .await
    .unwrap();

    (
        [
            (header::CONTENT_TYPE, "image/jpeg"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        jpeg,
    )
        .into_response()
}

// ── Control endpoints ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct DriveBody {
    left: i32,
    right: i32,
}

async fn drive(State(st): State<AppState>, Json(b): Json<DriveBody>) -> StatusCode {
    st.cmd_tx
        .send(CarCmd::SetMotors {
            left: b.left,
            right: b.right,
        })
        .await
        .ok();
    StatusCode::OK
}

async fn stop_car(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(CarCmd::Stop).await.ok();
    StatusCode::OK
}

#[derive(Deserialize)]
struct AngleBody {
    angle: f32,
}

async fn pan(State(st): State<AppState>, Json(b): Json<AngleBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetPan(b.angle)).await.ok();
    StatusCode::OK
}

async fn tilt(State(st): State<AppState>, Json(b): Json<AngleBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetTilt(b.angle)).await.ok();
    StatusCode::OK
}

#[derive(Deserialize)]
struct LedBody {
    r: u8,
    g: u8,
    b: u8,
}

async fn led(State(st): State<AppState>, Json(b): Json<LedBody>) -> StatusCode {
    st.cmd_tx
        .send(CarCmd::SetLed {
            r: b.r,
            g: b.g,
            b: b.b,
        })
        .await
        .ok();
    StatusCode::OK
}

async fn led_off(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(CarCmd::LedOff).await.ok();
    StatusCode::OK
}

#[derive(Deserialize)]
struct BuzzerBody {
    on: bool,
}

async fn buzzer(State(st): State<AppState>, Json(b): Json<BuzzerBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::Buzzer(b.on)).await.ok();
    StatusCode::OK
}

async fn sensor_config(State(st): State<AppState>, Json(b): Json<SensorConfig>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetSensors(b)).await.ok();
    StatusCode::OK
}

// ── Scan & exploration endpoints ──────────────────────────────────────────────

async fn trigger_scan(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(CarCmd::Scan).await.ok();
    StatusCode::OK
}

async fn explore_start(State(st): State<AppState>) -> StatusCode {
    st.explore_stop.store(false, Ordering::Relaxed);
    st.cmd_tx.send(CarCmd::ExploreStart).await.ok();
    StatusCode::OK
}

async fn explore_stop_handler(State(st): State<AppState>) -> StatusCode {
    st.explore_stop.store(true, Ordering::Relaxed);
    // Also queue Stop so motors are halted immediately after the current tick exits
    st.cmd_tx.send(CarCmd::Stop).await.ok();
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
