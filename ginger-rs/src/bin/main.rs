use std::{
    convert::Infallible,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, Instant},
};

use axum::{
    Router,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
    Json,
};
use futures::Stream;
use image::{DynamicImage, ImageFormat, RgbImage};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use ginger_rs::{camera::Camera, car::Car};

// ── Embedded web UI ───────────────────────────────────────────────────────────

const HTML: &str = include_str!("web/index.html");

// ── Shared types ──────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct SensorSnapshot {
    battery_v:   f32,
    light_left:  Option<f32>,
    light_right: Option<f32>,
    ir:          Option<[bool; 3]>,
    us_cm:       Option<f32>,
}

#[derive(Clone, Deserialize, Serialize)]
struct SensorConfig {
    light: bool,
    ir:    bool,
    us:    bool,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self { light: true, ir: true, us: true }
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
}

#[derive(Clone)]
struct AppState {
    cmd_tx:  mpsc::Sender<CarCmd>,
    sensors: Arc<RwLock<SensorSnapshot>>,
    camera:  Arc<Camera>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let (cmd_tx, cmd_rx) = mpsc::channel::<CarCmd>(32);

    let sensors = Arc::new(RwLock::new(SensorSnapshot {
        battery_v: 0.0, light_left: None, light_right: None, ir: None, us_cm: None,
    }));

    let sensors_hw = sensors.clone();
    thread::spawn(move || hardware_thread(cmd_rx, sensors_hw));

    println!("Initialising camera…");
    let camera = Arc::new(Camera::new().expect("camera init failed"));
    println!("Camera ready.");

    let state = AppState { cmd_tx, sensors, camera };

    let app = Router::new()
        .route("/",                   get(serve_html))
        .route("/api/sensors/stream", get(sensor_stream))
        .route("/api/camera/frame",   get(camera_frame))
        .route("/api/drive",          post(drive))
        .route("/api/stop",           post(stop_car))
        .route("/api/pan",            post(pan))
        .route("/api/tilt",           post(tilt))
        .route("/api/led",            post(led))
        .route("/api/led/off",        post(led_off))
        .route("/api/buzzer",         post(buzzer))
        .route("/api/sensors/config", post(sensor_config))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("Listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}

// ── Hardware thread ───────────────────────────────────────────────────────────

fn hardware_thread(mut cmd_rx: mpsc::Receiver<CarCmd>, sensors: Arc<RwLock<SensorSnapshot>>) {
    let mut car    = Car::new().expect("Car init failed");
    let mut config = SensorConfig::default();
    let mut last_drive = Instant::now();
    let mut is_driving = false;

    loop {
        // Poll sensors
        let battery_v = car.battery_v().unwrap_or(0.0);

        let (light_left, light_right) = if config.light {
            car.light().map(|(l, r)| (Some(l), Some(r))).unwrap_or((None, None))
        } else {
            (None, None)
        };

        let ir = if config.ir {
            let (l, c, r) = car.ir.read_all();
            Some([l, c, r])
        } else {
            None
        };

        let us_cm = if config.us { car.us().distance_cm() } else { None };

        *sensors.write().unwrap() = SensorSnapshot { battery_v, light_left, light_right, ir, us_cm };

        // Safety stop: if motors are spinning and no command for 500ms, stop
        if is_driving && last_drive.elapsed() > Duration::from_millis(500) {
            car.stop().ok();
            is_driving = false;
        }

        // Drain command queue
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                CarCmd::SetMotors { left, right } => {
                    car.motors().drive(left, right).ok();
                    last_drive = Instant::now();
                    is_driving = left != 0 || right != 0;
                }
                CarCmd::Stop => {
                    car.stop().ok();
                    is_driving = false;
                }
                CarCmd::SetPan(a)            => { car.pan_tilt().set_pan(a).ok(); }
                CarCmd::SetTilt(a)           => { car.pan_tilt().set_tilt(a).ok(); }
                CarCmd::SetLed { r, g, b }   => { car.leds.set_all(r, g, b); car.leds.show().ok(); }
                CarCmd::LedOff               => { car.leds.clear().ok(); }
                CarCmd::Buzzer(on)           => { if on { car.buzzer.on() } else { car.buzzer.off() } }
                CarCmd::SetSensors(cfg)      => { config = cfg; }
            }
        }

        thread::sleep(Duration::from_millis(80));
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn serve_html() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], HTML)
}

async fn sensor_stream(State(st): State<AppState>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
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
struct DriveBody { left: i32, right: i32 }

async fn drive(State(st): State<AppState>, Json(b): Json<DriveBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetMotors { left: b.left, right: b.right }).await.ok();
    StatusCode::OK
}

async fn stop_car(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(CarCmd::Stop).await.ok();
    StatusCode::OK
}

#[derive(Deserialize)]
struct AngleBody { angle: f32 }

async fn pan(State(st): State<AppState>, Json(b): Json<AngleBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetPan(b.angle)).await.ok();
    StatusCode::OK
}

async fn tilt(State(st): State<AppState>, Json(b): Json<AngleBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetTilt(b.angle)).await.ok();
    StatusCode::OK
}

#[derive(Deserialize)]
struct LedBody { r: u8, g: u8, b: u8 }

async fn led(State(st): State<AppState>, Json(b): Json<LedBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetLed { r: b.r, g: b.g, b: b.b }).await.ok();
    StatusCode::OK
}

async fn led_off(State(st): State<AppState>) -> StatusCode {
    st.cmd_tx.send(CarCmd::LedOff).await.ok();
    StatusCode::OK
}

#[derive(Deserialize)]
struct BuzzerBody { on: bool }

async fn buzzer(State(st): State<AppState>, Json(b): Json<BuzzerBody>) -> StatusCode {
    st.cmd_tx.send(CarCmd::Buzzer(b.on)).await.ok();
    StatusCode::OK
}

async fn sensor_config(State(st): State<AppState>, Json(b): Json<SensorConfig>) -> StatusCode {
    st.cmd_tx.send(CarCmd::SetSensors(b)).await.ok();
    StatusCode::OK
}
