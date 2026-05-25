use std::sync::{Arc, RwLock};
use std::thread;

use log::{info, warn};
use tokio::sync::mpsc;

use std::path::Path;
use std::time::Duration;

use ginger_rs::{
    api::{Command, SensorSnapshot},
    camera::Camera,
    imu::{self, Imu},
    motion::MotorModel,
    robot::supervisor,
    server::{self, AppState},
    slam::{self, MapSnapshot, SlamSnapshot},
};

/// Path to the persisted motor model (gitignored runtime state, see PLAN.md).
const MOTOR_MODEL_PATH: &str = "motor-model.toml";

/// Save the motor model to disk this often while the binary runs. The
/// label stream (Stage 2) will be hitting `observe` at ~5 Hz once it's
/// wired; this auto-save keeps a reasonably fresh snapshot in case of
/// an ungraceful shutdown.
const MOTOR_MODEL_AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    info!("Ginger starting — built {}", server::BUILD_TIME);

    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(32);
    let sensors = Arc::new(RwLock::new(SensorSnapshot::initial()));

    {
        let sensors = sensors.clone();
        thread::spawn(move || supervisor::run(cmd_rx, sensors));
    }

    println!("Initialising camera…");
    let camera = Arc::new(Camera::new().expect("camera init failed"));
    println!("Camera ready.");

    // Best-effort: a missing/flaky BMI160 must not stop the rest of the
    // robot booting. The SLAM tracking-predict (Stage 4) treats the IMU
    // as an enrichment; when absent it falls back to constant-velocity.
    let imu = match Imu::open(imu::DEFAULT_ADDR) {
        Ok(i) => {
            info!("imu: BMI160 opened at 0x{:02x}", imu::DEFAULT_ADDR);
            Some(Arc::new(i))
        }
        Err(e) => {
            warn!(
                "imu: BMI160 not available at 0x{:02x} ({e}); \
                 /api/imu/sample will return 503",
                imu::DEFAULT_ADDR
            );
            None
        }
    };

    let slam = Arc::new(RwLock::new(SlamSnapshot::initial()));
    let map = Arc::new(RwLock::new(MapSnapshot::initial()));
    {
        let (camera, imu, slam, map) = (camera.clone(), imu.clone(), slam.clone(), map.clone());
        thread::Builder::new()
            .name("slam".into())
            .spawn(move || slam::run(camera, imu, slam, map))
            .expect("spawn slam thread");
    }

    // Motor model: load if non-stale, else bootstrap. The first battery
    // read may not have happened yet — pass a sensible default; the live
    // staleness check kicks in on the *next* boot once we've been running
    // long enough to capture a real reading.
    let battery_v_now = sensors.read().unwrap().battery_v.max(7.8);
    let motor_model = Arc::new(RwLock::new(MotorModel::load_or_bootstrap(
        Path::new(MOTOR_MODEL_PATH),
        battery_v_now,
    )));
    {
        // Periodic auto-save thread.
        let model = motor_model.clone();
        thread::Builder::new()
            .name("motor-model-saver".into())
            .spawn(move || {
                loop {
                    thread::sleep(MOTOR_MODEL_AUTOSAVE_INTERVAL);
                    if let Err(e) = model.read().unwrap().save(Path::new(MOTOR_MODEL_PATH)) {
                        warn!("motor-model: autosave failed: {e}");
                    }
                }
            })
            .expect("spawn motor-model saver");
    }

    server::serve(AppState {
        cmd_tx,
        sensors,
        camera,
        slam,
        map,
        imu,
        motor_model,
    })
    .await;
}
