use std::sync::{Arc, RwLock};
use std::thread;

use log::{info, warn};
use tokio::sync::mpsc;

use ginger_rs::{
    api::{Command, SensorSnapshot},
    camera::Camera,
    imu::{self, Imu},
    robot::supervisor,
    server::{self, AppState},
    slam::{self, MapSnapshot, SlamSnapshot},
};

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

    let slam = Arc::new(RwLock::new(SlamSnapshot::initial()));
    let map = Arc::new(RwLock::new(MapSnapshot::initial()));
    {
        let (camera, slam, map) = (camera.clone(), slam.clone(), map.clone());
        thread::Builder::new()
            .name("slam".into())
            .spawn(move || slam::run(camera, slam, map))
            .expect("spawn slam thread");
    }

    // Best-effort: a missing/flaky BMI160 must not stop the rest of the
    // robot booting. The SLAM tracking-predict (Stage 4) treats the IMU
    // as an enrichment; Stage 2 only ships /api/imu/sample for manual
    // verification.
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

    server::serve(AppState {
        cmd_tx,
        sensors,
        camera,
        slam,
        map,
        imu,
    })
    .await;
}
