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
    motion::{
        ExploreHandle, LabelStats, MotionTarget, MotorModel, PoseState, explore, labels, pose,
    },
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

/// Poll-sleep until the supervisor has published a non-zero battery
/// reading, capped by [`BATTERY_BOOT_WAIT`]. Falls back to `7.8 V` (the
/// middle of the 2S LiPo's usable range) and logs a warning if the ADC
/// is slow or unreachable — the model still loads, it just anchors on
/// a synthetic value and may re-bootstrap on the next session.
const BATTERY_BOOT_WAIT: Duration = Duration::from_secs(3);
const BATTERY_BOOT_FALLBACK_V: f32 = 7.8;
const BATTERY_BOOT_VALID_THRESHOLD_V: f32 = 0.5;

fn wait_for_battery(sensors: &Arc<RwLock<SensorSnapshot>>) -> f32 {
    let deadline = std::time::Instant::now() + BATTERY_BOOT_WAIT;
    loop {
        let v = sensors.read().unwrap().battery_v;
        if v > BATTERY_BOOT_VALID_THRESHOLD_V {
            return v;
        }
        if std::time::Instant::now() >= deadline {
            warn!(
                "motor-model: no valid battery reading after {:?}; anchoring on {:.1} V — \
                 next boot will likely re-bootstrap",
                BATTERY_BOOT_WAIT, BATTERY_BOOT_FALLBACK_V
            );
            return BATTERY_BOOT_FALLBACK_V;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

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

    // Motor model: load if non-stale, else bootstrap. The supervisor
    // hasn't polled the battery yet at this point, so `SensorSnapshot`
    // still carries the `initial()` zero. Wait briefly for a real
    // reading — anchoring the staleness check on a placeholder defeats
    // the whole point, because the next boot at a different voltage
    // will then trigger an unnecessary re-bootstrap and a saved-model
    // session is effectively never reused.
    let battery_v_now = wait_for_battery(&sensors);
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

    // Stage 2: feed labelled windows to the motor model. Skip if the IMU
    // isn't on the bus — the labeller needs gyro `ω` as its only
    // always-available label source, and the model can stay at its
    // bootstrap weights without it.
    let label_stats: Arc<RwLock<LabelStats>> = Arc::new(RwLock::new(LabelStats::default()));
    if let Some(imu_arc) = imu.as_ref() {
        labels::spawn(
            sensors.clone(),
            imu_arc.clone(),
            motor_model.clone(),
            label_stats.clone(),
        );
    } else {
        warn!("motion-labels: IMU absent — label worker not spawned");
    }

    // Stage 3: pose integrator. Same IMU dependency as the label
    // worker — without gyro we'd integrate trash.
    let motion_target: Arc<RwLock<MotionTarget>> = Arc::new(RwLock::new(MotionTarget::default()));
    let pose_state: Arc<RwLock<PoseState>> = Arc::new(RwLock::new(PoseState::default()));
    if let Some(imu_arc) = imu.as_ref() {
        pose::spawn(
            sensors.clone(),
            imu_arc.clone(),
            motion_target.clone(),
            pose_state.clone(),
        );
    } else {
        warn!("motion-pose: IMU absent — pose integrator not spawned");
    }

    // Stage 4: exploration controller. Always spawned (cheap when off);
    // the WebUI flips `on` via `/api/motion/explore?on=1`.
    let explore_handle = Arc::new(ExploreHandle::new());
    explore::spawn(
        sensors.clone(),
        motor_model.clone(),
        motion_target.clone(),
        pose_state.clone(),
        cmd_tx.clone(),
        &explore_handle,
    );

    server::serve(AppState {
        cmd_tx,
        sensors,
        camera,
        slam,
        map,
        imu,
        motor_model,
        label_stats,
        motion_target,
        pose: pose_state,
        explore: explore_handle,
    })
    .await;
}
