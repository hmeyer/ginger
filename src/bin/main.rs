use std::sync::{Arc, RwLock};
use std::thread;

use log::info;
use tokio::sync::mpsc;

use ginger_rs::{
    api::{Command, SensorSnapshot},
    camera::Camera,
    robot::supervisor,
    server::{self, AppState},
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

    server::serve(AppState {
        cmd_tx,
        sensors,
        camera,
    })
    .await;
}
