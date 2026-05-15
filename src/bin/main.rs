use std::sync::{Arc, RwLock, atomic::AtomicBool};
use std::thread;

use log::info;
use tokio::sync::mpsc;

use ginger_rs::{
    api::{Command, SensorSnapshot},
    camera::Camera,
    robot::{map::Map, supervisor},
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
    let map = Arc::new(RwLock::new(Map::new()));
    let explore_stop = Arc::new(AtomicBool::new(false));

    {
        let (sensors, map, explore_stop) = (sensors.clone(), map.clone(), explore_stop.clone());
        thread::spawn(move || supervisor::run(cmd_rx, sensors, map, explore_stop));
    }

    println!("Initialising camera…");
    let camera = Arc::new(Camera::new().expect("camera init failed"));
    println!("Camera ready.");

    server::serve(AppState {
        cmd_tx,
        sensors,
        camera,
        map,
        explore_stop,
    })
    .await;
}
