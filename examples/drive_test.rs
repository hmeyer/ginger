//! Basic smoke-test: battery check → forward 0.5 s → stop.

use std::time::Duration;

use ginger_rs::car::Car;

fn main() -> ginger_rs::Result<()> {
    let mut car = Car::new()?;

    let batt = car.battery_v()?;
    println!("Battery: {batt:.2} V");

    let (safe, dist) = car.clear_ahead()?;
    println!("Ahead: safe={safe}, dist={dist:?}");

    if safe {
        println!("Driving forward…");
        car.forward(2000, Duration::from_millis(500))?;
    }

    car.close()?;
    Ok(())
}
