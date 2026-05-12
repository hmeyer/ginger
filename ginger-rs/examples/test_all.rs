//! Interactive hardware test — exercises every component in sequence.

use std::io::{self, Write};
use std::thread::sleep;
use std::time::Duration;

use ginger_rs::camera::Camera;
use ginger_rs::car::Car;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn section(n: u8, name: &str) {
    println!("\n┌─────────────────────────────────────────┐");
    println!("│  {n}. {name:<38}│");
    println!("└─────────────────────────────────────────┘");
}

fn ok()  { println!("  ✓ PASS"); }
fn wait_enter(prompt: &str) {
    print!("  {prompt} [Enter] ");
    io::stdout().flush().unwrap();
    let mut s = String::new();
    io::stdin().read_line(&mut s).unwrap();
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> ginger_rs::Result<()> {
    println!("╔═════════════════════════════════════════╗");
    println!("║   Ginger — full hardware test (Rust)    ║");
    println!("╚═════════════════════════════════════════╝");

    let mut car = Car::new()?;
    println!("  Car initialised (servos centred).");

    // ── 1. Battery / ADC ─────────────────────────────────────────────────────
    section(1, "ADC — battery + light sensors");
    let batt = car.battery_v()?;
    println!("  Battery:     {batt:.2} V");
    let (l, r) = car.light()?;
    println!("  Light left:  {l:.2} V");
    println!("  Light right: {r:.2} V");
    if batt < 6.0 { println!("  ⚠ Battery low!"); }
    ok();

    // ── 2. LEDs ───────────────────────────────────────────────────────────────
    section(2, "LEDs — 8× WS2812B");
    for (name, r, g, b) in [
        ("Red",   255u8, 0,   0  ),
        ("Green", 0,     255, 0  ),
        ("Blue",  0,     0,   255),
        ("White", 128,   128, 128),
    ] {
        println!("  {name}…");
        car.leds.set_all(r, g, b);
        car.leds.show()?;
        sleep(Duration::from_millis(400));
    }
    car.leds.clear()?;
    ok();

    // ── 3. Buzzer ─────────────────────────────────────────────────────────────
    section(3, "Buzzer — GPIO 17");
    println!("  Beeping 3 times…");
    car.buzzer.beep(3, 100, 100);
    ok();

    // ── 4. Infrared sensors ──────────────────────────────────────────────────
    section(4, "Infrared line sensors — GPIO 14/15/23");
    println!("  Reading 5 samples (try moving a surface under the sensors):");
    for i in 1..=5 {
        let (l, c, r) = car.ir.read_all();
        println!("  [{i}]  left={l}  center={c}  right={r}");
        sleep(Duration::from_millis(400));
    }
    ok();

    // ── 5. Ultrasonic ────────────────────────────────────────────────────────
    section(5, "Ultrasonic — HC-SR04  GPIO 27/22");
    println!("  Reading 5 distances (move your hand in front):");
    for i in 1..=5 {
        let d = car.us().distance_cm();
        match d {
            Some(cm) => println!("  [{i}]  {cm:.1} cm"),
            None     => println!("  [{i}]  — (timeout / out of range)"),
        }
        sleep(Duration::from_millis(400));
    }
    ok();

    // ── 6. Pan/tilt servos ───────────────────────────────────────────────────
    section(6, "Pan/tilt servos — PCA9685 ch 8/9");
    println!("  Pan sweep: right (45°) → centre (90°) → left (135°)");
    for &angle in &[45.0f32, 90.0, 135.0, 90.0] {
        car.pan_tilt().set_pan(angle)?;
        sleep(Duration::from_millis(500));
    }
    println!("  Tilt sweep: down (60°) → centre (90°) → up (120°)");
    for &angle in &[60.0f32, 90.0, 120.0, 90.0] {
        car.pan_tilt().set_tilt(angle)?;
        sleep(Duration::from_millis(500));
    }
    ok();

    // ── 7. Motors ────────────────────────────────────────────────────────────
    section(7, "Motors — 4WD via PCA9685");
    wait_enter("Place the car on the floor with space in all directions, then press");

    let duty = 1800i32;
    let dur  = Duration::from_millis(500);
    for (label, l, r) in [
        ("Forward",    duty,  duty ),
        ("Backward",  -duty, -duty ),
        ("Turn left", -duty,  duty ),
        ("Turn right",  duty, -duty),
    ] {
        println!("  {label}…");
        car.motors().drive(l, r)?;
        sleep(dur);
        car.motors().stop()?;
        sleep(Duration::from_millis(300));
    }
    ok();

    // ── 8. Camera ────────────────────────────────────────────────────────────
    section(8, "Camera — OV5647 via libcamera");
    println!("  Starting camera (warmup takes ~1 s)…");
    let cam   = Camera::new()?;
    let frame = cam.get_frame();
    println!("  Captured {}×{} YUYV frame ({} bytes)", frame.width, frame.height, frame.data.len());
    frame.save_ppm("/tmp/ginger_test.ppm").unwrap();
    println!("  Saved to /tmp/ginger_test.ppm");
    ok();

    // ── Done ─────────────────────────────────────────────────────────────────
    car.close()?;
    println!("\n╔═════════════════════════════════════════╗");
    println!("║   All tests passed.                     ║");
    println!("╚═════════════════════════════════════════╝");

    Ok(())
}
