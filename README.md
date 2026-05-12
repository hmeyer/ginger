# Ginger

Rust driver library and web control interface for the [Freenove 4WD Smart Car Kit (FNK0043)](https://docs.freenove.com/projects/fnk0043/en/latest/) running on a Raspberry Pi 4, PCB v2.0.

## Hardware

| Subsystem | Chip / interface | Notes |
|---|---|---|
| Motors (4WD) | PCA9685 → I2C 0x40 | Channels 0–7, H-bridge pairs |
| Pan/tilt servos | PCA9685 → I2C 0x40 | Channels 8 (pan) and 9 (tilt) |
| ADC / battery / light | ADS7830 → I2C 0x48 | Ch 0 = light L, Ch 1 = light R, Ch 2 = battery |
| LED strip | 8× WS2812B → SPI0 | Bit-banged at 6.4 MHz, GRB order |
| Buzzer | Active → GPIO 17 | HIGH = on |
| Ultrasonic | HC-SR04 → GPIO 27/22 | Trigger / Echo |
| IR line sensors | 3× → GPIO 14/15/23 | Active-high: HIGH = line detected |
| Camera | OV5647 → CSI | Via libcamera, 800×600 YUYV |

## Crate layout

```
ginger-rs/
  src/
    pca9685.rs    — I2C PWM driver (PCA9685)
    motors.rs     — 4WD differential drive
    servo.rs      — Pan/tilt with invert + trim
    adc.rs        — ADS7830 battery/voltage/light ADC
    led.rs        — WS2812B strip over SPI
    buzzer.rs     — Active buzzer
    ultrasonic.rs — HC-SR04 distance sensor
    infrared.rs   — 3-sensor line tracker
    camera.rs     — OV5647 via libcamera (streaming background thread)
    car.rs        — Top-level Car struct with obstacle-avoidance safety
  bin/
    main.rs       — Web server (axum) with live sensor stream + controls
    web/
      index.html  — Embedded mobile-first control UI
  examples/
    test_all.rs   — Interactive component-by-component hardware test
    drive_test.rs — Battery check + forward drive smoke test
    camera_test.rs — Capture one frame and save as PPM
```

## Dependencies

```bash
sudo apt install libcamera-dev libclang-dev
```

Rust toolchain via [rustup](https://rustup.rs).

## Building

```bash
cd ginger-rs
cargo build
```

## Web interface

The `ginger` binary serves a mobile-first control UI on port 8080:

```bash
cargo run --bin ginger
# Open http://<pi-ip>:8080  or  http://ginger.local:8080
```

Features:
- **Live sensor feed** via SSE — battery voltage, light sensors (L/R), IR line tracker (3 dots), ultrasonic distance, all updating at 200 ms
- **Camera stream** — JPEG frames polled at ~20 fps
- **Drive controls** — D-pad (tap or hold), keyboard arrow keys, space to stop; server safety-stops motors after 500 ms of silence
- **Pan / tilt sliders** — 0–180° range
- **LED** — colour picker + off button
- **Buzzer** — hold to beep
- **Sensor toggles** — enable/disable light, IR, and ultrasonic per sensor

The UI is mobile-first: on phones it stacks camera → scrollable sensor strip → footer controls. On screens ≥ 700 px it switches to a camera + sidebar layout.

## Running the hardware test

```bash
cargo run --example test_all
```

Tests each subsystem in order: ADC → LEDs → Buzzer → IR sensors → Ultrasonic → Servos → Motors → Camera. Motor test prompts before moving.

## Camera

The `Camera` struct runs a background thread that continuously captures YUYV frames from the OV5647. Frames are published behind an `Arc` — callers either poll or block:

```rust
let cam = Camera::new()?;          // blocks until first real frame (~1 s warmup)
let frame = cam.get_frame();       // latest frame, non-blocking
let frame = cam.wait_frame();      // blocks until next new frame
let rgb = frame.to_rgb();          // YUYV → packed RGB
frame.save_ppm("/tmp/out.ppm")?;   // save without extra deps
```

## Car safety

`Car::drive()` checks the ultrasonic sensor before and during forward motion and stops automatically if an obstacle is closer than 30 cm.

```rust
let mut car = Car::new()?;
car.forward(2000, Duration::from_secs(1))?;  // stops early if blocked
car.turn_right(2000, Duration::from_millis(400))?;
car.stop()?;
car.close()?;
```
