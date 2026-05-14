//! Capture one frame via the Camera module and save to /tmp/frame.ppm + /tmp/frame.jpg

use ginger_rs::camera::Camera;

fn main() -> ginger_rs::Result<()> {
    println!("Starting camera (includes warmup)…");
    let cam = Camera::new()?;

    let frame = cam.get_frame();
    println!(
        "Frame: {}×{}  {} bytes",
        frame.width,
        frame.height,
        frame.data.len()
    );

    frame.save_ppm("/tmp/frame.ppm").unwrap();
    println!("Saved /tmp/frame.ppm");

    // Also save a JPEG via Python-free conversion (PPM is lossless; JPEG needs an encoder crate)
    println!("Done. Open /tmp/frame.ppm to view.");
    Ok(())
}
