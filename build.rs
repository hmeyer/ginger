fn main() {
    // ISO-8601 UTC so the web UI can parse it with `new Date()` and
    // render it in the viewer's local timezone.
    let ts = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .map(|o| String::from_utf8(o.stdout).unwrap_or_default())
        .unwrap_or_default();
    println!("cargo:rustc-env=BUILD_TIME={}", ts.trim());
    // No rerun-if-changed → build script always runs → timestamp always fresh
}
