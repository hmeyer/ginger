fn main() {
    // ISO-8601 UTC so the web UI can parse it with `new Date()` and
    // render it in the viewer's local timezone.
    let ts = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .map(|o| String::from_utf8(o.stdout).unwrap_or_default())
        .unwrap_or_default();
    println!("cargo:rustc-env=BUILD_TIME={}", ts.trim());
    // Force this script to rerun every build so the timestamp stays
    // fresh. Cargo's no-directive default is "rerun on package file
    // change", which fingerprints incremental builds and silently
    // stales BUILD_TIME (the WebUI then shows the timestamp of the
    // last build that *triggered* the script, not the current one).
    // Pointing rerun-if-changed at a path that doesn't exist opts into
    // "rerun every time" — the documented escape hatch.
    println!("cargo:rerun-if-changed=build.rs.always-rerun");
}
