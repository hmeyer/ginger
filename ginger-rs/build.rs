fn main() {
    let ts = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%d %H:%M UTC"])
        .output()
        .map(|o| String::from_utf8(o.stdout).unwrap_or_default())
        .unwrap_or_default();
    println!("cargo:rustc-env=BUILD_TIME={}", ts.trim());
    println!("cargo:rerun-if-changed=build.rs");
}
