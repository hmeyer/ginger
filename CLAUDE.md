# CLAUDE.md

Operational guide for working in this repo. Architecture and hardware
detail live in `README.md`; this file is the "how to work here" companion.

## What this is

Rust driver library + web control interface for a Freenove 4WD Smart Car
on a Raspberry Pi 4. A Cargo workspace:

- `crates/fast` — FAST-9 + grayscale image/pyramid (NEON), camera-free.
- `crates/slam-core` — camera-free geometry/optimization (Lie groups,
  two-view init, tracking, BA, BoW, PnP, Sim3 pose-graph).
- `.` (`ginger-rs`) — the hardware-coupled binary: HAL → devices →
  robot → server → bin, with `camera`/`video`/`slam` as parallel stacks.

The two `crates/*` are dependency-light and cross-compile/unit-test
without libcamera. The numerically sensitive math lives there on purpose.

## Build / test / lint

```bash
make lint                                  # fmt --check + clippy -D warnings (pre-commit hook + CI)
make build                                 # cargo build --bin ginger
cargo test --workspace --no-default-features   # full headless test (mock camera, no libcamera)
cargo check -p ginger-slam-core --target aarch64-unknown-linux-gnu  # Pi cross-check
cargo check -p ginger-fast      --target aarch64-unknown-linux-gnu
```

Always build/run release on the Pi (`cargo build --release`); debug is
10–20× slower for JPEG encoding.

### The libcamera feature gate

`libcamera` is an opt-out **default** feature — the only system-coupled
dep. On a dev machine / CI / this environment, work
`--no-default-features`: that swaps in the mock camera and exercises the
**full** SLAM pipeline headless. The Pi build stays a plain
`cargo build --release` (feature on by default). Never make code paths
that only compile with libcamera; keep the mock path first-class.

## Definition of done for a change

A change is not done until all of these pass (this is what CI gates on):

1. `cargo test --workspace --no-default-features` — green, no
   regressions vs. the prior count.
2. `cargo clippy --workspace --no-default-features --all-targets -- -D warnings`.
3. `make lint` (fmt + clippy).
4. `cargo check` for both camera-free crates on
   `aarch64-unknown-linux-gnu`.
5. WebUI changes: the Playwright harness in `webui-tests/` still passes.

Refactors must be behavior-preserving and keep the test count
non-decreasing. New behavior needs new tests in the same change.

## Conventions

- **Coordinate / pose conventions are pinned and tested** — do not
  "fix" them casually. `slam-core` operates only on *calibrated*
  (normalized, undistorted) image coordinates; pixel-space stays in
  `src/slam`. SE(3) twist is `xi = [rho (translation); phi (rotation)]`;
  poses are `Tcw` (world→camera). See `crates/slam-core/src/lie.rs` and
  the module docstrings.
- **Determinism** — replay/test paths must stay deterministic.
  Observation ordering in `local_ba`/`map` is load-bearing; preserve it.
- **NEON discipline** — every SIMD path in `crates/fast` keeps a scalar
  reference and a `parity` test. Add NEON only at a measured hotspot.
- **Comments** — explain *why*, not *what*. The codebase favors dense,
  intentful module docstrings; match that style.
- **Errors** — crate-wide `Error`/`Result` in `src/error.rs`
  (`thiserror`). Propagate with `?`; validate only at boundaries.

## Git

- Develop on the assigned feature branch; commit per logical unit with a
  descriptive message; push with `git push -u origin <branch>`.
- Do not create a PR unless explicitly asked.
- Pre-commit hook runs `make lint`; never bypass with `--no-verify`.

## TODO / status tracking

There is no separate plan file (PLAN.md was intentionally removed). The
project's running TODO list lives in `README.md` ("SLAM status" →
*Deferred* / *Performance / refinement passes*). Keep that section
authoritative when finishing or adding work.
