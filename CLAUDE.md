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

Prefer **not** to compile on the Pi — see *Deploying to the Pi* below.
If you do build on it (offline iteration, no network), always build
release: `cargo build --release`. Debug is 10–20× slower for JPEG
encoding.

### The libcamera feature gate

`libcamera` is an opt-out **default** feature — the only system-coupled
dep. On a dev machine / CI / this environment, work
`--no-default-features`: that swaps in the mock camera and exercises the
**full** SLAM pipeline headless. The Pi build stays a plain
`cargo build --release` (feature on by default). Never make code paths
that only compile with libcamera; keep the mock path first-class.

## Deploying to the Pi

The Pi is too slow to compile on; ship a CI-built binary instead.

```bash
make deploy                                # = git push + start the burst pull
make deploy ARGS="--force-with-lease"      # extra args forwarded to git push
journalctl --user -u ginger-pull -f        # watch a deploy land
```

`scripts/deploy.sh` pushes, then `restart`s `ginger-pull.service` —
`scripts/pull-burst.sh`, a 10s loop that polls GitHub Actions until the
new `ginger-aarch64` artifact is up (~3 min), verifies its SHA-256,
atomically replaces `target/release/ginger`, and exits. The
`ginger-watch.path` unit detects the binary swap and restarts
`ginger.service`. No recurring timer — the service is idle between
deploys; merges done via the GitHub web UI need a manual
`systemctl --user restart ginger-pull.service`.

Auth: `gh auth login` is enough (`pull-binary.sh` falls through to
`gh auth token`); see *Deploying CI builds* in `README.md` for PAT /
env-var alternatives. Exit codes: `0`=installed, `10`=no new artifact
yet, `11`=no token. Deploys only fire from `main`.

## Driving / probing the live robot from a shell

The same HTTP endpoints the WebUI uses are also the cleanest way to
script test drives. Two gotchas worth knowing before you `curl`:

* **`/api/drive` left/right are raw PWM, not 0–100.** Range is ±4095
  (hard cap in `devices/motors.rs`), the WebUI joystick saturates at
  ±2000 (see `DUTY` in `bin/web/index.html`). So when the running
  TODO talks about "35% duty" it means *~35 % of the WebUI scale =
  `~700`* — not `35`. `35` is essentially zero, the wheels don't move.
* **Always stop before sleeping more than a beat.** No watchdog will
  cut motion if your script crashes mid-pulse. The forward path is
  also guarded by a 30 cm ultrasonic stop that auto-unlocks on a
  reverse command (`robot/supervisor.rs`).

A reasonable single pulse + observation loop:

```bash
curl -sX POST localhost:8080/api/drive -H 'content-type: application/json' \
     -d '{"left":1500,"right":1500}'     # firm forward
sleep 0.7                                 # ≤ 1 s; longer = motion blur
curl -sX POST localhost:8080/api/stop
curl -s "localhost:8080/api/camera/frame?w=480&q=70" -o /tmp/now.jpg
curl -s localhost:8080/api/slam/map | jq '{kf:.n_keyframes, pts:.n_points,
       tracking, n_lost, status, last_lost_reason}'
```

For SLAM verification specifically: poll `/api/slam/map` every ~300 ms
in the background — if the JSON body is *identical* two ticks in a
row the supervisor thread is dead and the HUD is serving a stale
snapshot (the failure mode `fd8fc13` fixed).

## Monitor recipes (the Monitor tool)

The Monitor tool emits a notification per stdout line of the supplied
script. Two failure modes have burned us; both are silent (the monitor
stays running but you never get told anything).

### 1. Pipe buffering — every loop-shape monitor needs a `stdbuf` shim

When the script is **bash** (not a pipeline like
`tail -f log | grep --line-buffered ...`), stdout is **fully buffered**
because it's connected to a pipe, not a terminal. Lines you `echo`
sit in libc's 4 KB buffer and never reach the monitor until the
script exits.

The fix is one line at the top of the script:

```bash
exec 1> >(stdbuf -oL cat)
```

This re-routes the script's stdout through `stdbuf -oL cat`, which
runs with **line-buffered** output. Every `\n` from your loop now
flushes immediately. Use this for any monitor script that:

* loops with `while true; ...; done`,
* emits via `echo`, `printf`, `comm`, `jq -r`, etc. (not a piped
  `grep --line-buffered`),
* relies on the harness's per-line notification semantics.

For pipeline-shaped monitors (`tail -f log | grep …`) the
`--line-buffered` flag on `grep` (or `awk -W interactive`, `sed -u`)
is sufficient and the `exec 1>` prefix isn't needed.

### 2. Silent-fallback poll loops eat errors and never exit

The "wrap a remote call in `|| echo "[]"` so the loop survives one
transient failure" pattern is **dangerous** if the command syntax is
wrong: the loop polls forever, never emits, never exits. This is
how we lost a CI-watch session to `gh pr checks --json` (the flag
doesn't exist on that subcommand — `gh pr view <PR> --json
statusCheckRollup` is correct).

Mitigations:

* **Test your remote command interactively first.** One `gh ...`
  invocation in a normal Bash call before arming the monitor catches
  the syntax error in 1 s instead of a 5-minute timeout.
* **Test the monitor against a known-complete state.** For a CI
  watch, arm it on a PR that has already finished — events should
  arrive within seconds. If they don't, your script is broken; fix
  before arming on an in-flight PR.
* **Don't swallow errors silently when designing the gate** — log
  to stderr so the output file shows the failure. E.g.
  `s=$(gh pr view 55 --json statusCheckRollup 2>>/tmp/mon-err.log \
       || { echo "gh failed" >&2; echo '{"statusCheckRollup":[]}'; })`.

### Pattern that works (CI-watch example)

```bash
exec 1> >(stdbuf -oL cat)   # ← load-bearing; without this, silence
prev=""
while true; do
  s=$(gh pr view <PR#> --json statusCheckRollup 2>/dev/null \
        || echo '{"statusCheckRollup":[]}')
  cur=$(jq -r '.statusCheckRollup[] | select(.status=="COMPLETED")
        | "\(.name): \(.conclusion)"' <<<"$s" | sort)
  comm -13 <(echo "$prev") <(echo "$cur")  # diff: newly-resolved checks
  prev=$cur
  jq -e '.statusCheckRollup | length > 0 and all(.status=="COMPLETED")' \
       <<<"$s" >/dev/null && break
  sleep 30
done
echo "all-checks-done"
```

Verified working: arms against PR #55 (already complete), all four
checks emit within ~5 s, then `all-checks-done`, then exit. If you
need to adapt this for a different state (e.g. emit on failures only,
or watch for a specific check name), change the `jq` filters — keep
the `exec 1>`, the `comm -13` diff, and the gate test.

## Definition of done for a change

A change is not done until all of these pass (this is what CI gates on):

1. `cargo test --workspace --no-default-features` — green, no
   regressions vs. the prior count. CI runs this natively on aarch64,
   so the `crates/fast` NEON paths execute (not just the scalar
   reference); a local x86 run only exercises the scalar branch.
2. `cargo clippy --workspace --no-default-features --all-targets -- -D warnings`.
3. `make lint` (fmt + clippy). The default-features clippy needs
   libcamera headers; on a non-Pi dev box without libcamera, lean on
   CI's `build` job for that coverage.
4. WebUI changes: the Playwright harness in `webui-tests/` still passes.

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
- **Seeded RNGs** — use `rand::rngs::SmallRng` (Xoshiro256++) with
  `SmallRng::seed_from_u64(seed)` for every reproducible stream
  (RANSAC sampling, k-means++ BoW vocabulary, the BRIEF sampling
  pattern, test fixtures). Within-build determinism (same binary +
  same seed → same stream) is what replay and the headless suite
  rely on; cross-`rand`-version byte stability is not promised and
  nothing in the tests asserts on specific bit sequences (gates are
  invariants: counts, tolerances, structural properties). Do not
  re-introduce a hand-rolled PRNG.
- **NEON discipline** — every SIMD path in `crates/fast` keeps a scalar
  reference and a `parity` test. Add NEON only at a measured hotspot.
- **Comments** — explain *why*, not *what*. The codebase favors dense,
  intentful module docstrings; match that style.
- **Errors** — crate-wide `Error`/`Result` in `src/error.rs`
  (`thiserror`). Propagate with `?`; validate only at boundaries.

## Git

- Develop on a feature branch; commit per logical unit with a
  descriptive message; push with `git push -u origin <branch>`.
- **Push and open a PR as early as possible** — even on the first
  commit that builds and lints. CI on GitHub Actions runs on x86_64
  and is *much* faster than the Pi (workspace debug builds here take
  5–10 min; CI finishes in a fraction of that). Iterate on the PR by
  **force-pushing** to the same branch
  (`git push --force-with-lease origin <branch>`) rather than opening
  a new PR per round.
- Treat CI as the primary test runner. Local `cargo test` on the Pi
  is for tight single-test iteration; the full
  `cargo test --workspace --no-default-features` definition-of-done
  belongs to CI.
- Pre-commit hook runs `make lint`; never bypass with `--no-verify`.
- Publishing to `main` on the Pi: use `make deploy` (not bare `git push`)
  so the CI build is auto-pulled — see *Deploying to the Pi*. (Pure
  docs PRs to `main` can skip `make deploy` — there's no binary to pull.)

## TODO / status tracking

The project's running TODO list lives in `README.md` ("SLAM status" →
*Deferred* / *Performance / refinement passes*); keep that section
authoritative when finishing or adding work.

A `PLAN.md` at the repo root is allowed for multi-step, multi-commit work
that needs a sequenced, revertible roadmap (e.g. a workspace-wide
migration). When present it owns the *sequencing and rationale* for the
in-flight effort; `README.md` remains the canonical status board for the
project as a whole. Delete `PLAN.md` when the migration it describes is
done — it is not a permanent design doc.
