# PLAN.md — migrate `ginger-rand` → ecosystem `rand`

In-flight migration roadmap. Delete this file when Step 9 lands.

## Goal

Replace the workspace's hand-rolled `crates/rand` (`ginger-rand`:
`Rng64` / `Rng32` / `Xs64` / `noise_u8`) with the ecosystem `rand` crate
family (`rand`, `rand_core`, `rand_chacha`, `rand_distr`). In the
process, give up **cross-version** byte-for-byte reproducibility (the
property `rand` only guarantees for `ChaCha*Rng`) while keeping
**within-build** determinism (same binary + same seed → same stream),
which is what replay and the headless test suite actually need.

## Why

- Better RNG quality: `SmallRng` (Xoshiro256++) and `StdRng` (ChaCha12)
  pass statistical batteries `xorshift64*` does not. `gen_range` is
  Lemire-debiased; today's `rng.next_u64() % n` is modulo-biased.
- `Rng` trait ergonomics everywhere (`gen_range`, `choose`,
  `sample_iter`, `rand_distr::Normal`) instead of bespoke `upto` /
  `upto_unit` / hand Box–Muller with a `.max(1e-7)` floor hack.
- Delete ~170 LOC and one workspace crate. One fewer "single source of
  truth" doc to maintain.
- Tests become invariants ("inlier count ≥ N", "reprojection error
  < ε") rather than byte-goldens ("inlier set == [3, 7, 12, 18]") —
  describes behavior, survives refactors.

## Non-goals

- We do **not** keep cross-`rand`-version byte stability. Upgrading
  `rand` may change streams; the test suite must not care.
- We do **not** preserve byte-compatibility with maps stored under the
  old PRNG. Stored maps are throwaway for this project.
- No new behavior shipped. Algorithms keep their statistical contract;
  only the bytes that realise it change.

## Two determinism layers — pinned, do not conflate

1. **Within-build** — same binary, same seed → same stream. *Preserved*
   by every seeded `rand` RNG. Replay, debugging, "did this commit
   regress?" — all unaffected.
2. **Across-version** — bumping `rand` produces the same bytes.
   *Given up* deliberately. Only `ChaCha*Rng` would have guaranteed
   this, and the only thing demanding it today is the test suite
   itself.

## Sequenced steps

Every step is a self-contained commit on
`claude/plan-rust-rng-migration-yHPI1` that leaves the workspace green
on the full DoD checklist from `CLAUDE.md` (`cargo test --workspace
--no-default-features`, `cargo clippy ... -D warnings`, `make lint`,
both `aarch64-unknown-linux-gnu` cross-checks, Playwright if WebUI
touched).

### Step 1 — Audit & classify every assertion that touches an RNG ✅

**Done.** Full classification across `crates/slam-core/`,
`crates/fast/`, `src/slam/`, `src/robot/emote.rs`, `src/camera/mock.rs`,
and the examples. Result:

- **92 invariants** — tolerance / count / bound / structural checks.
- **11 intra-build determinism** — two-run comparisons under the same
  seed. Preserved automatically by any seeded RNG.
- **0 byte-goldens.** No checked-in `.bin`, no `include_bytes!` of a
  PRNG-derived artifact, no hard-coded expected sequence, no hash
  comparison. `slam_vocab.bin` is a *runtime* file (loaded if present,
  graceful fallback otherwise per `src/slam/place.rs:20`) — not a
  test fixture.

This collapses the plan: **Steps 5 and 6 become near-no-ops** (nothing
to re-bless, vocab artifact isn't asserted on). The PRNG swap proceeds
without test surgery — every gate is on algorithmic behavior, not on
bytes.

### Step 2 — Add `rand` deps and implement `RngCore` on existing streams

- Workspace `Cargo.toml`: `rand = "0.9"`, `rand_core = "0.9"`,
  `rand_distr = "0.5"`. Add to `crates/rand`, `crates/slam-core`,
  `crates/fast`, root crate as needed.
- `crates/rand/src/lib.rs`: implement `RngCore` + `SeedableRng` on
  `Rng64`, `Rng32`, `Xs64`. Keep every inherent method (`f`, `upto`,
  `upto_unit`, `byte`, `gauss`, `range`, `unit`) unchanged.
- Update the `crates/rand` module docstring: it is now the *adapter*
  exposing our streams as `RngCore`; byte-stability of the streams
  themselves is unchanged at this step.
- DoD check: tests pass with **zero** output changes — no algorithm
  swap yet.

### Step 3 — Convert ergonomics where bytes don't move

Per call site, switch local idioms to `rand::Rng` methods **only where
they don't change the byte stream**:

- Allow `&mut Rng64` to satisfy `&mut impl RngCore` at trait-bounded
  call sites.
- Replace bespoke loops with `Rng` methods only when the resulting
  byte stream is provably identical to today's.
- When in doubt, leave it.

DoD check: no test re-blessing; bit-for-bit identical outputs.

### Step 4 — Migrate the no-golden call sites to `SmallRng`

These have no checked-in goldens and no cross-version implications:

- `src/robot/emote.rs` — `Xs64` → `rand::rngs::SmallRng`. Buzzer/LED
  show is seeded from system time anyway.
- `crates/fast/src/image.rs`, `crates/fast/src/fast.rs` — image-fuzz
  fixtures → `SmallRng` with a fixed seed (within-build determinism
  preserved).

Mark `Xs64` `#[deprecated]` once it has no callers; delete in Step 8.

### Step 5 — Rewrite byte-goldens as invariants — **N/A (no goldens)**

Step 1 found zero byte-goldens. Skipped. If future audits reveal one,
this slot is reserved.

### Step 6 — Decouple the BoW vocab from a specific PRNG — **N/A**

`slam_vocab.bin` is a runtime artifact, not a test gate. No test
compares its bytes; no `include_bytes!` ties code to a specific PRNG.
The runtime fallback in `src/slam/place.rs` already handles its
absence. Verify in Step 9 that regenerating it under the new PRNG
still satisfies the runtime path; no migration work otherwise.

### Step 7 — Swap the PRNG implementation

With goldens now invariant-based, the algorithm swap is safe:

- `Rng64::next_u64` → delegate to a held `ChaCha8Rng` or `SmallRng`
  (decision made in this step). Same for `Rng32`.
- Keep the inherent-method API for one transition commit so call sites
  don't churn in the same diff; remove the inherent methods in favor
  of trait methods in a follow-up.
- This is the one deliberate "re-bless" commit. Message must say so
  explicitly, matching the CLAUDE.md convention for this class of
  change.

DoD check: tests green, cross-target green,
`examples/slam_bench.rs` shows no >5% regression on init / track / BA
timings.

### Step 8 — Demolish `ginger-rand`

- Move call sites from `ginger_rand::Rng64` to the chosen `rand` type
  directly.
- Move `noise_u8` (a pure integer hash, not a stream) next to its sole
  caller in `src/camera/mock.rs` — it doesn't belong with the PRNG
  crate.
- Remove `crates/rand/` from the workspace; drop `ginger-rand` from
  every `Cargo.toml`.
- Update `CLAUDE.md` "Shared primitives" bullet: we use `rand` from the
  ecosystem with seeded `StdRng` / `SmallRng` for within-build
  reproducibility.

DoD check: workspace builds, tests green, no dangling references,
`make lint` clean.

### Step 9 — Final cross-target & benchmark pass, retire this file

- `cargo test --workspace --no-default-features` — test count
  ≥ pre-migration count.
- `cargo clippy --workspace --no-default-features --all-targets
  -- -D warnings`.
- `cargo check -p ginger-slam-core --target aarch64-unknown-linux-gnu`,
  same for `ginger-fast`.
- Run `examples/slam_bench.rs` before/after; flag any >5% regression.
- WebUI Playwright tests if anything frontend-adjacent moved.
- **Delete `PLAN.md`** — per `CLAUDE.md`, this file lives only for the
  duration of the migration.

## Properties of this sequencing

- **Decoupled.** Steps 1–6 ship a more robust test suite with zero
  RNG change. Stopping after Step 6 keeps most of the upside ("tests
  describe behavior") at zero risk to bytes.
- **Revertible.** Each step is one commit, restoring the prior green
  state.
- **Bisectable.** Regressions land on a single step, not a 1000-line
  big-bang.
- **CLAUDE.md-compliant.** Every step ends with the DoD checklist;
  refactors stay behavior-preserving; the one deliberate behavior
  change (Step 7) is its own commit with an explicit re-bless message.
