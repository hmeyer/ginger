# PLAN: DeepInfra vision-model advisor

## Goal

Give Ginger a VLM-backed advisor that annotates the SLAM map with
semantic data (room labels, landmarks, free-space goal hints, change
notes) for the autonomous planner to consume. The geometric map stays
authoritative for collision / safety; the advisor is *soft* data that
the planner may weight but never override the ultrasonic stop or the
SLAM occupancy with.

Lives on the binary side. `slam-core` stays camera-free.

## Why advisory, not in-the-loop

Network round-trip to DeepInfra is hundreds of ms — too slow for the
~10 Hz drive loop. Advisor runs on its own thread, triggered by SLAM
events (~0.1–0.5 Hz), and writes results into a `SemanticMap` keyed
by `Keyframe` id. The drive loop reads `SemanticMap` opportunistically;
absence is normal.

## Model choice

**Default: `Qwen/Qwen3-VL-235B-A22B-Instruct`** on DeepInfra.
$0.20 in / $0.88 out per M tokens, $0.11 cached. At advisor call
rates (≤ 1/s, usually far less), latency is irrelevant and the
quality jump for room ID / change detection is worth it. Stable
system prompt + prior-frame summary in the prefix → cache hits
amortize cost.

Fallback: `Qwen/Qwen3-VL-30B-A3B-Instruct` if latency or cost
becomes a problem; same prompt shape, same JSON schema.

Both go through the OpenAI-compatible chat-completions endpoint
DeepInfra exposes, so the client code is identical.

## The contract (do this first, before any client code)

Once SLAM data depends on these fields, changing them is a migration.
Nail the JSON schema down before writing the HTTP call.

```jsonc
{
  "room_label":  "kitchen",         // or null if uncertain
  "room_confidence": 0.82,          // 0..1
  "semantic_landmarks": [
    { "label": "doorway", "bbox_norm": [0.12,0.30,0.28,0.95], "confidence": 0.7 }
  ],
  "goal_hints": [
    { "kind": "unexplored_passage", "bearing_deg": -15, "confidence": 0.6, "note": "open doorway, dark beyond" }
  ],
  "free_space_note": "carpet ahead clear ~2 m, chair on the right",
  "change_notes": [],               // populated only when prior-frame summary is provided
  "advisor_version": 1              // bump when schema changes
}
```

Bearings are camera-frame degrees (right-positive), bboxes are
normalized image coordinates `[x0,y0,x1,y1]`. No world coordinates
come out of the model — projection from bearing/bbox to map happens
locally using the keyframe pose and intrinsics, which keeps the
model's spatial errors quarantined.

Mitigations baked into the schema:
- Per-field confidence; a `min_confidence` threshold gates commit.
- `room_label` only commits after **N consecutive keyframes** in the
  same covisibility cluster agree (start with N=3).
- `goal_hints` must geometric-check against current free-space before
  the planner acts on them.
- `advisor_version` bumps on any schema change so the `SemanticMap`
  can ignore stale entries on load.

## Module layout

```
src/advisor/
  mod.rs            – public API: AdvisorHandle, start_advisor()
  client.rs         – DeepInfra HTTP (reqwest), image encoding, schema
  trigger.rs        – decides when to call from a SlamSnapshot delta
  semantic_map.rs   – the SemanticMap data structure + commit policy
  prompt.rs         – system prompt + few-shot, frozen with the schema
```

`SemanticMap` lives here, *not* in `slam-core`, because it depends on
camera frames and on the schema above. It references `Keyframe` ids
by value (u32) and re-checks `Map::keyframe(id)` before reading — same
defensive pattern `apply_loop_correction` uses post-`reset_world()`.

## New dependencies

- `reqwest = { version = "0.12", features = ["json", "rustls-tls"] }`
- `base64 = "0.22"` (data-URL encoding for image input)
- Reuse existing `serde`, `serde_json`, `tokio`, `thiserror`.

No native TLS — `rustls-tls` keeps the cross-build for
`aarch64-unknown-linux-gnu` clean. Verify on Step 1.

## Sequenced commits

Each step ends green: `cargo test --workspace --no-default-features`,
`make lint`, and `cargo check` for the two camera-free crates on
`aarch64-unknown-linux-gnu` (per CLAUDE.md "Definition of done").

### Step 1 — wire dependencies, empty module

- Add reqwest + base64 to `Cargo.toml`.
- Create `src/advisor/mod.rs` with a stub `AdvisorHandle` and
  `start_advisor()` that does nothing.
- Add `Error::Advisor(String)` variant in `src/error.rs`.
- Verify aarch64 `cargo check` for the binary still passes.

Revertible: dropping the module compiles fine.

### Step 2 — schema + SemanticMap, no networking

- `semantic_map.rs`: `AdvisorReport`, `SemanticMap` (HashMap<u32,
  AdvisorReport> + room-label voting state), commit policy with
  `min_confidence` and N-agreement.
- Unit tests covering: confidence-below-threshold drop, N-agreement
  promotion, version mismatch on load, keyframe-died-since
  defensive skip.
- No advisor calls yet; nothing wired into SLAM.

### Step 3 — prompt + offline schema validator

- `prompt.rs` with the frozen system prompt + 1–2 few-shot exchanges.
- `client::parse_response()` validating model output against the
  schema; rejects on malformed JSON, out-of-range bearings, bboxes
  outside `[0,1]`. Pure function, fully unit-tested with fixtures
  under `tests/fixtures/advisor/`.
- Still no HTTP.

### Step 4 — DeepInfra client behind a transport trait

- `trait AdvisorTransport { async fn call(&self, req: &AdvisorRequest)
  -> Result<String>; }` with two impls:
  - `DeepInfraTransport` — real reqwest call, OpenAI-compat
    chat-completions endpoint, `response_format: { type: "json_object" }`,
    7 s timeout, single retry on 5xx.
  - `MockTransport` — replays fixture JSON; used in tests and when
    `--no-default-features` (keeps headless suite offline-only).
- API key from env `DEEPINFRA_API_KEY`; missing key → advisor is a
  no-op that logs once and stays disabled. Never panic on missing
  credentials.
- Manual smoke test (not in CI): one round-trip against the real API
  with a still frame from the mock camera; output validated and
  logged.

### Step 5 — trigger module + supervisor wiring

- `trigger.rs` consumes `SlamSnapshot` deltas. Calls advisor when:
  - new keyframe inserted in a covisibility cluster with no committed
    `room_label`, OR
  - `last_lost_reason` changed since last call, OR
  - supervisor obstacle flag flipped, OR
  - heartbeat: every M=20 keyframes regardless.
- Rate-limit: never more than 1 call per 3 s (back-pressure cap).
- Advisor runs on its own tokio task spawned from `bin/main.rs`,
  receives `SlamSnapshot` clones over a watch channel, holds an
  `Arc<RwLock<SemanticMap>>`.

### Step 6 — read path + debug endpoint

- Planner (or, until there's a planner: the existing supervisor)
  reads `SemanticMap` to bias goal selection toward
  `unexplored_passage` hints with confidence ≥ 0.6 that geometric-
  check as drivable.
- New `/api/slam/semantic` endpoint returning the committed
  `SemanticMap` as JSON for WebUI debugging. WebUI overlay is
  out-of-scope for this plan; raw JSON is enough for now.

### Step 7 — change detection

- When the last committed report for a covisibility cluster exists,
  include its summary in the next prompt's prefix (this is what the
  cached-input pricing is for).
- Populate `change_notes` only when prior context is provided.
- Test: replay-mode fixture where a chair appears between two visits
  to the same keyframe cluster; assert a non-empty `change_notes`.

### Step 8 — telemetry + cleanup

- Counters: calls/min, error rate, mean latency, commits accepted vs.
  rejected (split by reason). Surface on `/api/slam/semantic?stats=1`.
- Update README "SLAM status" section per CLAUDE.md convention.
- **Delete this `PLAN.vision-model.md`** — the plan is no longer the
  source of truth.

## Determinism / test discipline

- Replay and `--no-default-features` paths use `MockTransport` only.
  No live HTTP in the headless suite.
- Advisor task is **not** in the SLAM determinism contract: the
  `local_ba` / `map` observation ordering is unaffected because the
  semantic write path never touches `Map`. `SemanticMap` is parallel
  storage.
- The N-agreement / confidence logic uses simple arithmetic — no
  RNG. If RANSAC-style sampling shows up later, use `SmallRng`
  seeded from the keyframe id.

## Out of scope (don't drift)

- Local on-device VLM. Out of compute budget on a Pi 4.
- WebRTC streaming the model output. Debug endpoint is enough.
- Letting the model emit world coordinates directly. Bearings + local
  projection only.
- A planner rewrite. This plan delivers the **data**; the planner
  changes are a follow-up tracked in README's TODO once Step 6 lands.

## Open questions to resolve before Step 4

1. Image resolution / JPEG quality to send. Probably 640×480 q=70 to
   start — matches the existing `/api/camera/frame` defaults and
   keeps input tokens bounded.
2. Whether to send the *current* frame or the keyframe's stored
   frame. Current is fresher; keyframe makes the report reproducible
   on replay. Start with keyframe.
3. Min interval between advisor calls under sustained obstacle
   flapping. Hard cap at 1/3 s for now; revisit after real data.
