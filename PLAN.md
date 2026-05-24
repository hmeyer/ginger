# PLAN: drive + iterate the SLAM tracking-LOST loop

## Status

Tracking was getting "almost immediately LOST" while driving. Live
journal showed why:

```
21:50:16  slam: relocalization stuck for 151 frames with BoW ready (117 kf, 7055 pts)
  — discarding, re-bootstrapping
21:50:16  slam: two-view init OK — 121 pts, model=essential, R_H=0.39, matches=158
21:50:16  thread 'slam-mapper' panicked at src/slam/mapper.rs:425:66:
          index out of bounds: the len is 115 but the index is 117
21:50:16  thread 'slam'        panicked at src/slam/frontend.rs:567:39:
          called `Result::unwrap()` on an `Err` value: PoisonError { .. }
```

A loop-closure pose-graph optimization was running off-lock when the
frontend's "relocalization gave up" path called `reset_world()`. The
new bootstrap inserted keyframes with ids ≥ `n`; the writeback indexed
`poses[id]` and panicked, poisoning the world mutex and taking the
tracking thread with it. `/api/slam/map` then returned the **last
snapshot before the crash**, which made tracking *look* fine in the
HUD while in fact both SLAM threads were dead — hence "immediately
LOST" the moment the user drove past the (frozen) mapped area.

**The panic is now fixed and shipped** (commit on `main`):

* `apply_loop_correction` extracted as a testable helper.
* Bails if the gauge-fix `origin` is no longer alive (`reset_world()`
  ran underneath the optimization).
* Uses `poses.get(id)` so keyframes inserted after the snapshot are
  skipped, not OOB-indexed.
* Three regression tests cover happy-path, mid-flight insert, and
  reset-under-us.

What we **don't yet know** is what the *real* tracking-loss behaviour
looks like once the threads stay alive. The HUD reported "tracking:
37/39 inliers" with `n_lost: 5` and `last_lost_reason: "weak 0/64
inliers"` — that "0/64" pattern (lots of matches, zero pass the
reprojection gate) is a smell: the const-velocity prediction is wildly
wrong, or the matches are mostly wrong correspondences, or the gate is
too tight for this descriptor noise. But that diagnosis came from a
frozen snapshot, so it might or might not be representative.

## Progress log

**2026-05-24 — session 1 (Claude + Ginger)**

* **Step 1 — fix is live and threads stay up: DONE.** Verified the
  panic-fix binary by polling `/api/slam/map` at ~3 Hz across two
  drive sessions (156 + 606 ticks, **0 frozen snapshots**). Service
  PID stayed put; no panic strings. Bootstrap succeeded during the
  test (kf 0 → 383 across the run), inlier ratios 60–95 %.
* **Step 2 — drive-in-pulses cadence: PARTIALLY DONE.** Did short
  forward / back / turn pulses in the middle of the living room
  (drying-rack + basket ~1 m on either side). `n_lost` stayed at 2
  the entire time — *no new LOST events triggered*. Real translation
  needed `duty≈1500` (≈0.7 s pulses ≈40 cm); 35-raw didn't move the
  wheels at all. That gotcha is now documented in `CLAUDE.md` and the
  step-2 wording above.
* **Step 3 — diagnose first reproducible LOST: BLOCKED.** No LOST
  reproduced in this session. The "0/64 weak inliers" pattern from
  the original journal trace looks more and more like the
  frozen-snapshot artifact: when the threads are actually alive and
  the robot moves at a sane cadence, tracking is healthy. The next
  session needs *deliberately adversarial* driving — see *Next* below.

## Next

1. **Move around enough to actually lose tracking.** Easy cases ruled
   out by session 1: gentle pulses in a textured scene. Try, in order:
   * Sustained ≥ 2 s forward at `duty≈1800` — pushes the const-velocity
     predict harder and increases motion blur on this rolling-shutter
     sensor.
   * Fast in-place spins (`±1800` for 1+ s) — pure rotation gives no
     parallax, breaks any tracker that relies on translation.
   * Drive into a low-texture area (a blank-ish wall, the dark area
     under the laundry rack) — match supply collapses, reproj gate
     starves.
   * U-turn so the camera *leaves* the mapped region — forces
     relocalization-from-cold; this is where the original bug bit.

2. **Diagnose the first reproducible LOST and ship a targeted fix.**
   Likely candidates, in rough order of suspicion:

   * **CV-prediction blow-up on motion onset / direction change.**
     `last_lost_reason: "weak 0/64 inliers"` with this many matches
     means the predict was so far off that none of the matches lie
     inside the 8 px/fx inlier gate. Two ways to attack: clamp the
     predicted twist by a sane max-speed bound, or fall back to "no
     motion" predict on the first frame after `relocalize()`. The
     latter is already done — confirm it actually fires and check
     whether two-frame turn-onsets need similar handling.
   * **Inlier reprojection gate too tight at this hardware noise
     floor.** 8 px/fx was already widened from 5. Try 10 or 12 and see
     if the 0/64 turns into N/64.
   * **Map points poorly conditioned in the current view.** If many
     points are far + observed once, the predict has to be exact.
     Tighten triangulation depth/parallax, or score points by
     "n_observations" before using them for tracking.

   Intrinsics are no longer a suspect: the verified ChArUco calibration
   (`slam.toml`, rms 0.289 px) landed on `main` ahead of this work.

3. **Success criterion (revised).** The goal is no longer just "no
   `lost: true` snapshots" — it's that the robot can *explore the
   room using tracking*: drive across the floor, around the basket,
   under the rack, return to the start, and the resulting map covers
   that path without re-bootstrapping. Save the `/api/slam/map`
   snapshot from such a session.

## Workflow

* Drive via the existing HTTP endpoints — short bursts only, and
  always look at `/api/camera/frame` first to confirm the camera sees
  what we think.
* Each iteration: form a hypothesis, write the targeted fix, add a
  test if the failure can be reproduced headlessly (preferred — see
  the `relocalizes_after_track_loss` style in
  `src/slam/frontend.rs`), run the workspace DoD, `make deploy`,
  observe.
* Keep the *running TODO* (what's been tried, what worked) in this
  PLAN.md and delete it when kitchen-exploration succeeds.

## Out of scope

* Camera calibration. The FOV-derived prior is good enough for a
  bench test; a real ChArUco run is a separate work item.
* Loop closure quality tuning. The mapper panic is fixed; observed
  loop closure behaviour can be revisited once tracking is reliable.
* Frontend panic-resilience around the world mutex. With the mapper
  panic gone we shouldn't see a poisoned lock in normal operation;
  defense-in-depth there is a deferred task.
