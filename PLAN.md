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

## Progress log (cont.)

**2026-05-24 — session 2 (Claude + Ginger)**

* **First reproducible LOST: fast in-place spin.** Sequence:
  forward 1 s @ `duty=1800` then *immediately* spin `±1800` for 1.2 s
  (no stop between). Tracked healthily through the forward
  (60–71 inliers, kf 4 → 26) and into the spin (30/48, kf 27 with
  +41 newly triangulated pts), then collapsed in ~14 frames:
  *only 8 map matches → soft-lost → 3 soft-lost frames → Stage::Lost*.
  Lost for 148 frames (~5 s) with the **map preserved** at 29 kf /
  196 pts — BoW-ready give-up budget. `RELOC_MAX_FRAMES_BOW` fired
  at frame 148; `reset_world` + re-bootstrap with zero panic. The
  panic fix is solid; the failure is upstream.
* **Slow spin survives.** Same starting state, spin `±600` for 3 s,
  ended at 15 kf / 187 pts, still `tracking=true`. So the failure
  mode is *rate*-driven, not a structural map issue.
* **Diagnosis (mono-SLAM intrinsic).** Two compounding effects:
  1. *Mapper can't keep up at high angular velocity.* `needs_keyframe`
     fires every couple of frames during a fast pan, but each kf's
     triangulation against its covisible neighbours takes finite time
     (`LOCAL_BA_K=6` window, `LOCAL_BA_ITERS=5`).
  2. *Pure rotation has zero parallax*, so even when the mapper does
     pick up a fast-spin kf, it can't triangulate fresh points against
     the just-previous (also-rotation-only) kf. New geometry only
     appears for pairs that bracket some translation, which a pure
     spin doesn't provide.
  Net effect: the camera sweeps into unmapped scenery faster than map
  points can be created there. Tracking starves on "only N map matches".
  This is a generic mono-SLAM limitation (ORB-SLAM3 et al. also need
  IMU or translation-during-rotation to bridge fast pans).
* **Two-tier exploration limit observed empirically (this room):**
  * Safe / mapping-friendly: differential ≤ ~1200 (e.g. `±600`
    in-place spin, or curved forward turns).
  * Tracking-breaking: differential ≥ ~3000 (e.g. `±1500..±2000`
    in-place spins).
  Threshold not narrowed further this session; ~2000 differential
  is the next data point to gather.

## Next

1. **Demonstrate "explore the room" works at safe rates.** Drive a
   multi-pulse session staying under `|left - right| ≤ 1200`,
   preferring curved arcs (translation + rotation) over in-place
   spins. Success = `tracking=true` throughout, map grows monotonically,
   `n_lost` stays at its starting value. Save the final
   `/api/slam/map` snapshot.

2. **Pick a real SLAM-side fix for fast rotation** (deferred to a
   later session — non-trivial). Realistic options in rough
   impact-to-cost order:
   * **Soft angular-velocity clamp in the supervisor** (~10 LOC):
     when `|left - right|` exceeds a threshold, scale it down while
     keeping the translation mean. Saves users from themselves; does
     not advance the SLAM's actual capability.
   * **Frame-to-frame fallback when map matches starve**: match
     current frame against `self.prev` for orientation-only updates
     during shaky spans. Keeps the trajectory rotational-consistent
     across short pans; modest scope (~100 LOC).
   * **Panoramic mode** (proper fix): detect pure-rotation segments
     and switch to 2D-2D homography tracking, populate BoW from
     orientation-only keyframes, fold them into the metric map when
     translation resumes. Multi-day, touches `slam-core`.

3. **Out of scope here:** IMU fusion (no IMU on this chassis).
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
