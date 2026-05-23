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

## Tomorrow's loop

1. **Confirm the fix is live and the SLAM threads stay up under load.**
   `git pull` on the Pi (already auto-deployed by `ginger-pull`), open
   `/api/slam/map`, drive forward, then back, then turn. The snapshot
   must keep updating (look at the keyframes growing); `n_lost`
   increment is fine, a *frozen* snapshot is the regression. Check
   `journalctl --user-unit=ginger | grep panic` is empty.

2. **Drive in ≤ 1 s pulses** through the kitchen, *checking the camera
   view* (`/api/camera/frame`) between pulses. The current trace had
   a 1 s forward pulse at 35% duty producing only ~20–26 px disparity —
   that's the cadence the gates are tuned for. Pulses that are too
   long or too aggressive cause motion blur and that's a separate
   failure mode we are not trying to solve yet.

3. **Diagnose the first reproducible LOST and ship a targeted fix.**
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

4. **Stop only when a multi-minute kitchen exploration session reports
   no `lost: true` snapshots.** Save the live `/api/slam/map` snapshot
   that proves it, so the success criterion is verifiable.

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
