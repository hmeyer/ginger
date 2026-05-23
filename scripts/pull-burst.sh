#!/usr/bin/env bash
# Poll GitHub Actions every 10s for a new CI build of `ginger` and
# install it as soon as it appears, then exit. Invoked by
# `ginger-pull.service`, which `make deploy` (i.e. `scripts/deploy.sh`)
# kicks off after a push to main.
#
# Bounded by INTERVAL × MAX_ITERS — CI normally finishes in ~3 min, so
# 15 min (90 × 10s) is a comfortable upper bound that also catches a
# slow / re-run build without polling forever.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${GINGER_BURST_INTERVAL:-10}"
MAX_ITERS="${GINGER_BURST_MAX_ITERS:-90}"

log() { printf '%s [burst] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

log "starting burst (interval=${INTERVAL}s, max=${MAX_ITERS} iters)"

for i in $(seq 1 "$MAX_ITERS"); do
  "$SCRIPT_DIR/pull-binary.sh"
  rc=$?
  case "$rc" in
    0)
      log "new binary installed on iteration $i — exiting"
      exit 0
      ;;
    10)
      # No new artifact yet; keep waiting.
      ;;
    11)
      log "no GitHub token configured — aborting burst"
      exit 11
      ;;
    *)
      log "pull-binary.sh failed (rc=$rc) — aborting burst"
      exit "$rc"
      ;;
  esac
  sleep "$INTERVAL"
done

log "timed out after $((INTERVAL * MAX_ITERS))s without a new build"
exit 1
