#!/usr/bin/env bash
# Push to GitHub and kick the Pi-side burst pull. After the push, CI
# (`.github/workflows/rpi-build.yml`) builds the aarch64 binary in
# ~3 min; the burst service polls every 10s and installs it as soon
# as the artifact is up.
#
# Forwards extra args to `git push` so this stays a drop-in replacement:
#   scripts/deploy.sh
#   scripts/deploy.sh origin main
#   scripts/deploy.sh --force-with-lease
#
# Only triggers the burst when the local branch is main, since
# rpi-build.yml only builds main.
set -euo pipefail

branch="$(git rev-parse --abbrev-ref HEAD)"

git push "$@"

if [ "$branch" != "main" ]; then
  printf 'deploy: on branch %q, not main — skipping burst trigger\n' "$branch"
  exit 0
fi

# `restart --no-block` so back-to-back deploys reset the 15-min burst
# window and the script returns immediately; the burst runs detached
# under systemd. Watch with `journalctl --user -u ginger-pull -f`.
systemctl --user restart --no-block ginger-pull.service
echo "deploy: triggered ginger-pull.service (watch: journalctl --user -u ginger-pull -f)"
