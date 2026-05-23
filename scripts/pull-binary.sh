#!/usr/bin/env bash
# Fetch the latest CI-built `ginger` binary from GitHub Actions and install
# it, so the Raspberry Pi runs a ready-made build instead of compiling
# locally.
#
# The binary is produced by .github/workflows/rpi-build.yml on every push
# to `main`. GitHub Actions artifacts require authentication even for a
# public repo, so this needs a GitHub token with read access to Actions:
#   * a fine-grained PAT scoped to hmeyer/ginger with `Actions: read`, or
#   * a classic token with the `repo` scope.
# Provide it via the GINGER_GH_TOKEN env var, or write it (chmod 600) to
# ~/.config/ginger/gh-token, or — easiest on a dev Pi — just be logged
# in with the `gh` CLI (`gh auth login`); this script will fall through
# to `gh auth token` automatically.
#
# Installing the binary triggers the ginger-watch.path unit, which
# restarts the service. Invoked in a 10s loop by scripts/pull-burst.sh
# (which `make deploy` kicks off after a `git push`), or by hand.
#
# Exit codes:
#   0   installed a new binary
#   10  no new artifact yet (idempotent no-op — burst loop should retry)
#   11  no token / not configured (burst loop should give up)
#   1   any other error (handled by `set -e` / die)
set -euo pipefail

REPO="hmeyer/ginger"
WORKFLOW="rpi-build.yml"
ARTIFACT="ginger-aarch64"
BRANCH="main"

DEST="$HOME/ginger/target/release/ginger"
STATE_DIR="$HOME/.config/ginger"
STATE_FILE="$STATE_DIR/last-deploy"
TOKEN_FILE="$STATE_DIR/gh-token"
API="https://api.github.com"

log()  { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

for cmd in curl jq unzip sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 \
    || die "missing required command: $cmd  (install with: sudo apt install jq unzip)"
done

# Resolve the GitHub token. Order: env var, then $TOKEN_FILE, then the
# logged-in `gh` CLI (if installed) — reusing the existing developer
# login avoids managing a second PAT. A missing token is a no-op
# (exit 0), not a failure, so the polling timer does not spam systemd
# with failed units before any token has been configured.
TOKEN="${GINGER_GH_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -r "$TOKEN_FILE" ]; then
  TOKEN="$(tr -d ' \t\r\n' < "$TOKEN_FILE")"
fi
if [ -z "$TOKEN" ] && command -v gh >/dev/null 2>&1; then
  TOKEN="$(gh auth token 2>/dev/null | tr -d ' \t\r\n' || true)"
fi
if [ -z "$TOKEN" ]; then
  log "no GitHub token found — set GINGER_GH_TOKEN, write $TOKEN_FILE (chmod 600), or run 'gh auth login'."
  exit 11
fi

gh_api() {
  curl -fsSL --retry 3 --retry-delay 2 \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$@"
}

log "looking up latest successful '$WORKFLOW' run on '$BRANCH'"
run_id="$(gh_api "$API/repos/$REPO/actions/workflows/$WORKFLOW/runs?branch=$BRANCH&status=success&per_page=1" \
  | jq -r '.workflow_runs[0].id // empty')"
[ -n "$run_id" ] || die "no successful '$WORKFLOW' run found on '$BRANCH'"

artifact_id="$(gh_api "$API/repos/$REPO/actions/runs/$run_id/artifacts" \
  | jq -r --arg n "$ARTIFACT" 'first(.artifacts[] | select(.name==$n) | .id) // empty')"
[ -n "$artifact_id" ] || die "run $run_id has no '$ARTIFACT' artifact (expired after 90 days?)"

if [ -f "$STATE_FILE" ] && [ "$(cat "$STATE_FILE")" = "$artifact_id" ]; then
  log "artifact $artifact_id already deployed — up to date"
  exit 10
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

log "downloading artifact $artifact_id (run $run_id)"
gh_api -o "$tmp/artifact.zip" "$API/repos/$REPO/actions/artifacts/$artifact_id/zip"
unzip -q "$tmp/artifact.zip" -d "$tmp"

[ -f "$tmp/ginger" ]        || die "artifact did not contain a 'ginger' binary"
[ -f "$tmp/ginger.sha256" ] || die "artifact did not contain 'ginger.sha256'"

log "verifying checksum"
( cd "$tmp" && sha256sum -c ginger.sha256 ) >/dev/null \
  || die "checksum verification failed — refusing to install"

# Atomic install: stage the new binary on the destination filesystem, then
# rename it into place. A same-filesystem rename is atomic, so the running
# service never execs a half-written file. The ginger-watch.path unit then
# restarts the service.
mkdir -p "$(dirname "$DEST")"
cp "$tmp/ginger" "$DEST.new"
chmod +x "$DEST.new"
mv -f "$DEST.new" "$DEST"

mkdir -p "$STATE_DIR"
printf '%s\n' "$artifact_id" > "$STATE_FILE"
log "installed new ginger binary (artifact $artifact_id) -> $DEST"
