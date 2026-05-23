#!/usr/bin/env bash
# Install ginger as a systemd user service.
# The service starts automatically at boot and restarts whenever
# target/release/ginger is replaced by a new build.
set -euo pipefail

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

# ── ginger.service ────────────────────────────────────────────────────────────
cat > "$UNIT_DIR/ginger.service" << 'EOF'
[Unit]
Description=Ginger robot server
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/ginger
ExecStart=%h/ginger/target/release/ginger
Restart=on-failure
RestartSec=3
Environment=RUST_LOG=info
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

# ── ginger-watch.path — watches the binary for changes ───────────────────────
cat > "$UNIT_DIR/ginger-watch.path" << 'EOF'
[Unit]
Description=Watch Ginger binary for updates

[Path]
PathChanged=%h/ginger/target/release/ginger

[Install]
WantedBy=default.target
EOF

# ── ginger-watch.service — triggered by the path unit ────────────────────────
cat > "$UNIT_DIR/ginger-watch.service" << 'EOF'
[Unit]
Description=Restart Ginger after binary update

[Service]
Type=oneshot
ExecStart=systemctl --user restart ginger.service
EOF

# ── ginger-pull.service — burst-poll GitHub Actions for a new build ─────────
# Started by `make deploy` (scripts/deploy.sh) after a push to main.
# Polls every 10s for up to 15 min, installs the new binary as soon as
# CI finishes, then exits. Idle (no recurring timer) between deploys.
cat > "$UNIT_DIR/ginger-pull.service" << 'EOF'
[Unit]
Description=Burst-poll GitHub Actions for a new ginger build

[Service]
Type=oneshot
ExecStart=%h/ginger/scripts/pull-burst.sh
EOF

# Retire any previous polling timer from earlier installs of this script.
systemctl --user disable --now ginger-pull.timer 2>/dev/null || true
rm -f "$UNIT_DIR/ginger-pull.timer"

systemctl --user daemon-reload
systemctl --user enable --now ginger.service
systemctl --user enable --now ginger-watch.path

# Allow user services to run at boot without an active login session.
loginctl enable-linger "$USER"

TOKEN_FILE="$HOME/.config/ginger/gh-token"
token_ok=0
if [ -n "${GINGER_GH_TOKEN:-}" ] \
    || [ -r "$TOKEN_FILE" ] \
    || { command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; }; then
  token_ok=1
fi

echo ""
echo "Installed. Ginger will:"
echo "  • start automatically at boot"
echo "  • restart within seconds whenever target/release/ginger is rebuilt"
echo "  • pull a fresh CI build whenever you run 'make deploy'"
if [ "$token_ok" -ne 1 ]; then
  echo ""
  echo "Heads up: no GitHub token detected. 'make deploy' will push fine,"
  echo "but the burst service will exit with rc=11 until you either:"
  echo "  • run 'gh auth login'  (pull-binary.sh will reuse that token), or"
  echo "  • write a PAT with 'Actions: read' to ~/.config/ginger/gh-token (chmod 600)"
fi
echo ""
systemctl --user status ginger.service --no-pager || true
