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

# ── ginger-pull.service — fetch the latest CI-built binary ───────────────────
cat > "$UNIT_DIR/ginger-pull.service" << 'EOF'
[Unit]
Description=Fetch the latest ginger binary from GitHub Actions

[Service]
Type=oneshot
ExecStart=%h/ginger/scripts/pull-binary.sh
EOF

# ── ginger-pull.timer — poll GitHub Actions for new builds ───────────────────
cat > "$UNIT_DIR/ginger-pull.timer" << 'EOF'
[Unit]
Description=Poll GitHub Actions for new ginger builds

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now ginger.service
systemctl --user enable --now ginger-watch.path

# The pull timer needs a GitHub token (Actions: read) to download
# artifacts. Enable it only once the token is configured, so it does not
# poll-and-fail before then.
TOKEN_FILE="$HOME/.config/ginger/gh-token"
pull_enabled=0
if [ -n "${GINGER_GH_TOKEN:-}" ] || [ -r "$TOKEN_FILE" ]; then
  systemctl --user enable --now ginger-pull.timer
  pull_enabled=1
fi

# Allow user services to run at boot without an active login session.
loginctl enable-linger "$USER"

echo ""
echo "Installed. Ginger will:"
echo "  • start automatically at boot"
echo "  • restart within seconds whenever target/release/ginger is rebuilt"
if [ "$pull_enabled" -eq 1 ]; then
  echo "  • poll GitHub Actions every 5 min and self-update to new CI builds"
else
  echo ""
  echo "Auto-update timer installed but NOT started — no GitHub token found."
  echo "Create a token with 'Actions: read' access to hmeyer/ginger, then:"
  echo "  mkdir -p ~/.config/ginger"
  echo "  install -m600 /dev/stdin ~/.config/ginger/gh-token   # paste token, Ctrl-D"
  echo "  systemctl --user enable --now ginger-pull.timer"
fi
echo ""
systemctl --user status ginger.service --no-pager || true
