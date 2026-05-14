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

systemctl --user daemon-reload
systemctl --user enable --now ginger.service
systemctl --user enable --now ginger-watch.path

# Allow user services to run at boot without an active login session.
loginctl enable-linger "$USER"

echo ""
echo "Installed. Ginger will:"
echo "  • start automatically at boot"
echo "  • restart within seconds whenever target/release/ginger is rebuilt"
echo ""
systemctl --user status ginger.service --no-pager || true
