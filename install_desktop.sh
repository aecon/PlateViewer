#!/bin/bash
# ---------------------------------------------------------------------------
# Creates a PlateViewer.desktop shortcut on the user's Desktop.
#
# Run this script from the repo directory:
#   bash install_desktop.sh
#
# It uses the current repo location to set the Exec path, copies the
# shortcut to ~/Desktop/, and marks it executable + trusted so that
# Ubuntu allows double-click launching without prompts.
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_FILE="$REPO_DIR/PlateViewer.desktop"
TARGET="$HOME/Desktop/PlateViewer.desktop"

# Generate the .desktop file in the repo
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=PlateViewer
Comment=QC visualization for high-content screening
Exec=$REPO_DIR/launch.sh
Terminal=false
Categories=Science;
EOF

# Copy to Desktop
cp "$DESKTOP_FILE" "$TARGET"

# Make executable and trusted
chmod +x "$REPO_DIR/launch.sh"
chmod +x "$TARGET"
gio set "$TARGET" metadata::trusted true 2>/dev/null

echo "Desktop shortcut installed at $TARGET"
