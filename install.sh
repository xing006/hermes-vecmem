#!/usr/bin/env bash
# Hermes VecMem — install script
# Copies plugin files into Hermes source tree and installs dependencies.
set -euo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/AppData/Local/hermes/hermes-agent}"
PLUGIN_SRC="$(cd "$(dirname "$0")" && pwd)/plugins/memory/vecmem"
PLUGIN_DEST="$HERMES_REPO/plugins/memory/vecmem"

echo "Installing vecmem plugin..."

# 1. Copy plugin files
if [ ! -d "$PLUGIN_SRC" ]; then
    echo "Error: plugin source not found at $PLUGIN_SRC"
    echo "Run this script from the hermes-vecmem project root."
    exit 1
fi

mkdir -p "$PLUGIN_DEST"
cp -r "$PLUGIN_SRC"/* "$PLUGIN_DEST/"
echo "✅ Plugin files copied to $PLUGIN_DEST"

# 2. Install Python dependencies
cd "$HERMES_REPO"
if command -v uv &> /dev/null; then
    uv pip install sqlite-vec httpx --python venv/Scripts/python.exe 2>/dev/null || true
    echo "✅ Dependencies installed via uv"
else
    pip install sqlite-vec httpx 2>/dev/null || true
    echo "✅ Dependencies installed via pip"
fi

# 3. Configure
echo ""
echo "========================================="
echo "  Installation complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Configure memory provider:"
echo "     hermes config set memory.provider vecmem"
echo "     hermes config set memory.vecmem.embed_mode api"
echo "     hermes config set memory.vecmem.api_base https://api.deepseek.com"
echo ""
echo "  2. Start a new Hermes session (/reset)"
echo ""
