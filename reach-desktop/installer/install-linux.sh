#!/usr/bin/env bash
# Reach Desktop Installer — Linux
# Run: chmod +x install-linux.sh && ./install-linux.sh
# Or for CPU-only: ./install-linux.sh --cpu

set -euo pipefail

CPU_ONLY=false
if [[ "${1:-}" == "--cpu" ]]; then
    CPU_ONLY=true
fi

REACH_DIR="$HOME/.local/share/reach"
MODELS_DIR="$REACH_DIR/models"

echo "========================================"
echo "  Reach Desktop Installer (Linux)"
echo "========================================"
echo ""

mkdir -p "$REACH_DIR" "$MODELS_DIR"

# ── Step 1: Check Python ──────────────────────────────────────
echo "[1/5] Checking Python..."
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version)
    echo "  Found: $PY_VERSION"
else
    echo "  Python 3 not found. Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# ── Step 2: GPU Detection ─────────────────────────────────────
GPU_BACKEND="cpu"
if [[ "$CPU_ONLY" != true ]]; then
    echo "[2/5] Detecting GPU..."
    if command -v nvidia-smi &>/dev/null; then
        echo "  NVIDIA GPU detected"
        GPU_BACKEND="cuda"
        if ! command -v nvcc &>/dev/null; then
            echo "  CUDA toolkit not found. Install: sudo apt install nvidia-cuda-toolkit"
            echo "  Reach will run in CPU mode until CUDA is installed."
            GPU_BACKEND="cpu"
        fi
    elif command -v rocm-smi &>/dev/null; then
        echo "  AMD GPU detected"
        GPU_BACKEND="rocm"
    elif command -v vulkaninfo &>/dev/null; then
        echo "  Vulkan-capable GPU found"
        GPU_BACKEND="vulkan"
    else
        echo "  No GPU detected. Running in CPU mode."
    fi
fi
echo "  Selected backend: $GPU_BACKEND"

# ── Step 3: Install Reach ─────────────────────────────────────
echo "[3/5] Installing Reach..."
pip3 install --quiet reach-core
echo "  reach-core installed"

# ── Step 4: Download Models ───────────────────────────────────
echo "[4/5] Downloading models (~12 GB)..."
echo "  Models cached at: $MODELS_DIR"
export REACH_MODELS_DIR="$MODELS_DIR"
python3 -c "from reach.models.download import download_all; download_all()"
echo "  Models ready"

# ── Step 5: Create Desktop Entry ──────────────────────────────
echo "[5/5] Creating desktop entry..."
LAUNCHER="$REACH_DIR/reach.sh"
cat > "$LAUNCHER" << 'LAUNCHEREOF'
#!/usr/bin/env bash
export REACH_MODELS_DIR="$HOME/.local/share/reach/models"
echo "Starting Reach on http://localhost:8501"
xdg-open http://localhost:8501 2>/dev/null || true
python3 -m uvicorn reach-server.main:app --host 127.0.0.1 --port 8501
LAUNCHEREOF
chmod +x "$LAUNCHER"

# Create .desktop file
DESKTOP_FILE="$HOME/.local/share/applications/reach.desktop"
cat > "$DESKTOP_FILE" << DESKTOPEOF
[Desktop Entry]
Name=Reach
Comment=Automated PDF/UA-2 Remediation
Exec=$LAUNCHER
Terminal=false
Type=Application
Categories=Office;Accessibility;
DESKTOPEOF

echo ""
echo "========================================"
echo "  Reach Desktop installed successfully!"
echo "  GPU backend: $GPU_BACKEND"
echo "  Launch from application menu: Reach"
echo "========================================"
