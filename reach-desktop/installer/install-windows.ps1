# Reach Desktop Installer — Windows
# Run this in PowerShell as Administrator for full GPU setup
# Or run normally for CPU-only mode

param(
    [switch]$CPUOnly = $false
)

$ErrorActionPreference = "Stop"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Reach Desktop Installer (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$REACH_DIR = "$env:LOCALAPPDATA\Reach"
$MODELS_DIR = "$REACH_DIR\models"

# Create directories
New-Item -ItemType Directory -Force -Path $REACH_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $MODELS_DIR | Out-Null

# ── Step 1: Check Python ──────────────────────────────────────
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    Write-Host "  Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  Python not found. Please install Python 3.12+ from https://python.org" -ForegroundColor Red
    Write-Host "  After installing, re-run this script." -ForegroundColor Red
    pause
    exit 1
}

# ── Step 2: GPU Detection ─────────────────────────────────────
$GPU_BACKEND = "cpu"
if (-not $CPUOnly) {
    Write-Host "[2/5] Detecting GPU..." -ForegroundColor Yellow
    try {
        $nvidia = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  NVIDIA GPU detected" -ForegroundColor Green
            $GPU_BACKEND = "cuda"
            Write-Host "  Checking CUDA toolkit..." -ForegroundColor Yellow
            try {
                nvcc --version 2>&1 | Out-Null
                Write-Host "  CUDA toolkit found" -ForegroundColor Green
            } catch {
                Write-Host "  CUDA toolkit not found. Install from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
                Write-Host "  Reach will run in CPU mode until CUDA is installed." -ForegroundColor Yellow
                $GPU_BACKEND = "cpu"
            }
        }
    } catch {
        # Try AMD
        try {
            rocm-smi 2>&1 | Out-Null
            Write-Host "  AMD GPU detected (ROCm)" -ForegroundColor Green
            $GPU_BACKEND = "rocm"
        } catch {
            # Try Vulkan / Intel
            Write-Host "  No NVIDIA/AMD GPU detected. Checking Vulkan..." -ForegroundColor Yellow
            try {
                vulkaninfo --summary 2>&1 | Out-Null
                Write-Host "  Vulkan-capable GPU found (Intel or integrated)" -ForegroundColor Green
                $GPU_BACKEND = "vulkan"
            } catch {
                Write-Host "  No GPU detected. Running in CPU mode." -ForegroundColor Yellow
            }
        }
    }
}
Write-Host "  Selected backend: $GPU_BACKEND" -ForegroundColor Cyan

# ── Step 3: Install Reach ─────────────────────────────────────
Write-Host "[3/5] Installing Reach..." -ForegroundColor Yellow
pip install reach-core --quiet 2>&1 | Out-Null
Write-Host "  reach-core installed" -ForegroundColor Green

# ── Step 4: Download Models ───────────────────────────────────
Write-Host "[4/5] Downloading models (~12 GB)..." -ForegroundColor Yellow
Write-Host "  This may take a while. Models are cached at: $MODELS_DIR" -ForegroundColor Cyan
$env:REACH_MODELS_DIR = $MODELS_DIR
python -c "from reach.models.download import download_all; download_all()"
Write-Host "  Models ready" -ForegroundColor Green

# ── Step 5: Create Desktop Shortcut ───────────────────────────
Write-Host "[5/5] Creating shortcut..." -ForegroundColor Yellow
$launcher = @"
@echo off
title Reach PDF Remediation
set REACH_MODELS_DIR=$MODELS_DIR
echo Starting Reach on http://localhost:8501
start http://localhost:8501
python -m uvicorn reach-server.main:app --host 127.0.0.1 --port 8501
pause
"@
$launcherPath = "$REACH_DIR\reach.bat"
$launcher | Out-File -FilePath $launcherPath -Encoding ASCII

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$desktop\Reach.lnk")
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $REACH_DIR
$shortcut.IconLocation = "$REACH_DIR\reach.ico"
$shortcut.Save()

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Reach Desktop installed successfully!" -ForegroundColor Green
Write-Host "  GPU backend: $GPU_BACKEND" -ForegroundColor Green
Write-Host "  Launch from desktop shortcut: Reach" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
pause
