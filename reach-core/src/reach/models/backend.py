"""
GPU backend detection and selection.

Supported backends in priority order:
    CUDA (NVIDIA) → ROCm (AMD) → Vulkan (Intel/AMD) → Metal (Apple) → CPU

Usage:
    from reach.models.backend import detect_gpu, GPUInfo

    gpu = detect_gpu()
    print(gpu)  # "CUDA (NVIDIA RTX 4080, 12.0 GB VRAM)"
"""
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a detected GPU."""

    backend: str  # "cuda", "rocm", "vulkan", "metal", "cpu"
    vendor: str  # "NVIDIA", "AMD", "Intel", "Apple", "Unknown"
    name: str  # "RTX 4080 Laptop GPU", "AMD Radeon RX 7900", etc.
    vram_mb: int  # Available VRAM in MB, 0 for CPU/shared
    vulkan_support: bool = False

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"

    def __str__(self) -> str:
        if not self.is_gpu:
            return "CPU (no GPU detected)"
        return f"{self.backend.upper()} ({self.vendor} {self.name}, {self.vram_gb:.1f} GB VRAM)"


def detect_gpu() -> GPUInfo:
    """Detect the best available GPU and return its info.

    Priority: CUDA > ROCm > Vulkan > Metal > CPU
    """
    # Try CUDA (NVIDIA)
    cuda = _detect_cuda()
    if cuda:
        return cuda

    # Try ROCm (AMD)
    rocm = _detect_rocm()
    if rocm:
        return rocm

    # Try Vulkan (cross-vendor)
    vulkan = _detect_vulkan()
    if vulkan:
        return vulkan

    # Try Metal (Apple Silicon)
    if sys.platform == "darwin":
        metal = _detect_metal()
        if metal:
            return metal

    # Fallback to CPU
    log.warning("No GPU detected — running in CPU mode (slow)")
    return GPUInfo(backend="cpu", vendor="Unknown", name="CPU", vram_mb=0)


def _detect_cuda() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU via nvidia-smi or torch.cuda."""
    # Try nvidia-smi first (most reliable)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, vram = result.stdout.strip().split(",", 1)
            return GPUInfo(
                backend="cuda",
                vendor="NVIDIA",
                name=name.strip(),
                vram_mb=int(vram.strip()),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Fall back to torch.cuda
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            return GPUInfo(backend="cuda", vendor="NVIDIA", name=name, vram_mb=vram)
    except ImportError:
        pass

    return None


def _detect_rocm() -> Optional[GPUInfo]:
    """Detect AMD GPU via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Header + data
                parts = lines[1].split(",")
                if len(parts) >= 2:
                    name = parts[1].strip().strip('"')
                    vram = int(float(parts[2].strip().strip('"')) % 1024) if len(parts) > 2 else 8192
                    return GPUInfo(backend="rocm", vendor="AMD", name=name, vram_mb=vram)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def _detect_vulkan() -> Optional[GPUInfo]:
    """Detect GPU via Vulkan (cross-vendor fallback)."""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "deviceName" in line:
                    name = line.split("=")[-1].strip().strip('"')
                    # Determine vendor from name
                    vendor = "Unknown"
                    if any(x in name.lower() for x in ("nvidia", "rtx", "gtx", "quadro")):
                        vendor = "NVIDIA"
                    elif any(x in name.lower() for x in ("amd", "radeon", "rx")):
                        vendor = "AMD"
                    elif any(x in name.lower() for x in ("intel", "arc", "iris", "uhd")):
                        vendor = "Intel"
                    vram = 4096  # Conservative default for Vulkan
                    return GPUInfo(
                        backend="vulkan",
                        vendor=vendor,
                        name=name,
                        vram_mb=vram,
                        vulkan_support=True,
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_metal() -> Optional[GPUInfo]:
    """Detect Apple Silicon GPU."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            cpu = result.stdout.strip()
            # Apple Silicon unified memory — approximate from total RAM
            mem_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            mem_mb = 0
            if mem_result.returncode == 0:
                mem_mb = int(mem_result.stdout.strip()) // (1024 * 1024)
            return GPUInfo(
                backend="metal",
                vendor="Apple",
                name=f"Apple Silicon ({cpu})",
                vram_mb=mem_mb,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_llamacpp_backend(gpu: GPUInfo) -> str:
    """Return the llama.cpp backend flag for the detected GPU."""
    backend_map = {
        "cuda": "cuda",
        "rocm": "rocm",
        "vulkan": "vulkan",
        "metal": "metal",
        "cpu": "cpu",
    }
    return backend_map.get(gpu.backend, "cpu")
