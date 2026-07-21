"""Tests for GPU backend detection."""
import sys

import pytest


class TestGPUDetection:
    def test_detect_gpu_returns_gpuinfo(self):
        from reach.models.backend import GPUInfo, detect_gpu

        gpu = detect_gpu()
        assert isinstance(gpu, GPUInfo)
        assert gpu.backend in ("cuda", "rocm", "vulkan", "metal", "cpu")
        assert isinstance(gpu.vram_mb, int)

    def test_cpu_backend_has_zero_vram(self, monkeypatch):
        """When no GPU is present, we should get CPU backend."""
        from reach.models.backend import GPUInfo, detect_gpu

        def _mock_none(*args, **kwargs):
            return None

        # Mock all detection functions to return None
        monkeypatch.setattr("reach.models.backend._detect_cuda", _mock_none)
        monkeypatch.setattr("reach.models.backend._detect_rocm", _mock_none)
        monkeypatch.setattr("reach.models.backend._detect_vulkan", _mock_none)

        if sys.platform == "darwin":
            monkeypatch.setattr("reach.models.backend._detect_metal", _mock_none)

        gpu = detect_gpu()
        assert gpu.backend == "cpu"
        assert gpu.vram_mb == 0
        assert not gpu.is_gpu

    def test_gpuinfo_string_representation(self):
        from reach.models.backend import GPUInfo

        gpu = GPUInfo(backend="cuda", vendor="NVIDIA", name="RTX 4080", vram_mb=12288)
        assert "CUDA" in str(gpu)
        assert "RTX 4080" in str(gpu)
        assert "12.0 GB" in str(gpu)

        cpu = GPUInfo(backend="cpu", vendor="Unknown", name="CPU", vram_mb=0)
        assert "CPU" in str(cpu)

    def test_get_llamacpp_backend(self):
        from reach.models.backend import GPUInfo, get_llamacpp_backend

        assert get_llamacpp_backend(GPUInfo("cuda", "NVIDIA", "", 0)) == "cuda"
        assert get_llamacpp_backend(GPUInfo("rocm", "AMD", "", 0)) == "rocm"
        assert get_llamacpp_backend(GPUInfo("vulkan", "Intel", "", 0)) == "vulkan"
        assert get_llamacpp_backend(GPUInfo("metal", "Apple", "", 0)) == "metal"
        assert get_llamacpp_backend(GPUInfo("cpu", "", "", 0)) == "cpu"
