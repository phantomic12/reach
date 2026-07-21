"""
Reach test fixtures and configuration.
"""
import os
from pathlib import Path

import pytest

# Ensure tests run against the source tree, not an installed package
TEST_DIR = Path(__file__).parent
CORPUS_DIR = TEST_DIR / "corpus"
PROJECT_ROOT = TEST_DIR.parent


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    """Path to the test PDF corpus."""
    CORPUS_DIR.mkdir(exist_ok=True)
    return CORPUS_DIR


@pytest.fixture(scope="session")
def sample_pdf(corpus_dir: Path) -> Path:
    """Return path to a known-good test PDF, downloading if needed."""
    pdf = corpus_dir / "sample.pdf"
    if not pdf.exists():
        # Generate a minimal test PDF using pypdf
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_metadata({"/Title": "Reach Test PDF"})
        with open(pdf, "wb") as f:
            writer.write(f)
    return pdf


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """Set custom models directory for tests."""
    test_models = TEST_DIR / "test_models"
    os.environ["REACH_MODELS_DIR"] = str(test_models)
    return test_models


@pytest.fixture(autouse=True)
def mock_gpu_detection(monkeypatch):
    """By default, tests run in CPU mode (no GPU required)."""
    from reach.models.backend import GPUInfo

    def _mock_detect():
        return GPUInfo(backend="cpu", vendor="Test", name="Test CPU", vram_mb=0)

    monkeypatch.setattr("reach.models.backend.detect_gpu", _mock_detect)
