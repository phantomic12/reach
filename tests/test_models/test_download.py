"""Tests for model download and caching."""
import os
from pathlib import Path

import pytest


class TestModelDownload:
    def test_manifest_has_all_models(self):
        from reach.models.download import MODEL_MANIFEST

        assert "unlimited-ocr" in MODEL_MANIFEST
        assert "surya" in MODEL_MANIFEST
        assert "qwen3.5-9b" in MODEL_MANIFEST

    def test_manifest_has_licenses(self):
        from reach.models.download import MODEL_MANIFEST

        for model_id, info in MODEL_MANIFEST.items():
            assert "license" in info, f"{model_id} missing license"
            assert "size_gb" in info, f"{model_id} missing size"

    def test_verify_model_missing(self):
        from reach.models.download import verify_model

        assert not verify_model("nonexistent-model-id")

    def test_models_dir_default(self):
        from reach.models.download import MODELS_DIR

        assert MODELS_DIR == Path.home() / ".reach" / "models"

    def test_models_dir_env_override(self, monkeypatch):
        custom = Path("/tmp/test-reach-models")
        monkeypatch.setenv("REACH_MODELS_DIR", str(custom))

        # Re-import to pick up env var
        from importlib import reload
        import reach.models.download as mod

        reload(mod)
        assert mod.MODELS_DIR == custom
