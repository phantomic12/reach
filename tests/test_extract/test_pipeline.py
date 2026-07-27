"""
Tests for reach.extract — extraction orchestrator and pipeline.

Tests the full parse→layout→OCR flow using mocks for all models.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from reach.extract import (
    ExtractedDocument,
    ExtractionPipeline,
    extract_document,
)
from reach.extract.layout import SuryaBlock, SuryaLayoutModel
from reach.extract.models import (
    BBox,
    BlockType,
    DocumentMetadata,
    Page,
    PageMetadata,
    TextBlock,
)
from reach.extract.ocr import UnlimitedOCREngine
from reach.extract.parser import ParsedDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_pdf_path(tmp_path_factory):
    """Create a real PDF for integration tests."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Integration Test", "/Author": "Reach Tests"})

    tmp = tmp_path_factory.mktemp("pdftests") / "test.pdf"
    with open(tmp, "wb") as f:
        writer.write(f)
    return tmp


@pytest.fixture
def mock_surya():
    """A Surya model that returns fake layout blocks."""
    model = SuryaLayoutModel(backend="cpu")

    # Make it appear available
    model._available = True
    model._model = MagicMock()

    # DocLayout-YOLO path: model._model.detect(images) -> list of pages,
    # each page is an iterable of dicts with label/bbox/confidence keys.
    def fake_detect(images, *args, **kwargs):
        results = []
        for _ in images:
            page = []
            for i, (label, bbox, order) in enumerate(
                [
                    ("Section-header", [50, 50, 550, 80], 0),
                    ("Plain text", [50, 100, 550, 400], 1),
                    ("Page-footer", [200, 750, 400, 780], 2),
                ]
            ):
                page.append({
                    "label": label,
                    "bbox": bbox,
                    "confidence": 0.9 - i * 0.05,
                    "reading_order": order,
                })
            results.append(page)
        return results

    model._model.detect.side_effect = fake_detect
    return model


@pytest.fixture
def mock_ocr():
    """An OCR engine that returns recognizable text via the new infer() API."""
    engine = UnlimitedOCREngine(backend="cpu")
    engine._available = True
    engine._model = MagicMock()
    engine._tokenizer = MagicMock()

    def fake_infer(tokenizer, prompt="", image_file="", output_path="", **_kwargs):
        from pathlib import Path
        out = Path(output_path) / "result.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("Mocked OCR text for region.", encoding="utf-8")

    engine._model.infer.side_effect = fake_infer
    return engine


class _FakeTorchNoGrad:
    """Mock for torch.no_grad() context manager."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


_real_import = __import__


def _mock_import_for_ocr(name, *args, **kwargs):
    """Mock __import__ to handle 'torch' gracefully during OCR tests."""
    if name == "torch":
        fake = MagicMock()
        fake.no_grad = _FakeTorchNoGrad
        return fake
    return _real_import(name, *args, **kwargs)


@pytest.fixture(autouse=True)
def _mock_torch_import(monkeypatch):
    """Ensure __import__('torch') returns a mock for all pipeline tests."""
    monkeypatch.setattr("builtins.__import__", _mock_import_for_ocr)


# ---------------------------------------------------------------------------
# ExtractionPipeline
# ---------------------------------------------------------------------------


class TestExtractionPipeline:
    def test_init_no_models(self):
        """Pipeline can be created without loading models."""
        pipeline = ExtractionPipeline(load_models=False)
        assert pipeline.surya is None
        assert pipeline.ocr is None

    def test_set_models(self):
        pipeline = ExtractionPipeline(load_models=False)
        pipeline.surya = MagicMock(spec=SuryaLayoutModel)
        pipeline.ocr = MagicMock(spec=UnlimitedOCREngine)
        assert pipeline.surya is not None
        assert pipeline.ocr is not None

    def test_run_full_pipeline(self, real_pdf_path, mock_surya, mock_ocr):
        """End-to-end: parse → layout → OCR → ExtractedDocument."""
        pipeline = ExtractionPipeline(load_models=False)
        pipeline.surya = mock_surya
        pipeline.ocr = mock_ocr

        doc = pipeline.run(real_pdf_path)
        assert isinstance(doc, ExtractedDocument)
        assert doc.metadata.title == "Integration Test"
        assert len(doc.pages) == 1
        assert doc.pages[0].metadata.page_number == 1

    def test_run_output_blocks(self, real_pdf_path, mock_surya, mock_ocr):
        """Verify that blocks are populated correctly."""
        pipeline = ExtractionPipeline(load_models=False)
        pipeline.surya = mock_surya
        pipeline.ocr = mock_ocr

        doc = pipeline.run(real_pdf_path)
        page = doc.pages[0]
        # Should have 3 blocks (from mock_surya)
        assert len(page.blocks) == 3

        # Verify block types
        assert page.blocks[0].block_type == BlockType.SECTION_HEADER
        assert page.blocks[1].block_type == BlockType.TEXT
        assert page.blocks[2].block_type == BlockType.PAGE_FOOTER

        # Text should be populated by OCR
        assert page.blocks[1].text

    def test_run_without_models(self, real_pdf_path):
        """Run without any models: layout blocks empty, but parse still works."""
        pipeline = ExtractionPipeline(load_models=False)

        doc = pipeline.run(real_pdf_path)
        assert isinstance(doc, ExtractedDocument)
        assert len(doc.pages) == 1
        # Without models, blocks are empty
        assert doc.pages[0].blocks == []

    def test_unload(self):
        pipeline = ExtractionPipeline(load_models=False)
        mock_s = MagicMock(spec=SuryaLayoutModel)
        mock_o = MagicMock(spec=UnlimitedOCREngine)
        pipeline.surya = mock_s
        pipeline.ocr = mock_o

        pipeline.unload()
        mock_s.unload.assert_called_once()
        mock_o.unload.assert_called_once()


# ---------------------------------------------------------------------------
# extract_document convenience function
# ---------------------------------------------------------------------------


class TestExtractDocument:
    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            extract_document("/nonexistent/file.pdf", load_models=False)

    def test_with_output_path(self, real_pdf_path, mock_surya, mock_ocr, tmp_path):
        """Write structured JSON output."""
        pipeline = ExtractionPipeline(load_models=False)
        pipeline.surya = mock_surya
        pipeline.ocr = mock_ocr

        output = tmp_path / "result.json"
        doc = extract_document(real_pdf_path, output_path=output, pipeline=pipeline)

        assert output.exists()
        with open(output) as f:
            data = json.load(f)

        assert "pages" in data
        assert data["metadata"]["title"] == "Integration Test"

    def test_json_output_valid(self, real_pdf_path, mock_surya, mock_ocr, tmp_path):
        """Output JSON should be parseable and validated."""
        pipeline = ExtractionPipeline(load_models=False)
        pipeline.surya = mock_surya
        pipeline.ocr = mock_ocr

        output = tmp_path / "output.json"
        doc = extract_document(real_pdf_path, output_path=output, pipeline=pipeline)

        # Reload and validate
        loaded = ExtractedDocument.from_json(output)
        assert loaded.metadata.title == doc.metadata.title
        assert len(loaded.pages) == len(doc.pages)

    def test_write_json_method(self, tmp_path):
        """Test ExtractedDocument.write_json directly."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="test")
        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        page = Page(metadata=meta, blocks=[block])
        doc = ExtractedDocument(pages=[page])

        output = tmp_path / "direct_output.json"
        doc.write_json(output)
        assert output.exists()

        loaded = ExtractedDocument.from_json(output)
        assert loaded.pages[0].blocks[0].text == "test"

    def test_from_json_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ExtractedDocument.from_json("/nonexistent/file.json")


# ---------------------------------------------------------------------------
# Scanned vs digital page classification in pipeline
# ---------------------------------------------------------------------------


class TestPageClassification:
    def test_scanned_pages_get_ocr(self, real_pdf_path, mock_surya, mock_ocr):
        """When page is classified as scanned, full-page OCR should run."""
        pipeline = ExtractionPipeline(load_models=False)
        pipeline.surya = mock_surya
        pipeline.ocr = mock_ocr

        doc = pipeline.run(real_pdf_path)
        page = doc.pages[0]
        # Blank pages from pypdf have no text → classified as scanned
        assert page.metadata.is_scanned is True
