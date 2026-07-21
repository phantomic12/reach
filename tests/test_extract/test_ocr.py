"""
Tests for reach.extract.ocr — Unlimited OCR integration.

All tests use mocks — no model download required.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from reach.extract.models import (
    BBox,
    BlockType,
    EquationBlock,
    TableBlock,
    TextBlock,
)
from reach.extract.ocr import (
    UnlimitedOCREngine,
    _split_table_row,
    extract_page_text_with_ocr,
    load_unlimited_ocr,
    ocr_layout_blocks,
    ocr_scanned_page,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image():
    return Image.new("RGB", (612, 792), color=(255, 255, 255))


@pytest.fixture
def text_blocks():
    """Sample layout blocks for a single page."""
    return [
        TextBlock(
            block_type=BlockType.SECTION_HEADER,
            bbox=BBox(x1=50, y1=50, x2=550, y2=80),
            reading_order=0,
            text="",
        ),
        TextBlock(
            block_type=BlockType.TEXT,
            bbox=BBox(x1=50, y1=100, x2=550, y2=300),
            reading_order=1,
            text="",
        ),
        TableBlock(
            bbox=BBox(x1=50, y1=320, x2=550, y2=500),
            reading_order=2,
        ),
        EquationBlock(
            bbox=BBox(x1=50, y1=520, x2=400, y2=560),
            reading_order=3,
        ),
    ]


def _make_mock_engine(
    *,
    available: bool = True,
    backend: str = "cpu",
    decode_return: list[str] | None = None,
) -> UnlimitedOCREngine:
    """Create a pre-configured OCR engine with mocked internals."""
    engine = UnlimitedOCREngine(backend=backend)
    engine._available = available
    engine._model = MagicMock()
    engine._processor = MagicMock()

    if decode_return is None:
        decode_return = ["Mocked OCR text"]

    engine._processor.batch_decode.return_value = decode_return
    engine._processor.return_value = {"pixel_values": MagicMock()}
    engine._model.generate.return_value = MagicMock()
    return engine


class _FakeTorchNoGrad:
    """Mock for torch.no_grad() context manager."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _make_mock_import():
    """Create a mock import function that handles 'torch' gracefully."""
    import builtins
    _real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "torch":
            fake = MagicMock()
            fake.no_grad = _FakeTorchNoGrad
            return fake
        return _real_import(name, *args, **kwargs)

    return _mock_import


# ---------------------------------------------------------------------------
# _split_table_row
# ---------------------------------------------------------------------------


class TestSplitTableRow:
    def test_tab_separated(self):
        cells = _split_table_row("Name\tAge\tCity")
        assert cells == ["Name", "Age", "City"]

    def test_multi_space_separated(self):
        cells = _split_table_row("Hello    World    Test")
        assert len(cells) == 3
        assert "Hello" in cells

    def test_single_space(self):
        cells = _split_table_row("One Two Three")
        assert cells == ["One", "Two", "Three"]

    def test_empty_string(self):
        assert _split_table_row("") == []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestLoadUnlimitedOCR:
    def test_load_when_not_installed(self):
        """When the model dir doesn't exist, engine loads gracefully."""
        with patch("reach.extract.ocr.DEFAULT_OCR_DIR", MagicMock()):
            engine = load_unlimited_ocr()
            assert not engine.is_available


# ---------------------------------------------------------------------------
# UnlimitedOCREngine
# ---------------------------------------------------------------------------


class TestUnlimitedOCREngine:
    def test_default_unavailable(self):
        engine = UnlimitedOCREngine(backend="cpu")
        engine.load()
        assert not engine.is_available

    def test_recognize_empty_when_unavailable(self, sample_image):
        engine = UnlimitedOCREngine(backend="cpu")
        engine.load()
        text = engine.recognize(sample_image)
        assert text == ""

    def test_recognize_with_mock(self, sample_image):
        """Verify the OCR flow works with a mocked model."""
        engine = _make_mock_engine(decode_return=["Hello from OCR"])
        mock_import = _make_mock_import()

        with patch("builtins.__import__", side_effect=mock_import):
            text = engine.recognize(sample_image)
            assert text == "Hello from OCR"

    def test_recognize_region(self, sample_image):
        engine = _make_mock_engine(decode_return=["Region text"])
        mock_import = _make_mock_import()

        with patch("builtins.__import__", side_effect=mock_import):
            text = engine.recognize_region(sample_image, (50, 50, 200, 100))
            assert text == "Region text"

    def test_extract_table(self, sample_image):
        engine = _make_mock_engine(
            decode_return=["Name\tAge\tCity\nAlice\t30\tNYC\nBob\t25\tSF"]
        )
        mock_import = _make_mock_import()

        with patch("builtins.__import__", side_effect=mock_import):
            table = engine.extract_table(sample_image, (50, 320, 550, 500))
            assert table.rows == 3
            assert table.columns == 3

    def test_extract_formula(self, sample_image):
        engine = _make_mock_engine(decode_return=[r"E = mc^2"])
        mock_import = _make_mock_import()

        with patch("builtins.__import__", side_effect=mock_import):
            latex = engine.extract_formula(sample_image, (50, 520, 400, 560))
            assert latex == r"E = mc^2"

    def test_unload(self):
        engine = UnlimitedOCREngine(backend="cpu")
        engine.load()
        engine.unload()
        assert engine._model is None


# ---------------------------------------------------------------------------
# High-level OCR functions
# ---------------------------------------------------------------------------


class TestOCRScannedPage:
    def test_returns_text(self, sample_image):
        engine = _make_mock_engine(decode_return=["Full page text"])

        mock_import = _make_mock_import()
        with patch("builtins.__import__", side_effect=mock_import):
            text = ocr_scanned_page(sample_image, engine)
            assert text == "Full page text"


class TestOCRLayoutBlocks:
    def test_ocr_all_block_types(self, sample_image, text_blocks):
        engine = _make_mock_engine(decode_return=["Block text"])

        mock_import = _make_mock_import()
        with patch("builtins.__import__", side_effect=mock_import):
            result = ocr_layout_blocks(sample_image, text_blocks, engine)

        assert result[0].text == "Block text"
        assert result[1].text == "Block text"
        assert len(result[2].cells) > 0
        assert result[3].latex == "Block text"

    def test_skip_blocks_with_existing_text(self, sample_image):
        """Blocks that already have text should NOT be OCR'd again."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        blocks = [
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=0, text="Already has text"),
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=1, text=""),
        ]
        engine = _make_mock_engine(decode_return=["New OCR text"])

        mock_import = _make_mock_import()
        with patch("builtins.__import__", side_effect=mock_import):
            result = ocr_layout_blocks(sample_image, blocks, engine)

        assert result[0].text == "Already has text"
        assert result[1].text == "New OCR text"


class TestExtractPageTextWithOCR:
    def test_scanned_page_full_ocr(self, sample_image):
        """Scanned page: full-page OCR → distribute across blocks."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        blocks = [
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=0),
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=1),
        ]
        engine = _make_mock_engine(
            decode_return=["Paragraph one.\n\nParagraph two.\n\nParagraph three."]
        )

        mock_import = _make_mock_import()
        with patch("builtins.__import__", side_effect=mock_import):
            result = extract_page_text_with_ocr(sample_image, blocks, engine, is_scanned=True)

        assert result[0].text
        assert result[1].text

    def test_digital_page_targeted_ocr(self, sample_image):
        """Digital-born page: only OCR blocks without text."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        blocks = [
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=0, text="From content stream"),
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=1, text=""),
        ]
        engine = _make_mock_engine(decode_return=["From OCR"])

        mock_import = _make_mock_import()
        with patch("builtins.__import__", side_effect=mock_import):
            result = extract_page_text_with_ocr(sample_image, blocks, engine, is_scanned=False)

        assert result[0].text == "From content stream"
        assert result[1].text == "From OCR"
