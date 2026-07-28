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
    sequential_returns: list[str] | None = None,
) -> UnlimitedOCREngine:
    """Create a pre-configured OCR engine with mocked internals.

    Args:
        decode_return: Single value returned for every infer() call.
        sequential_returns: List of values returned in order for each
            successive infer() call. Use to test the gundam→base fallback:
            first call returns a placeholder, second call returns real text.
    """
    engine = UnlimitedOCREngine(backend=backend)
    engine._available = available
    engine._model = MagicMock()
    engine._tokenizer = MagicMock()

    if decode_return is None and sequential_returns is None:
        decode_return = ["Mocked OCR text"]

    call_iter = iter(sequential_returns or [])

    # New OCR path: model.infer() writes <output_path>/result.md; we intercept
    # that by making infer() create the file with the desired text.
    def fake_infer(tokenizer, prompt="", image_file="", output_path="", **_kwargs):
        from pathlib import Path

        if sequential_returns is not None:
            try:
                text = next(call_iter)
            except StopIteration:
                text = ""
        else:
            text = "\n".join(decode_return)  # type: ignore[arg-type]

        out = Path(output_path) / "result.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")

    engine._model.infer.side_effect = fake_infer
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


# ============================================================================
# recognize() — gundam→base fallback, image resize, artifact stripping
# ============================================================================


class TestRecognizeGundamFallback:
    """The Unlimited OCR model emits a placeholder ('one image, no text')
    when it can't detect any actual content in a single-image input. The
    recognize() method retries with the 'base' config in that case."""

    def test_gundam_returns_placeholder_retries_with_base(self):
        """First call returns the placeholder; second returns real text."""
        engine = _make_mock_engine(
            sequential_returns=[
                "<|det|>image [0, 0, 999, 999]<|/det|>",
                "Actual OCR text from base config",
            ]
        )
        result = engine.recognize(Image.new("RGB", (100, 100), "white"))
        assert result == "Actual OCR text from base config"
        # Both gundam and base configs were tried
        assert engine._model.infer.call_count == 2

    def test_gundam_returns_text_skips_base(self):
        """If gundam produces real text, don't retry."""
        engine = _make_mock_engine(decode_return=["First-call text"])
        result = engine.recognize(Image.new("RGB", (100, 100), "white"))
        assert result == "First-call text"
        assert engine._model.infer.call_count == 1

    def test_gundam_returns_placeholder_and_base_returns_placeholder(self):
        """Both configs return placeholders → empty result, no crash."""
        engine = _make_mock_engine(
            sequential_returns=[
                "<|det|>image [0, 0, 999, 999]<|/det|>",
                "<|det|>image [0, 0, 999, 999]<|/det|>",
            ]
        )
        result = engine.recognize(Image.new("RGB", (100, 100), "white"))
        assert result == ""
        assert engine._model.infer.call_count == 2


class TestRecognizeImageResize:
    """Unlimited OCR silently rejects oversized images. recognize() should
    auto-resize anything larger than 2048px on the longest side."""

    def _capture_image_size(self, engine):
        """Attach a side-effect to capture the image size on first call.

        We can't read the saved image after recognize() returns because
        recognize() cleans up its TemporaryDirectory in a `with` block. So
        the side-effect reads the image synchronously while the temp dir is
        still alive and stores just the (width, height) tuple.
        """
        captured = {}

        original_side_effect = engine._model.infer.side_effect

        def wrapping(*args, **kwargs):
            image_file = kwargs.get("image_file") or args[2]
            from PIL import Image as PILImage

            with PILImage.open(image_file) as img:
                captured["size"] = img.size
            return original_side_effect(*args, **kwargs)

        engine._model.infer.side_effect = wrapping
        return captured

    def test_oversized_image_resized_before_infer(self):
        engine = _make_mock_engine(decode_return=["resized text"])
        captured = self._capture_image_size(engine)

        # 3000x3000 image → should be resized to 2048x2048
        big = Image.new("RGB", (3000, 3000), "white")
        result = engine.recognize(big)
        assert result == "resized text"
        assert max(captured["size"]) <= 2048

    def test_small_image_not_resized(self):
        engine = _make_mock_engine(decode_return=["small text"])
        captured = self._capture_image_size(engine)

        small = Image.new("RGB", (800, 600), "white")
        engine.recognize(small)
        assert captured["size"] == (800, 600)


class TestStripOcrArtifacts:
    """Cleanup of Unlimited OCR placeholder tokens and duplicate lines."""

    def test_strips_detection_boxes(self):
        text = "<|det|>title [43, 109, 999, 202]<|/det|>Hello world"
        out = UnlimitedOCREngine._strip_ocr_artifacts(text)
        assert out == "Hello world"

    def test_strips_image_markdown(self):
        text = "Caption here\n![](images/0.jpg)\nMore caption"
        out = UnlimitedOCREngine._strip_ocr_artifacts(text)
        assert "![](/images/" not in out
        assert "Caption here" in out
        assert "More caption" in out

    def test_dedups_consecutive_duplicate_lines(self):
        # Simulates overlapping detection boxes that emit the same line twice
        text = "Line one\nLine two\nLine two\nLine three"
        out = UnlimitedOCREngine._strip_ocr_artifacts(text)
        assert out == "Line one\nLine two\nLine three"

    def test_preserves_intentional_repetition(self):
        # Same line in different paragraphs should NOT be deduplicated
        text = "Paragraph one\n\nRepeated line\n\nParagraph two\n\nRepeated line"
        out = UnlimitedOCREngine._strip_ocr_artifacts(text)
        assert out.count("Repeated line") == 2

    def test_collapse_excess_newlines(self):
        text = "Para one\n\n\n\n\nPara two"
        out = UnlimitedOCREngine._strip_ocr_artifacts(text)
        assert "\n\n\n" not in out
        assert "Para one" in out
        assert "Para two" in out

    def test_combined_artifacts_and_dedup(self):
        # Real-world example: model emits overlapping detection boxes
        raw = (
            "<|det|>text [43, 616, 714, 675]<|/det|>that should be detected\n"
            "<|det|>text [43, 625, 830, 785]<|/det|>that should be detected\n"
            "as separate regions by OCR.\n"
            "![](images/0.jpg)\n"
        )
        out = UnlimitedOCREngine._strip_ocr_artifacts(raw)
        assert out.count("that should be detected") == 1
        assert "as separate regions by OCR." in out
        assert "![](/images/" not in out


# ============================================================================
# extract_page_text_with_ocr — fallback when layout produces no blocks
# ============================================================================


class TestScannedPageLayoutFailureFallback:
    """When layout detection fails on a scanned page (no text blocks) but
    OCR recovers text, the function should wrap the recovered text in a
    single page-spanning TextBlock so downstream stages have content."""

    def test_ocr_text_with_no_blocks_creates_single_text_block(self):
        engine = _make_mock_engine(decode_return=["Recovered text from OCR"])
        # Empty blocks list — simulates layout failure
        result = extract_page_text_with_ocr(
            Image.new("RGB", (100, 100), "white"),
            blocks=[],
            engine=engine,
            is_scanned=True,
        )

        assert len(result) == 1
        assert isinstance(result[0], TextBlock)
        assert result[0].text == "Recovered text from OCR"
        # Page-spanning bbox so the writer doesn't choke
        assert result[0].bbox.x1 == 0 and result[0].bbox.y1 == 0
        assert result[0].bbox.x2 == 612 and result[0].bbox.y2 == 792
