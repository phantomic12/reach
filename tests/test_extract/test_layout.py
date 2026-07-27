"""
Tests for reach.extract.layout — Surya layout analysis integration.

All tests use mocks — no Surya download required.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from reach.extract.layout import (
    SuryaBlock,
    SuryaLayoutModel,
    _naive_text_distribution,
    analyze_layout,
    convert_surya_blocks,
    load_surya,
    merge_text_with_layout,
)
from reach.extract.models import (
    BlockType,
    BBox,
    EquationBlock,
    ImageBlock,
    TableBlock,
    TextBlock,
    map_surya_block_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_images():
    """Create a few small test images."""
    return [
        Image.new("RGB", (612, 792), color=(255, 255, 255)),
        Image.new("RGB", (612, 792), color=(240, 240, 240)),
    ]


@pytest.fixture
def mock_surya_blocks():
    """Simulate Surya detection output for one page."""
    return [
        SuryaBlock(label="Section-header", bbox=[50, 50, 550, 80], reading_order=0, confidence=0.95),
        SuryaBlock(label="Plain text", bbox=[50, 100, 550, 300], reading_order=1, confidence=0.90),
        SuryaBlock(label="Table", bbox=[50, 320, 550, 500], reading_order=2, confidence=0.88),
        SuryaBlock(label="Picture", bbox=[50, 520, 300, 700], reading_order=3, confidence=0.92),
        SuryaBlock(label="Formula", bbox=[50, 720, 400, 760], reading_order=4, confidence=0.85),
        SuryaBlock(label="Page-footer", bbox=[200, 770, 400, 785], reading_order=5, confidence=0.99),
    ]


# ---------------------------------------------------------------------------
# SuryaBlock
# ---------------------------------------------------------------------------


class TestSuryaBlock:
    def test_creation(self):
        sb = SuryaBlock(label="Text", bbox=[0, 0, 10, 10], reading_order=1, confidence=0.9)
        assert sb.label == "Text"
        assert sb.bbox == [0, 0, 10, 10]
        assert sb.reading_order == 1
        assert sb.confidence == 0.9

    def test_repr(self):
        sb = SuryaBlock(label="Table", bbox=[1, 2, 3, 4])
        r = repr(sb)
        assert "Table" in r
        assert "[1, 2, 3, 4]" in r


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestLoadSurya:
    def test_load_when_not_installed(self, monkeypatch):
        """When REACH_NO_ML is set, model loads without trying backends."""
        monkeypatch.setenv("REACH_NO_ML", "1")
        model = load_surya(backend="cpu")
        assert not model.is_available


# ---------------------------------------------------------------------------
# SuryaLayoutModel
# ---------------------------------------------------------------------------


class TestSuryaLayoutModel:
    def test_default_unavailable(self):
        model = SuryaLayoutModel(backend="cpu")
        model.load()
        assert not model.is_available

    def test_detect_empty_when_unavailable(self, sample_images):
        model = SuryaLayoutModel(backend="cpu")
        model.load()
        result = model.detect(sample_images)
        assert result == [[] for _ in sample_images]

    def test_detect_with_mock(self, sample_images, mock_surya_blocks):
        """Inject mock DocLayout-YOLO so we can test the conversion pipeline."""
        model = SuryaLayoutModel(backend="cpu")
        # Fake availability + mock model
        model._available = True
        model._model = MagicMock()
        # Build the shape detect() expects: list[page_result], each page_result is iterable
        # of dicts with label/bbox/confidence keys.
        page_results = []
        for sb in mock_surya_blocks:
            page_results.append({
                "label": sb.label,
                "bbox": sb.bbox,
                "confidence": sb.confidence,
            })
        model._model.detect.return_value = [page_results]

        result = model.detect(sample_images[:1])
        assert len(result) == 1
        # Should have our mock blocks
        assert len(result[0]) == len(mock_surya_blocks)
        assert result[0][0].label == "Section-header"

    def test_unload(self):
        model = SuryaLayoutModel(backend="cpu")
        model.load()
        model.unload()
        assert model._model is None


# ---------------------------------------------------------------------------
# Block conversion
# ---------------------------------------------------------------------------


class TestConvertSuryaBlocks:
    def test_convert_all_types(self, mock_surya_blocks):
        blocks = convert_surya_blocks(mock_surya_blocks, page_num=0)
        assert len(blocks) == len(mock_surya_blocks)

        # Verify types
        assert isinstance(blocks[0], TextBlock)  # Section-header → TextBlock
        assert blocks[0].block_type == BlockType.SECTION_HEADER
        assert isinstance(blocks[1], TextBlock)  # Plain text → TextBlock
        assert blocks[1].block_type == BlockType.TEXT
        assert isinstance(blocks[2], TableBlock)   # Table
        assert isinstance(blocks[3], ImageBlock)   # Picture → Figure
        assert isinstance(blocks[4], EquationBlock)  # Formula
        assert isinstance(blocks[5], TextBlock)    # Page-footer → TextBlock
        assert blocks[5].block_type == BlockType.PAGE_FOOTER

    def test_convert_empty(self):
        blocks = convert_surya_blocks([])
        assert blocks == []

    def test_sorted_by_reading_order(self):
        """Blocks should be returned sorted by reading_order regardless of input order."""
        unordered = [
            SuryaBlock(label="Text", bbox=[0, 0, 10, 10], reading_order=3),
            SuryaBlock(label="Text", bbox=[0, 0, 10, 10], reading_order=1),
            SuryaBlock(label="Text", bbox=[0, 0, 10, 10], reading_order=2),
        ]
        blocks = convert_surya_blocks(unordered)
        orders = [b.reading_order for b in blocks]
        assert orders == [1, 2, 3]

    def test_bbox_conversion(self, mock_surya_blocks):
        blocks = convert_surya_blocks(mock_surya_blocks[:1])
        bbox = blocks[0].bbox
        assert bbox.to_list() == [50, 50, 550, 80]


# ---------------------------------------------------------------------------
# Text merging
# ---------------------------------------------------------------------------


class TestMergeTextWithLayout:
    def test_merge_distributes_text_to_blocks(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        blocks: list = [
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=0),
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=1),
            TableBlock(bbox=bbox, reading_order=2),
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=3),
        ]
        page_text = "First paragraph about something.\n\nSecond paragraph.\n\nThird one."

        result = merge_text_with_layout(blocks, page_text)
        text_blocks = [b for b in result if isinstance(b, TextBlock)]
        # Three text blocks should have text
        non_empty = [b for b in text_blocks if b.text]
        assert len(non_empty) > 0

    def test_merge_empty_text(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        blocks = [TextBlock(block_type=BlockType.TEXT, bbox=bbox, reading_order=0)]
        result = merge_text_with_layout(blocks, "")
        assert result[0].text == ""

    def test_merge_no_text_blocks(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        blocks = [TableBlock(bbox=bbox, reading_order=0)]
        result = merge_text_with_layout(blocks, "some text")
        assert result[0].block_type == BlockType.TABLE


# ---------------------------------------------------------------------------
# Naive text distribution
# ---------------------------------------------------------------------------


class TestNaiveTextDistribution:
    def test_zero_blocks(self):
        assert _naive_text_distribution("hello", 0) == []

    def test_one_block(self):
        result = _naive_text_distribution("hello world", 1)
        assert len(result) == 1
        assert result[0].strip()

    def test_by_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = _naive_text_distribution(text, 2)
        assert len(result) == 2
        assert result[0].strip()
        assert result[1].strip()

    def test_fallback_char_split(self):
        """Single word, many blocks — falls back to character-level split."""
        result = _naive_text_distribution("ABCDEFGHI", 3)
        assert len(result) == 3
        assert all(r for r in result)  # all non-empty


# ---------------------------------------------------------------------------
# analyze_layout
# ---------------------------------------------------------------------------


class TestAnalyzeLayout:
    def test_returns_empty_when_no_surya(self, sample_images):
        result = analyze_layout(sample_images, model=None)
        # With no surya installed, should return empty per page
        assert len(result) == len(sample_images)
        assert result == [[] for _ in sample_images]

    def test_with_mock_surya(self, sample_images):
        """Full flow with a mocked layout model."""
        model = SuryaLayoutModel(backend="cpu")

        # Set up mock availability
        model._available = True
        model._model = MagicMock()
        # detect() expects a list of page_results, each iterable of dicts.
        model._model.detect.return_value = [[{
            "label": "Plain text",
            "bbox": [0, 0, 100, 20],
            "confidence": 0.95,
        }]]

        page_texts = ["Hello world from the PDF content stream."]
        result = analyze_layout(sample_images[:1], model=model, page_texts=page_texts)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], TextBlock)
        # Text should be merged from content stream
        assert result[0][0].text
