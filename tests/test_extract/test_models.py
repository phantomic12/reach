"""
Tests for reach.extract.models — Pydantic models for extraction output.
"""

import json

import pytest
from pydantic import ValidationError

from reach.extract.models import (
    SURYA_TO_REACH,
    BaseBlock,
    BBox,
    BlockType,
    DocumentMetadata,
    EquationBlock,
    ExtractedDocument,
    FontInfo,
    ImageBlock,
    Page,
    PageMetadata,
    TableBlock,
    TableCell,
    TextBlock,
    map_surya_block_type,
)


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------


class TestBBox:
    def test_create_valid_bbox(self):
        bbox = BBox(x1=0, y1=10, x2=100, y2=200)
        assert bbox.width == 100
        assert bbox.height == 190
        assert bbox.area == 19000

    def test_x2_lt_x1_raises(self):
        with pytest.raises(ValidationError):
            BBox(x1=100, y1=0, x2=50, y2=200)

    def test_y2_lt_y1_raises(self):
        with pytest.raises(ValidationError):
            BBox(x1=0, y1=200, x2=100, y2=50)

    def test_from_list(self):
        bbox = BBox.from_list([10.0, 20.0, 110.0, 220.0])
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 110.0
        assert bbox.y2 == 220.0

    def test_to_list(self):
        bbox = BBox(x1=5, y1=15, x2=55, y2=65)
        assert bbox.to_list() == [5, 15, 55, 65]

    def test_roundtrip_json(self):
        bbox = BBox(x1=1.5, y1=2.5, x2=3.5, y2=4.5)
        data = bbox.model_dump_json()
        loaded = BBox.model_validate_json(data)
        assert loaded.to_list() == [1.5, 2.5, 3.5, 4.5]


# ---------------------------------------------------------------------------
# BlockType & mapping
# ---------------------------------------------------------------------------


class TestBlockType:
    def test_all_surya_labels_mapped(self):
        """Every Surya label in the mapping should resolve to a valid BlockType."""
        for raw, bt in SURYA_TO_REACH.items():
            assert isinstance(bt, BlockType), f"{raw!r} → {bt} is not a BlockType"

    def test_map_known_labels(self):
        assert map_surya_block_type("Section-header") == BlockType.SECTION_HEADER
        assert map_surya_block_type("Table") == BlockType.TABLE
        assert map_surya_block_type("Picture") == BlockType.FIGURE
        assert map_surya_block_type("Formula") == BlockType.EQUATION
        assert map_surya_block_type("Stamp") == BlockType.ARTIFACT

    def test_map_unknown_label_falls_back_to_text(self):
        assert map_surya_block_type("Unknown-thing") == BlockType.TEXT

    def test_enum_values(self):
        assert BlockType.SECTION_HEADER.value == "SectionHeader"
        assert BlockType.TEXT.value == "Text"


# ---------------------------------------------------------------------------
# FontInfo
# ---------------------------------------------------------------------------


class TestFontInfo:
    def test_defaults(self):
        fi = FontInfo()
        assert fi.name == "Unknown"
        assert fi.size == 0.0
        assert not fi.bold
        assert not fi.italic
        assert fi.color_hex == "#000000"

    def test_custom(self):
        fi = FontInfo(name="Times New Roman", size=12.0, bold=True, italic=True, color_hex="#ff0000")
        assert fi.name == "Times New Roman"
        assert fi.size == 12.0
        assert fi.bold
        assert fi.italic
        assert fi.color_hex == "#ff0000"


# ---------------------------------------------------------------------------
# TextBlock
# ---------------------------------------------------------------------------


class TestTextBlock:
    def test_minimal_creation(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=50)
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="Hello")
        assert block.text == "Hello"
        assert block.reading_order == 0
        assert block.confidence == 0.0

    def test_with_font(self):
        fi = FontInfo(name="Arial", size=10)
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        block = TextBlock(block_type=BlockType.SECTION_HEADER, bbox=bbox, text="Title", font_info=fi)
        assert block.font_info.name == "Arial"

    def test_json_serialization(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=25)
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="test", reading_order=1)
        data = block.model_dump_json()
        parsed = json.loads(data)
        assert parsed["text"] == "test"
        assert parsed["reading_order"] == 1


# ---------------------------------------------------------------------------
# TableBlock / TableCell
# ---------------------------------------------------------------------------


class TestTableBlock:
    def test_empty_table(self):
        bbox = BBox(x1=0, y1=0, x2=200, y2=100)
        table = TableBlock(bbox=bbox)
        assert table.rows == 0
        assert table.columns == 0
        assert table.cells == []

    def test_populated_table(self):
        bbox = BBox(x1=0, y1=0, x2=200, y2=200)
        cells = [
            TableCell(text="Name", row=0, col=0, bbox=BBox(x1=0, y1=0, x2=100, y2=100), is_header=True),
            TableCell(text="Alice", row=1, col=0, bbox=BBox(x1=0, y1=100, x2=100, y2=200)),
        ]
        table = TableBlock(bbox=bbox, rows=2, columns=1, cells=cells, caption="Table 1")
        assert table.rows == 2
        assert table.columns == 1
        assert len(table.cells) == 2
        assert table.caption == "Table 1"

    def test_cell_rowspan_default(self):
        cell = TableCell(text="A", row=0, col=0, bbox=BBox(x1=0, y1=0, x2=10, y2=10))
        assert cell.rowspan == 1
        assert cell.colspan == 1
        assert not cell.is_header

    def test_json_roundtrip(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=200)
        cells = [
            TableCell(text="H1", row=0, col=0, bbox=BBox(x1=0, y1=0, x2=50, y2=50), is_header=True),
        ]
        table = TableBlock(bbox=bbox, rows=1, columns=1, cells=cells)
        data = table.model_dump_json()
        loaded = TableBlock.model_validate_json(data)
        assert loaded.rows == 1
        assert loaded.cells[0].text == "H1"


# ---------------------------------------------------------------------------
# ImageBlock
# ---------------------------------------------------------------------------


class TestImageBlock:
    def test_minimal_creation(self):
        bbox = BBox(x1=0, y1=0, x2=300, y2=200)
        img = ImageBlock(bbox=bbox)
        assert img.block_type == BlockType.FIGURE
        assert img.image_mime == "image/png"

    def test_with_caption(self):
        bbox = BBox(x1=0, y1=0, x2=200, y2=150)
        img = ImageBlock(bbox=bbox, caption="Figure 1: Overview")
        assert img.caption == "Figure 1: Overview"
        assert img.alt_text is None

    def test_image_bytes_excluded_from_serialization(self):
        bbox = BBox(x1=0, y1=0, x2=10, y2=10)
        img = ImageBlock(bbox=bbox, image_bytes=b"\x89PNG")
        data = img.model_dump(mode="json")
        assert "image_bytes" not in data


# ---------------------------------------------------------------------------
# EquationBlock
# ---------------------------------------------------------------------------


class TestEquationBlock:
    def test_creation(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=30)
        eq = EquationBlock(bbox=bbox, latex=r"E = mc^2")
        assert eq.latex == r"E = mc^2"
        assert eq.block_type == BlockType.EQUATION

    def test_empty_latex(self):
        bbox = BBox(x1=0, y1=0, x2=50, y2=20)
        eq = EquationBlock(bbox=bbox)
        assert eq.latex == ""


# ---------------------------------------------------------------------------
# Page / PageMetadata
# ---------------------------------------------------------------------------


class TestPage:
    def test_empty_page(self):
        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        page = Page(metadata=meta)
        assert page.metadata.page_number == 1
        assert page.blocks == []

    def test_property_filters(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        tb = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="hello")
        tbl = TableBlock(bbox=bbox)
        img = ImageBlock(bbox=bbox)
        eq = EquationBlock(bbox=bbox, latex="x=1")

        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        page = Page(metadata=meta, blocks=[tb, tbl, img, eq])

        assert page.text_blocks == [tb]
        assert page.table_blocks == [tbl]
        assert page.figure_blocks == [img]
        assert page.equation_blocks == [eq]

    def test_scanned_flag(self):
        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792, is_scanned=True)
        assert meta.is_scanned
        assert meta.text_coverage == 0.0


# ---------------------------------------------------------------------------
# ExtractedDocument
# ---------------------------------------------------------------------------


class TestExtractedDocument:
    def test_empty_document(self):
        doc = ExtractedDocument()
        assert doc.pages == []
        assert doc.all_blocks == []

    def test_document_with_pages(self):
        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        bbox = BBox(x1=0, y1=0, x2=100, y2=20)
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="test")
        page = Page(metadata=meta, blocks=[block])

        doc = ExtractedDocument(
            metadata=DocumentMetadata(title="Test", page_count=1),
            pages=[page],
        )
        assert len(doc.pages) == 1
        assert len(doc.all_blocks) == 1
        assert doc.metadata.title == "Test"

    def test_json_roundtrip(self, tmp_path):
        bbox = BBox(x1=0, y1=0, x2=100, y2=30)
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="hello world", reading_order=0)
        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        page = Page(metadata=meta, blocks=[block])
        doc_meta = DocumentMetadata(title="Test Doc", author="Yoav", page_count=1)
        doc = ExtractedDocument(metadata=doc_meta, pages=[page])

        # Write
        output = tmp_path / "output.json"
        doc.write_json(output)

        # Read
        assert output.exists()
        loaded = ExtractedDocument.from_json(output)
        assert loaded.metadata.title == "Test Doc"
        assert len(loaded.pages) == 1
        assert loaded.pages[0].blocks[0].text == "hello world"

    def test_json_output_is_valid_json(self, tmp_path):
        bbox = BBox(x1=0, y1=0, x2=10, y2=10)
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="hi")
        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        page = Page(metadata=meta, blocks=[block])
        doc = ExtractedDocument(pages=[page])

        output = tmp_path / "test.json"
        doc.write_json(output)

        with open(output) as f:
            data = json.load(f)
        assert data["pages"][0]["metadata"]["page_number"] == 1

    def test_confidence_range_validation(self):
        bbox = BBox(x1=0, y1=0, x2=10, y2=10)
        # Valid range
        block = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="ok", confidence=0.85)
        assert block.confidence == 0.85

        # Out of range should fail
        with pytest.raises(ValidationError):
            TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="bad", confidence=1.5)

    def test_block_type_discriminated_union(self):
        """Ensure all block variant types can be stored together in a Page.blocks list."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=50)
        tb = TextBlock(block_type=BlockType.TEXT, bbox=bbox, text="text")
        tbl = TableBlock(bbox=bbox)
        img = ImageBlock(bbox=bbox)
        eq = EquationBlock(bbox=bbox, latex="x")

        meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
        page = Page(metadata=meta, blocks=[tb, tbl, img, eq])

        doc = ExtractedDocument(pages=[page])
        data = doc.model_dump(mode="json")

        # Round-trip
        loaded = ExtractedDocument.model_validate(data)
        assert len(loaded.pages[0].blocks) == 4
