"""Tests for tag tree tag_models."""

import pytest

from reach.tagtree.tag_models import (
    BBox,
    ExtractionBlock,
    ExtractionDocument,
    ExtractionPage,
    FontInfo,
    SURYA_TAG_MAP,
    TagNode,
    TagTree,
    TagTreeConfig,
    TagType,
)


class TestBBox:
    def test_bbox_properties(self):
        bbox = BBox(x0=0, y0=10, x1=100, y1=50)
        assert bbox.width == 100
        assert bbox.height == 40
        assert bbox.area == 4000

    def test_bbox_serialization(self):
        bbox = BBox(x0=1.5, y0=2.0, x1=3.0, y1=4.0)
        data = bbox.model_dump()
        assert data == {"x0": 1.5, "y0": 2.0, "x1": 3.0, "y1": 4.0}


class TestExtractionModels:
    def test_extraction_block_minimal(self):
        block = ExtractionBlock(
            type="Text",
            bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            text="Hello",
        )
        assert block.type == "Text"
        assert block.text == "Hello"
        assert block.confidence == 1.0

    def test_extraction_document_validate(self):
        doc = ExtractionDocument(
            filename="test.pdf",
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        ExtractionBlock(
                            type="SectionHeader",
                            bbox=BBox(x0=50, y0=20, x1=200, y1=40),
                            text="Chapter 1",
                            font_info=FontInfo(size=16, bold=True),
                        )
                    ],
                )
            ],
        )
        assert len(doc.pages) == 1
        assert doc.pages[0].blocks[0].text == "Chapter 1"


class TestTagType:
    def test_surya_tag_mapping(self):
        """Test that common Surya block types map to correct tag types."""
        assert SURYA_TAG_MAP["Text"] == TagType.P
        assert SURYA_TAG_MAP["SectionHeader"] == TagType.H
        assert SURYA_TAG_MAP["Title"] == TagType.H1
        assert SURYA_TAG_MAP["Table"] == TagType.TABLE
        assert SURYA_TAG_MAP["Figure"] == TagType.FIGURE
        assert SURYA_TAG_MAP["List"] == TagType.L
        assert SURYA_TAG_MAP["PageHeader"] == TagType.ARTIFACT
        assert SURYA_TAG_MAP["PageFooter"] == TagType.ARTIFACT
        assert SURYA_TAG_MAP["PageNumber"] == TagType.ARTIFACT
        assert SURYA_TAG_MAP["Formula"] == TagType.FORMULA
        assert SURYA_TAG_MAP["Caption"] == TagType.CAPTION
        assert SURYA_TAG_MAP["BlockQuote"] == TagType.BLOCKQUOTE
        assert SURYA_TAG_MAP["Code"] == TagType.CODE

    def test_tag_type_enum_values(self):
        assert TagType.DOCUMENT.value == "Document"
        assert TagType.H1.value == "H1"
        assert TagType.H6.value == "H6"
        assert TagType.P.value == "P"
        assert TagType.L.value == "L"
        assert TagType.TABLE.value == "Table"


class TestTagNode:
    def test_basic_tag_node(self):
        node = TagNode(tag=TagType.P, text="A paragraph.")
        assert node.tag == TagType.P
        assert node.text == "A paragraph."
        assert node.children == []
        assert node.attributes == {}

    def test_nested_structure(self):
        root = TagNode(tag=TagType.DOCUMENT)
        h1 = TagNode(tag=TagType.H1, text="Title")
        p = TagNode(tag=TagType.P, text="Content")
        root.children = [h1, p]

        assert len(root.children) == 2
        assert root.children[0].tag == TagType.H1
        assert root.children[1].text == "Content"

    def test_table_cell_attributes(self):
        th = TagNode(
            tag=TagType.TH,
            text="Header",
            scope="Column",
            colspan=1,
            rowspan=1,
        )
        assert th.scope == "Column"
        assert th.colspan == 1

    def test_alt_text(self):
        node = TagNode(tag=TagType.FIGURE, alt_text="A chart showing sales data.")
        assert node.alt_text == "A chart showing sales data."
        assert not node.is_decorative

    def test_decorative_figure(self):
        node = TagNode(
            tag=TagType.FIGURE,
            is_decorative=True,
        )
        assert node.is_decorative
        assert node.alt_text == ""

    def test_to_dict_serialization(self):
        node = TagNode(tag=TagType.H1, text="Hello", page_number=1)
        data = node.to_dict()
        assert data["tag"] == "H1"
        assert data["text"] == "Hello"
        assert data["page_number"] == 1


class TestTagTree:
    def test_full_tag_tree(self, tmp_path):
        doc = ExtractionDocument(filename="test.pdf")
        root = TagNode(tag=TagType.DOCUMENT)
        root.children = [
            TagNode(tag=TagType.H1, text="Title"),
            TagNode(tag=TagType.P, text="Body"),
        ]
        tree = TagTree(
            document=doc,
            root=root,
            title="Test Document",
            language="en-US",
        )

        assert tree.title == "Test Document"
        assert tree.language == "en-US"
        assert len(tree.root.children) == 2

    def test_json_roundtrip(self, tmp_path):
        doc = ExtractionDocument(filename="roundtrip.pdf")
        root = TagNode(tag=TagType.DOCUMENT)
        root.children = [TagNode(tag=TagType.P, text="text")]

        tree = TagTree(document=doc, root=root, title="Test")
        json_path = tmp_path / "tags.json"
        tree.to_json(json_path)

        loaded = TagTree.from_json(json_path)
        assert loaded.title == "Test"
        assert len(loaded.root.children) == 1


class TestTagTreeConfig:
    def test_defaults(self):
        config = TagTreeConfig()
        assert config.enable_enrichment is True
        assert config.enable_xy_cut is True
        assert config.min_alt_text_length == 10
        assert config.max_alt_text_length == 500
        assert "a chart" in config.reject_generic_phrases

    def test_custom_config(self):
        config = TagTreeConfig(
            enable_enrichment=False,
            min_alt_text_length=20,
            column_gap_threshold=30.0,
        )
        assert config.enable_enrichment is False
        assert config.min_alt_text_length == 20
        assert config.column_gap_threshold == 30.0
