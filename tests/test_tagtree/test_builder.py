"""Tests for the tag tree builder."""

import json

import pytest

from reach.tagtree.builder import (
    TagTreeBuilder,
    build_tag_tree,
    build_tag_tree_from_file,
)
from reach.tagtree.tag_models import (
    BBox,
    ExtractionBlock,
    ExtractionDocument,
    ExtractionPage,
    FontInfo,
    TagTree,
    TagType,
)


def make_block(
    block_type: str,
    text: str,
    x0: float = 0,
    y0: float = 0,
    x1: float = 100,
    y1: float = 20,
    font_size: float = 12.0,
    bold: bool = False,
    page: int = 1,
) -> ExtractionBlock:
    """Helper to create extraction blocks quickly."""
    return ExtractionBlock(
        type=block_type,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        text=text,
        font_info=FontInfo(family="Times", size=font_size, bold=bold),
        page_number=page,
    )


def make_doc(blocks: list[ExtractionBlock]) -> ExtractionDocument:
    """Create a single-page extraction document from a list of blocks."""
    return ExtractionDocument(
        filename="test.pdf",
        pages=[
            ExtractionPage(
                page_number=1,
                width=612,
                height=792,
                blocks=blocks,
            )
        ],
    )


class TestTagTreeBuilder:
    def test_build_empty_document(self):
        doc = ExtractionDocument()
        builder = TagTreeBuilder()
        tree = builder.build(doc)
        assert tree.root.tag == TagType.DOCUMENT
        assert len(tree.root.children) == 0

    def test_text_to_paragraph(self):
        doc = make_doc([make_block("Text", "A simple paragraph.")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert len(tree.root.children) == 1
        assert tree.root.children[0].tag == TagType.P
        assert tree.root.children[0].text == "A simple paragraph."

    def test_section_header_to_heading(self):
        doc = make_doc([
            make_block("SectionHeader", "Introduction", font_size=18, bold=True),
            make_block("Text", "Some content."),
        ])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        headings = [n for n in tree.root.children if n.tag.value.startswith("H")]
        assert len(headings) == 1
        assert headings[0].text == "Introduction"

    def test_title_to_h1(self):
        doc = make_doc([make_block("Title", "My Document Title", font_size=24, bold=True)])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.H1
        assert tree.root.children[0].text == "My Document Title"

    def test_heading_hierarchy_inference(self):
        """Headings with different font sizes should get different levels."""
        doc = make_doc([
            make_block("SectionHeader", "Chapter 1", font_size=20, bold=True),
            make_block("SectionHeader", "Section 1.1", font_size=16, bold=True),
            make_block("SectionHeader", "Subsection 1.1.1", font_size=14, bold=True),
            make_block("Text", "Content under subsection."),
        ])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        headings = [n for n in tree.root.children if "H" in n.tag.value]
        assert len(headings) == 3

        # Largest font size → H1
        assert headings[0].tag == TagType.H1
        assert headings[1].tag == TagType.H2
        assert headings[2].tag == TagType.H3

    def test_numbered_heading_depth(self):
        """Numbered headings like '1.1.1' should get deeper levels."""
        doc = make_doc([
            make_block("SectionHeader", "1 Introduction", font_size=18, bold=True),
            make_block("SectionHeader", "1.1 Background", font_size=16, bold=True),
            make_block("SectionHeader", "1.1.1 Details", font_size=14, bold=True),
        ])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        headings = [n for n in tree.root.children if "H" in n.tag.value]
        assert headings[0].tag == TagType.H1
        assert headings[1].tag == TagType.H2
        assert headings[2].tag == TagType.H3

    def test_page_header_to_artifact(self):
        doc = make_doc([make_block("PageHeader", "Page 1 of 10")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.ARTIFACT

    def test_page_footer_to_artifact(self):
        doc = make_doc([make_block("PageFooter", "Copyright 2024")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.ARTIFACT

    def test_page_number_to_artifact(self):
        doc = make_doc([make_block("PageNumber", "42")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.ARTIFACT

    def test_figure_to_figure(self):
        doc = make_doc([make_block("Figure", "chart.png")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.FIGURE

    def test_formula_to_formula(self):
        doc = make_doc([make_block("Formula", "E = mc^2")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.FORMULA

    def test_list_item_with_bullet(self):
        doc = make_doc([make_block("List", "• First item in list")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        node = tree.root.children[0]
        assert node.tag == TagType.L
        assert len(node.children) == 1  # LI
        li = node.children[0]
        assert li.tag == TagType.LI
        assert len(li.children) == 2  # Lbl + LBody
        assert li.children[0].tag == TagType.LBL
        assert "•" in li.children[0].text
        assert li.children[1].tag == TagType.LBODY
        assert "First item" in li.children[1].text

    def test_list_item_with_number(self):
        doc = make_doc([make_block("List", "1. First numbered item")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        node = tree.root.children[0]
        assert node.tag == TagType.L
        li = node.children[0]
        assert li.tag == TagType.LI
        assert li.children[0].tag == TagType.LBL
        assert "1." in li.children[0].text
        assert li.children[1].tag == TagType.LBODY

    def test_plain_list_item_no_bullet(self):
        doc = make_doc([make_block("List", "Just an item without bullet")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        node = tree.root.children[0]
        li = node.children[0]
        assert li.children[0].tag == TagType.LBODY

    def test_table_block(self):
        doc = make_doc([make_block("Table", "table data")])
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert tree.root.children[0].tag == TagType.TABLE

    def test_multi_page_document(self):
        doc = ExtractionDocument(
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[make_block("Text", "Page 1 text", page=1)],
                ),
                ExtractionPage(
                    page_number=2,
                    width=612,
                    height=792,
                    blocks=[make_block("Text", "Page 2 text", page=2)],
                ),
            ],
        )
        builder = TagTreeBuilder()
        tree = builder.build(doc)

        assert len(tree.root.children) == 2
        assert tree.root.children[0].page_number == 1
        assert tree.root.children[1].page_number == 2


class TestBuildTagTreeConvenience:
    def test_from_dict(self):
        data = {
            "filename": "test.pdf",
            "pages": [
                {
                    "page_number": 1,
                    "width": 612,
                    "height": 792,
                    "blocks": [
                        {
                            "type": "Text",
                            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
                            "text": "Hello world",
                            "reading_order": 0,
                            "page_number": 1,
                            "confidence": 1.0,
                        }
                    ],
                }
            ],
        }
        tree = build_tag_tree(data)
        assert isinstance(tree, TagTree)
        assert len(tree.root.children) == 1
        assert tree.root.children[0].text == "Hello world"

    def test_from_file(self, tmp_path):
        data = {
            "filename": "test.pdf",
            "pages": [
                {
                    "page_number": 1,
                    "width": 612,
                    "height": 792,
                    "blocks": [
                        {
                            "type": "Title",
                            "bbox": {"x0": 0, "y0": 0, "x1": 200, "y1": 30},
                            "text": "Test Title",
                            "reading_order": 0,
                            "page_number": 1,
                            "confidence": 1.0,
                        }
                    ],
                }
            ],
        }
        json_path = tmp_path / "extraction.json"
        json_path.write_text(json.dumps(data))

        tree = build_tag_tree_from_file(str(json_path))
        assert tree.root.children[0].text == "Test Title"
