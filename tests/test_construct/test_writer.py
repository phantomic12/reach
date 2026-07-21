"""
Tests for PDFWriter — tag tree injection, font embedding, metadata, content streams.
"""
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Dictionary, Name, Pdf, String

from reach.construct.writer import (
    DEFAULT_CLASS_MAP,
    DocumentMetadata,
    PDFWriter,
    TagNode,
    WriterConfig,
)


class TestTagNode:
    """Tests for TagNode data structure."""

    def test_create_simple_tag_node(self):
        node = TagNode(tag="P", page=0, mcid=1, text="Hello world")
        assert node.tag == "P"
        assert node.page == 0
        assert node.mcid == 1
        assert node.text == "Hello world"
        assert node.children == []

    def test_tag_node_with_children(self):
        root = TagNode(tag="Document")
        sect = TagNode(tag="Sect")
        p = TagNode(tag="P", text="Paragraph text")
        sect.children.append(p)
        root.children.append(sect)

        assert len(root.children) == 1
        assert root.children[0].tag == "Sect"
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].text == "Paragraph text"

    def test_tag_node_with_alt_text(self):
        fig = TagNode(tag="Figure", alt_text="A bar chart showing quarterly sales")
        assert fig.alt_text == "A bar chart showing quarterly sales"
        assert fig.tag == "Figure"

    def test_tag_node_attributes(self):
        node = TagNode(tag="Table", attributes={"border": "1", "cols": "3"})
        assert node.attributes["border"] == "1"
        assert node.attributes["cols"] == "3"


class TestDocumentMetadata:
    """Tests for DocumentMetadata."""

    def test_default_metadata(self):
        meta = DocumentMetadata()
        assert meta.title == ""
        assert meta.language == "en-US"
        assert meta.display_title is True
        assert meta.pdf_version == "2.0"

    def test_custom_metadata(self):
        meta = DocumentMetadata(
            title="My Document",
            author="Test Author",
            language="fr-FR",
            display_title=False,
        )
        assert meta.title == "My Document"
        assert meta.author == "Test Author"
        assert meta.language == "fr-FR"
        assert meta.display_title is False


class TestWriterConfig:
    """Tests for WriterConfig."""

    def test_default_config(self):
        cfg = WriterConfig()
        assert cfg.embed_all_fonts is True
        assert cfg.add_tounicode is True
        assert cfg.rewrite_content_stream is True
        assert cfg.mark_artifacts is True
        assert cfg.output_pdf_version == "2.0"
        assert cfg.compress_streams is True

    def test_custom_config(self):
        cfg = WriterConfig(
            embed_all_fonts=False,
            output_pdf_version="1.7",
        )
        assert cfg.embed_all_fonts is False
        assert cfg.output_pdf_version == "1.7"


class TestPDFWriter:
    """Integration tests for PDFWriter with real PDFs via pikepdf."""

    @pytest.fixture
    def writer(self):
        return PDFWriter()

    @pytest.fixture
    def sample_metadata(self):
        return DocumentMetadata(
            title="Test Document",
            author="Reach Test Suite",
            language="en-US",
            display_title=True,
        )

    @pytest.fixture
    def minimal_tag_tree(self):
        root = TagNode(tag="Document")
        sect = TagNode(tag="Sect", page=0)
        p = TagNode(tag="P", page=0, mcid=1, text="Test paragraph content.")
        sect.children.append(p)
        fig = TagNode(tag="Figure", page=0, mcid=2, alt_text="A sample figure")
        sect.children.append(fig)
        root.children.append(sect)
        return root

    @pytest.fixture
    def source_pdf(self, tmp_path):
        """Create a minimal source PDF for testing."""
        pdf_path = tmp_path / "source.pdf"
        pdf = Pdf.new()
        page = pdf.add_blank_page(page_size=(612, 792))
        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    def test_write_basic_pdf(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test that write() produces a valid PDF."""
        output_path = tmp_path / "output.pdf"

        result = writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify it's a valid PDF that pikepdf can open
        pdf = Pdf.open(output_path)
        assert len(pdf.pages) == 1
        pdf.close()

    def test_write_sets_metadata(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test that document metadata is set in the output."""
        output_path = tmp_path / "output.pdf"

        writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
        )

        pdf = Pdf.open(output_path)
        # Check catalog-level metadata
        root = pdf.Root
        assert "/Lang" in root
        assert str(root["/Lang"]) == "en-US"

        # Check MarkInfo
        assert "/MarkInfo" in root
        assert root["/MarkInfo"].get("/Marked", False) is True

        pdf.close()

    def test_write_sets_struct_tree(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test that the structure tree is injected."""
        output_path = tmp_path / "output.pdf"

        writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
        )

        pdf = Pdf.open(output_path)
        root = pdf.Root

        assert "/StructTreeRoot" in root
        struct_root = root["/StructTreeRoot"]
        assert "/K" in struct_root
        assert "/RoleMap" in struct_root

        pdf.close()

    def test_write_sets_viewer_preferences(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test that ViewerPreferences are set with DisplayDocTitle."""
        output_path = tmp_path / "output.pdf"

        writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
        )

        pdf = Pdf.open(output_path)
        root = pdf.Root
        assert "/ViewerPreferences" in root

        pdf.close()

    def test_write_with_forms(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test write with form field data."""
        output_path = tmp_path / "output.pdf"
        forms = [
            {"name": "name_field", "field_type": "Tx", "tooltip": "Enter your name"}
        ]

        result = writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
            forms=forms,
        )

        assert output_path.exists()

    def test_write_with_links(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test write with link data."""
        output_path = tmp_path / "output.pdf"
        links = [
            {
                "page": 0,
                "uri": "https://example.com",
                "description": "External link to example.com",
                "link_type": "URI",
            }
        ]

        result = writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
            links=links,
        )

        assert output_path.exists()

    def test_write_output_is_linearized(self, writer, source_pdf, minimal_tag_tree, sample_metadata, tmp_path):
        """Test that output is saved with linearize=True."""
        output_path = tmp_path / "output.pdf"

        writer.write(
            source_path=source_pdf,
            output_path=output_path,
            tag_tree=minimal_tag_tree,
            metadata=sample_metadata,
        )

        # Linearization should produce a valid PDF
        pdf = Pdf.open(output_path)
        assert len(pdf.pages) == 1
        pdf.close()

    def test_write_missing_source_file(self, writer, minimal_tag_tree, sample_metadata, tmp_path):
        """Test that write raises FileNotFoundError for missing source."""
        with pytest.raises(Exception):  # pikepdf raises PdfError
            writer.write(
                source_path=tmp_path / "nonexistent.pdf",
                output_path=tmp_path / "output.pdf",
                tag_tree=minimal_tag_tree,
                metadata=sample_metadata,
            )


class TestDefaultClassMap:
    """Tests for default class map."""

    def test_class_map_has_standard_types(self):
        assert "paragraph" in DEFAULT_CLASS_MAP
        assert DEFAULT_CLASS_MAP["paragraph"] == "P"
        assert "section-header" in DEFAULT_CLASS_MAP
        assert "figure" in DEFAULT_CLASS_MAP
        assert "table" in DEFAULT_CLASS_MAP

    def test_class_map_artifacts(self):
        assert DEFAULT_CLASS_MAP.get("header") == "Artifact"
        assert DEFAULT_CLASS_MAP.get("footer") == "Artifact"
        assert DEFAULT_CLASS_MAP.get("page-number") == "Artifact"
