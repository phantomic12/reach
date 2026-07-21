"""
Tests for RemediationPipeline — full pipeline integration.
"""
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Pdf

from reach.construct.pipeline import RemediationPipeline
from reach.construct.writer import DocumentMetadata, TagNode


class TestRemediationPipeline:
    """Integration tests for the full remediation pipeline."""

    @pytest.fixture
    def pipeline(self):
        return RemediationPipeline()

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a simple sample PDF for pipeline testing."""
        pdf_path = tmp_path / "input.pdf"
        pdf = Pdf.new()
        pdf.add_blank_page(page_size=(612, 792))
        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    @pytest.fixture
    def pdf_with_form_and_links(self, tmp_path):
        """Create a PDF with both form fields and link annotations."""
        from pikepdf import Array, Dictionary, Name, String

        pdf_path = tmp_path / "rich.pdf"
        pdf = Pdf.new()
        page = pdf.add_blank_page(page_size=(612, 792))

        # Add form field
        acroform = Dictionary()
        acroform[Name("/Fields")] = Array()
        field = Dictionary()
        field[Name("/FT")] = Name("/Tx")
        field[Name("/T")] = String("test_field")
        field[Name("/P")] = page.obj
        acroform["/Fields"].append(field)
        pdf.Root[Name("/AcroForm")] = acroform

        # Add link annotation
        annot = Dictionary()
        annot[Name("/Type")] = Name("/Annot")
        annot[Name("/Subtype")] = Name("/Link")
        annot[Name("/Rect")] = Array([10, 700, 100, 730])
        action = Dictionary()
        action[Name("/S")] = Name("/URI")
        action[Name("/URI")] = String("https://example.com")
        annot[Name("/A")] = action
        page[Name("/Annots")] = Array([annot])

        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    def test_remediate_creates_output(self, pipeline, sample_pdf, tmp_path):
        """Test that remediate() produces an output PDF."""
        output = tmp_path / "output.pdf"
        report_path = tmp_path / "report.json"

        report = pipeline.remediate(
            input_path=sample_pdf,
            output=output,
            report_path=report_path,
            validate=False,  # Skip veraPDF in tests
        )

        assert output.exists()
        assert output.stat().st_size > 0
        assert report.overall_status in ("pass", "partial_pass")

    def test_remediate_creates_report(self, pipeline, sample_pdf, tmp_path):
        """Test that remediate() produces a report JSON."""
        output = tmp_path / "output.pdf"
        report_path = tmp_path / "report.json"

        report = pipeline.remediate(
            input_path=sample_pdf,
            output=output,
            report_path=report_path,
            validate=False,
        )

        assert report_path.exists()
        # report.input_path is a full path — should exist
        assert report.input_path
        assert report.total_fixes > 0

    def test_remediate_missing_input(self, pipeline, tmp_path):
        """Test that remediate() raises for missing input PDF."""
        with pytest.raises(FileNotFoundError):
            pipeline.remediate(
                input_path=tmp_path / "nonexistent.pdf",
                output=tmp_path / "output.pdf",
                report_path=tmp_path / "report.json",
                validate=False,
            )

    def test_remediate_with_forms(self, pipeline, pdf_with_form_and_links, tmp_path):
        """Test remediation with form fields."""
        output = tmp_path / "output.pdf"
        report_path = tmp_path / "report.json"

        report = pipeline.remediate(
            input_path=pdf_with_form_and_links,
            output=output,
            report_path=report_path,
            validate=False,
        )

        assert output.exists()

    def test_remediate_with_metadata_overrides(self, pipeline, sample_pdf, tmp_path):
        """Test remediation with custom metadata."""
        output = tmp_path / "output.pdf"
        report_path = tmp_path / "report.json"

        report = pipeline.remediate(
            input_path=sample_pdf,
            output=output,
            report_path=report_path,
            validate=False,
            metadata={"title": "Custom Title", "language": "es-ES"},
        )

        # Open output and verify metadata
        pdf = Pdf.open(output)
        root = pdf.Root
        assert str(root["/Lang"]) == "es-ES"
        pdf.close()

    def test_extract_returns_structured_data(self, pipeline, sample_pdf):
        """Test that _extract returns the expected structure."""
        data = pipeline._extract(sample_pdf)

        assert "pages" in data
        assert "forms" in data
        assert "links" in data
        assert "fonts" in data
        assert "metadata" in data
        assert isinstance(data["pages"], list)
        assert len(data["pages"]) > 0

    def test_build_tag_tree_returns_root_document(self, pipeline, sample_pdf):
        """Test that _build_tag_tree returns a TagNode with Document tag."""
        extraction = pipeline._extract(sample_pdf)
        tag_tree = pipeline._build_tag_tree(extraction)

        assert tag_tree.tag == "Document"
        assert len(tag_tree.children) > 0

    def test_build_metadata_returns_document_metadata(self, pipeline, sample_pdf):
        """Test that _build_metadata returns valid DocumentMetadata."""
        extraction = pipeline._extract(sample_pdf)
        meta = pipeline._build_metadata(extraction)

        assert isinstance(meta, DocumentMetadata)
        assert meta.language is not None
        assert meta.display_title is True

    def test_build_metadata_with_overrides(self, pipeline, sample_pdf):
        """Test metadata overrides."""
        extraction = pipeline._extract(sample_pdf)
        meta = pipeline._build_metadata(extraction, {"title": "Override", "language": "fr-FR"})

        assert meta.title == "Override"
        assert meta.language == "fr-FR"
