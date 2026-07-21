"""
Tests for reach.extract.parser — PDF content stream parsing.
"""

import io

import pytest
from PIL import Image

from reach.extract.parser import (
    MIN_DIGITAL_PAGE_RATIO,
    MIN_TEXT_COVERAGE_FOR_DIGITAL,
    ParsedDocument,
    _count_text_chars,
    _estimate_text_area,
    classify_document,
    classify_page,
    extract_document_metadata,
    extract_form_fields,
    extract_images,
    extract_page_text,
    rasterize_all_pages,
    rasterize_page,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory):
    """Create a real multi-page test PDF using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()

    # Page 1: text-heavy page (digital-born)
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata(
        {"/Title": "Test Document", "/Author": "Yoav", "/Subject": "Testing"}
    )

    # Page 2: another blank page (simulates scanned)
    writer.add_blank_page(width=612, height=792)

    tmp = tmp_path_factory.mktemp("pdfs") / "sample.pdf"
    with open(tmp, "wb") as f:
        writer.write(f)
    return tmp


@pytest.fixture(scope="module")
def sample_pdf(sample_pdf_path):
    """Open the sample PDF via pikepdf."""
    import pikepdf

    return pikepdf.open(sample_pdf_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_count_text_chars_empty(self):
        assert _count_text_chars("") == 0
        assert _count_text_chars("   \t\n  ") == 0

    def test_count_text_chars_mixed(self):
        assert _count_text_chars("Hello world") == 10
        assert _count_text_chars("a b c") == 3

    def test_estimate_text_area(self):
        area = _estimate_text_area("hello", font_size=10.0)
        assert area > 0


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


class TestRasterizePage:
    def test_returns_image(self, sample_pdf):
        img = rasterize_page(sample_pdf, 0, dpi=72)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_dimensions_match_page(self, sample_pdf):
        img = rasterize_page(sample_pdf, 0, dpi=72)
        page = sample_pdf.pages[0]
        w_pt = float(page.MediaBox[2] - page.MediaBox[0])
        h_pt = float(page.MediaBox[3] - page.MediaBox[1])
        assert img.width == int(w_pt)
        assert img.height == int(h_pt)

    def test_rasterize_all_pages(self, sample_pdf):
        images = rasterize_all_pages(sample_pdf, dpi=72)
        assert len(images) == len(sample_pdf.pages)
        for img in images:
            assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestExtractPageText:
    def test_extract_from_blank_page(self, sample_pdf):
        text = extract_page_text(sample_pdf, 0)
        # Blank page from pypdf should return "" or no parseable text
        assert isinstance(text, str)

    def test_extract_out_of_range(self, sample_pdf):
        text = extract_page_text(sample_pdf, 999)
        assert text == ""


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------


class TestClassifyPage:
    def test_classify_blank_page_as_scanned(self, sample_pdf):
        # Blank page has no text → scanned
        assert classify_page(sample_pdf, 0) is True

    def test_classify_document(self, sample_pdf):
        flags, overall = classify_document(sample_pdf)
        assert len(flags) == len(sample_pdf.pages)
        # Both blank pages → both scanned → overall digital is False
        assert overall is False


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


class TestExtractDocumentMetadata:
    def test_extracts_title_and_author(self, sample_pdf):
        meta = extract_document_metadata(sample_pdf)
        assert meta.title == "Test Document"
        assert meta.author == "Yoav"
        assert meta.subject == "Testing"
        assert meta.page_count == len(sample_pdf.pages)

    def test_defaults_for_missing_metadata(self):
        import pikepdf

        stream = io.BytesIO()
        writer = pikepdf.Pdf.new()
        writer.save(stream)
        stream.seek(0)
        pdf = pikepdf.open(stream)

        meta = extract_document_metadata(pdf)
        assert meta.title == ""
        assert meta.author == ""
        assert meta.language == "en"


# ---------------------------------------------------------------------------
# Form fields and images
# ---------------------------------------------------------------------------


class TestExtractFormFields:
    def test_no_form_fields_in_blank_pdf(self, sample_pdf):
        fields = extract_form_fields(sample_pdf)
        assert fields == []


class TestExtractImages:
    def test_no_images_in_blank_pdf(self, sample_pdf):
        images = extract_images(sample_pdf)
        assert images == []


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------


class TestParsedDocument:
    def test_from_path(self, sample_pdf_path):
        parsed = ParsedDocument.from_path(sample_pdf_path, dpi=72)
        assert parsed.metadata.page_count == 2
        assert len(parsed.page_images) == 2
        assert len(parsed.page_texts) == 2
        assert len(parsed.scanned_flags) == 2
        # Both blank → both scanned
        assert parsed.scanned_flags == [True, True]
        assert not parsed.is_digital_born

    def test_is_digital_born_false_when_all_scanned(self, sample_pdf_path):
        parsed = ParsedDocument.from_path(sample_pdf_path)
        assert not parsed.is_digital_born

    def test_pdf_is_stored(self, sample_pdf_path):
        parsed = ParsedDocument.from_path(sample_pdf_path)
        assert parsed.pdf is not None
        assert len(parsed.pdf.pages) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_classify_empty_pdf(self):
        """PDF with no pages."""
        import pikepdf

        pdf = pikepdf.Pdf.new()
        flags, overall = classify_document(pdf)
        assert flags == []
        assert overall is False

    def test_parsed_document_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            ParsedDocument.from_path("/nonexistent/path.pdf")
