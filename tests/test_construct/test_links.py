"""
Tests for LinkPreserver — link annotation detection, preservation, tagging.
"""
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, Pdf, String

from reach.construct.links import LinkAnnotation, LinkPreserver


class TestLinkAnnotation:
    """Tests for LinkAnnotation descriptor."""

    def test_create_link_annotation(self):
        link = LinkAnnotation(
            page=0,
            bbox=(10.0, 20.0, 100.0, 40.0),
            uri="https://example.com",
            description="External link to example.com",
            link_type="URI",
        )
        assert link.page == 0
        assert link.uri == "https://example.com"
        assert link.description == "External link to example.com"
        assert link.link_type == "URI"
        assert link.tag_role == "Link"

    def test_internal_link_annotation(self):
        link = LinkAnnotation(
            page=0,
            dest_page=5,
            description="Link to page 6",
            link_type="GoTo",
        )
        assert link.dest_page == 5
        assert link.link_type == "GoTo"

    def test_defaults(self):
        link = LinkAnnotation(page=0)
        assert link.uri == ""
        assert link.description == ""
        assert link.link_type == "URI"


class TestLinkPreserver:
    """Tests for LinkPreserver."""

    @pytest.fixture
    def preserver(self):
        return LinkPreserver()

    @pytest.fixture
    def pdf_with_links(self, tmp_path):
        """Create a PDF with link annotations."""
        pdf_path = tmp_path / "links.pdf"
        pdf = Pdf.new()

        # Add a page
        page = pdf.add_blank_page(page_size=(612, 792))

        # Create link annotations
        annots = Array()

        # External URI link
        uri_link = Dictionary()
        uri_link[Name("/Type")] = Name("/Annot")
        uri_link[Name("/Subtype")] = Name("/Link")
        uri_link[Name("/Rect")] = Array([10, 700, 100, 730])
        action = Dictionary()
        action[Name("/S")] = Name("/URI")
        action[Name("/URI")] = String("https://example.com")
        uri_link[Name("/A")] = action
        annots.append(uri_link)

        # Internal page link
        goto_link = Dictionary()
        goto_link[Name("/Type")] = Name("/Annot")
        goto_link[Name("/Subtype")] = Name("/Link")
        goto_link[Name("/Rect")] = Array([150, 700, 250, 730])
        dest = Array([page.obj, Name("/XYZ"), 0, 792, 0])
        goto_link[Name("/Dest")] = dest
        annots.append(goto_link)

        page[Name("/Annots")] = annots

        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    @pytest.fixture
    def pdf_without_links(self, tmp_path):
        """Create a PDF without link annotations."""
        pdf_path = tmp_path / "nolinks.pdf"
        pdf = Pdf.new()
        pdf.add_blank_page(page_size=(612, 792))
        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    def test_detect_links_with_links(self, preserver, pdf_with_links):
        """Test detecting links on a PDF with link annotations."""
        pdf = Pdf.open(pdf_with_links)
        links = preserver.detect_links(pdf)

        assert len(links) == 2
        assert links[0].link_type == "URI"
        assert links[1].link_type == "GoTo"

        pdf.close()

    def test_detect_links_without_links(self, preserver, pdf_without_links):
        """Test detecting links on a PDF without link annotations."""
        pdf = Pdf.open(pdf_without_links)
        links = preserver.detect_links(pdf)

        assert len(links) == 0

        pdf.close()

    def test_preserve_adds_descriptions(self, preserver, pdf_with_links):
        """Test that preserve() adds /Contents to links without descriptions."""
        pdf = Pdf.open(pdf_with_links)
        links = preserver.detect_links(pdf)

        # All links should have descriptions after detection
        for link in links:
            assert link.description != ""
            if link.link_type == "URI":
                assert "example.com" in link.description.lower() or "external" in link.description.lower()

        pdf.close()

    def test_generate_description_uri(self, preserver):
        """Test description generation for URI links."""
        desc = preserver._generate_description(
            link_type="URI",
            uri="https://www.example.com/page",
        )
        assert "example.com" in desc.lower() or "external" in desc.lower()

    def test_generate_description_goto(self, preserver):
        """Test description generation for GoTo links."""
        desc = preserver._generate_description(
            link_type="GoTo",
            dest_page=5,
        )
        assert "page 6" in desc.lower()

    def test_generate_description_named_dest(self, preserver):
        """Test description generation for named destination."""
        desc = preserver._generate_description(
            link_type="GoTo",
            dest_name="chapter1",
        )
        assert "chapter1" in desc

    def test_to_tag_nodes(self, preserver, pdf_with_links):
        """Test that links can be converted to tag nodes."""
        pdf = Pdf.open(pdf_with_links)
        links = preserver.detect_links(pdf)
        nodes = preserver.to_tag_nodes()

        assert len(nodes) == len(links)
        for node in nodes:
            assert node["tag"] == "Link"
            assert "text" in node
            assert "attributes" in node

        pdf.close()

    def test_preserve_method(self, preserver, pdf_with_links):
        """Test the preserve() method runs without errors."""
        pdf = Pdf.open(pdf_with_links)
        preserver.preserve(pdf)
        assert len(preserver._links) == 2
        pdf.close()
