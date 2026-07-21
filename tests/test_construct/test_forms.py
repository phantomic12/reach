"""
Tests for FormRemediator — AcroForm detection, tagging, tooltips, tab order.
"""
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, Pdf, String

from reach.construct.forms import FormField, FormRemediator, FIELD_TYPE_ROLE_MAP


class TestFormField:
    """Tests for FormField descriptor."""

    def test_create_form_field(self):
        field = FormField(
            name="text1",
            field_type="Tx",
            tooltip="Enter your name",
            page=0,
            bbox=(10.0, 20.0, 200.0, 40.0),
            tab_order=0,
        )
        assert field.name == "text1"
        assert field.field_type == "Tx"
        assert field.tooltip == "Enter your name"
        assert field.page == 0
        assert field.bbox == (10.0, 20.0, 200.0, 40.0)
        assert field.tab_order == 0
        assert field.role == "Form"

    def test_form_field_defaults(self):
        field = FormField(name="btn1", field_type="Btn")
        assert field.tooltip == ""
        assert field.page == 0
        assert field.bbox is None
        assert field.tab_order == 0


class TestFieldTypeRoleMap:
    """Tests for field type → role mapping."""

    def test_text_field_role(self):
        assert FIELD_TYPE_ROLE_MAP["Tx"] == "Form"

    def test_button_field_role(self):
        assert FIELD_TYPE_ROLE_MAP["Btn"] == "Form"

    def test_choice_field_role(self):
        assert FIELD_TYPE_ROLE_MAP["Ch"] == "Form"

    def test_signature_field_role(self):
        assert FIELD_TYPE_ROLE_MAP["Sig"] == "Form"


class TestFormRemediator:
    """Tests for FormRemediator."""

    @pytest.fixture
    def remediator(self):
        return FormRemediator()

    @pytest.fixture
    def pdf_with_form(self, tmp_path):
        """Create a PDF with AcroForm fields."""
        pdf_path = tmp_path / "form.pdf"
        pdf = Pdf.new()

        # Add a page
        page = pdf.add_blank_page(page_size=(612, 792))

        # Create AcroForm with fields
        acroform = Dictionary()
        acroform[Name("/Fields")] = Array()

        # Add a text field
        text_field = Dictionary()
        text_field[Name("/FT")] = Name("/Tx")
        text_field[Name("/T")] = String("name_field")
        text_field[Name("/Rect")] = Array([10, 700, 200, 730])
        text_field[Name("/P")] = page.obj
        acroform["/Fields"].append(text_field)

        # Add a button field
        btn_field = Dictionary()
        btn_field[Name("/FT")] = Name("/Btn")
        btn_field[Name("/T")] = String("submit_btn")
        btn_field[Name("/Rect")] = Array([250, 700, 350, 730])
        btn_field[Name("/P")] = page.obj
        acroform["/Fields"].append(btn_field)

        pdf.Root[Name("/AcroForm")] = acroform

        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    @pytest.fixture
    def pdf_without_form(self, tmp_path):
        """Create a PDF without AcroForm."""
        pdf_path = tmp_path / "noform.pdf"
        pdf = Pdf.new()
        pdf.add_blank_page(page_size=(612, 792))
        pdf.save(pdf_path)
        pdf.close()
        return pdf_path

    def test_detect_fields_with_form(self, remediator, pdf_with_form):
        """Test field detection on a PDF with AcroForm."""
        pdf = Pdf.open(pdf_with_form)
        fields = remediator.detect_fields(pdf)

        assert len(fields) == 2
        assert fields[0].name == "name_field"
        assert fields[0].field_type == "Tx"
        assert fields[1].name == "submit_btn"
        assert fields[1].field_type == "Btn"
        assert all(f.page == 0 for f in fields)

        pdf.close()

    def test_detect_fields_without_form(self, remediator, pdf_without_form):
        """Test field detection on a PDF without AcroForm."""
        pdf = Pdf.open(pdf_without_form)
        fields = remediator.detect_fields(pdf)

        assert len(fields) == 0

        pdf.close()

    def test_generate_tooltip(self, remediator):
        """Test default tooltip generation."""
        tooltip = remediator._generate_tooltip("Tx", "first_name")
        assert "Text field" in tooltip
        assert "first_name" in tooltip

        tooltip = remediator._generate_tooltip("Btn", "submit")
        assert "Button" in tooltip

    def test_set_tab_order(self, remediator, pdf_with_form):
        """Test that tab order is set on pages."""
        pdf = Pdf.open(pdf_with_form)
        remediator.remediate(pdf)

        for page in pdf.pages:
            assert "/Tabs" in page
            assert page["/Tabs"] == Name("/S")

        pdf.close()

    def test_to_tag_nodes(self, remediator, pdf_with_form):
        """Test that form fields can be converted to tag nodes."""
        pdf = Pdf.open(pdf_with_form)
        fields = remediator.detect_fields(pdf)
        nodes = remediator.to_tag_nodes()

        assert len(nodes) == len(fields)
        for node in nodes:
            assert node["tag"] == "Form"
            assert "attributes" in node
            assert "field_name" in node["attributes"]
            assert "field_type" in node["attributes"]

        pdf.close()
