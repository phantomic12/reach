"""Tests for CLI interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from reach.cli import app

runner = CliRunner()


class TestCLIVersion:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Reach" in result.stdout

    def test_download_models_dry(self):
        """Test that download-models command exists and accepts flags."""
        result = runner.invoke(app, ["download-models", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.stdout

    def test_remediate_help(self):
        result = runner.invoke(app, ["remediate", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout
        assert "--report" in result.stdout

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True returns 0 and shows help
        assert "remediate" in result.stdout.lower() or "Commands" in result.stdout


class TestExtractCommand:
    """Tests for `reach extract` Phase 1 command."""

    def test_extract_help(self):
        """Verify the extract command exists and shows options."""
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout
        assert "--dpi" in result.stdout
        assert "--no-models" in result.stdout

    def test_extract_nonexistent_file(self):
        """Should exit with error for missing file."""
        result = runner.invoke(app, ["extract", "/nonexistent/file.pdf"])
        assert result.exit_code == 1
        assert "not found" in (result.stdout + result.stderr).lower()

    def test_extract_existing_file(self, tmp_path):
        """Full extract command on a real PDF (with mocks)."""
        from pypdf import PdfWriter

        # Create a real PDF
        pdf_path = tmp_path / "test.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_metadata({"/Title": "CLI Test"})
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # We need to mock the ExtractionPipeline to avoid loading models
        output_path = tmp_path / "output.json"

        with patch("reach.cli.ExtractionPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            # Mock pipeline.run() to return a minimal valid doc
            from reach.extract.models import (
                BBox,
                BlockType,
                DocumentMetadata,
                ExtractedDocument,
                Page,
                PageMetadata,
                TextBlock,
            )

            bbox = BBox(x1=0, y1=0, x2=100, y2=50)
            block = TextBlock(
                block_type=BlockType.SECTION_HEADER,
                bbox=bbox,
                text="CLI Test Title",
                reading_order=0,
                confidence=0.95,
            )
            meta = PageMetadata(page_number=1, width_pt=612, height_pt=792)
            page = Page(metadata=meta, blocks=[block])
            doc = ExtractedDocument(
                metadata=DocumentMetadata(title="CLI Test", page_count=1),
                pages=[page],
            )

            mock_pipeline.run.return_value = doc

            result = runner.invoke(
                app,
                ["extract", str(pdf_path), "--output", str(output_path), "--no-ocr"],
            )

            # Should succeed
            assert result.exit_code == 0
            assert mock_pipeline.run.called

            # Since doc.write_json is called on the actual doc (not mock),
            # the output file should be created
            assert output_path.exists()

    def test_extract_with_dpi_flag(self):
        """Verify --dpi flag is accepted."""
        result = runner.invoke(app, ["extract", "--dpi", "300", "--help"])
        assert result.exit_code == 0

    def test_extract_with_backend_flag(self):
        """Verify --backend flag is accepted."""
        result = runner.invoke(app, ["extract", "--backend", "cpu", "--help"])
        assert result.exit_code == 0
