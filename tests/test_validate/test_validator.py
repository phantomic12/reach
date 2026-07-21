"""
Tests for VeraPDFValidator — validation execution, XML parsing, failure categorization.
"""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from reach.validate.validator import (
    FailureCategory,
    RULE_CATEGORY_MAP,
    ValidationFailure,
    ValidationResult,
    VeraPDFValidator,
)


class TestValidationFailure:
    """Tests for ValidationFailure dataclass."""

    def test_create_failure(self):
        f = ValidationFailure(
            rule_id="6.2.1",
            category=FailureCategory.TAG_TREE,
            description="Missing structure element",
            auto_fixable=False,
        )
        assert f.rule_id == "6.2.1"
        assert f.category == FailureCategory.TAG_TREE
        assert f.description == "Missing structure element"
        assert f.auto_fixable is False

    def test_auto_fixable_failure(self):
        f = ValidationFailure(
            rule_id="6.10.1",
            category=FailureCategory.FONTS,
            description="Font not embedded",
            auto_fixable=True,
        )
        assert f.auto_fixable is True


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_pass_result(self):
        result = ValidationResult(passed=True, is_compliant=True)
        assert result.is_compliant is True
        assert len(result.failures) == 0

    def test_fail_result_with_failures(self):
        failures = [
            ValidationFailure(
                rule_id="6.2.1",
                category=FailureCategory.TAG_TREE,
                description="test",
            )
        ]
        result = ValidationResult(
            passed=False,
            is_compliant=False,
            failures=failures,
        )
        assert result.is_compliant is False
        assert len(result.failures) == 1


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_all_categories_exist(self):
        categories = list(FailureCategory)
        assert FailureCategory.TAG_TREE in categories
        assert FailureCategory.METADATA in categories
        assert FailureCategory.ALT_TEXT in categories
        assert FailureCategory.TABLE_STRUCTURE in categories
        assert FailureCategory.FORMS in categories
        assert FailureCategory.FONTS in categories
        assert FailureCategory.UNICODE in categories

    def test_category_values(self):
        assert FailureCategory.TAG_TREE.value == "tag_tree"
        assert FailureCategory.METADATA.value == "metadata"
        assert FailureCategory.ALT_TEXT.value == "alt_text"


class TestRuleCategoryMap:
    """Tests for RULE_CATEGORY_MAP."""

    def test_tag_tree_rules(self):
        assert RULE_CATEGORY_MAP["6.2"] == FailureCategory.TAG_TREE
        assert RULE_CATEGORY_MAP["6.3"] == FailureCategory.TAG_TREE

    def test_metadata_rules(self):
        assert RULE_CATEGORY_MAP["6.1"] == FailureCategory.METADATA
        assert RULE_CATEGORY_MAP["7.1"] == FailureCategory.METADATA

    def test_alt_text_rules(self):
        assert RULE_CATEGORY_MAP["6.15"] == FailureCategory.ALT_TEXT

    def test_table_rules(self):
        assert RULE_CATEGORY_MAP["6.17"] == FailureCategory.TABLE_STRUCTURE

    def test_form_rules(self):
        assert RULE_CATEGORY_MAP["6.21"] == FailureCategory.FORMS

    def test_font_rules(self):
        assert RULE_CATEGORY_MAP["6.10"] == FailureCategory.FONTS


class TestVeraPDFValidator:
    """Tests for VeraPDFValidator."""

    @pytest.fixture
    def validator(self):
        return VeraPDFValidator()

    def test_init_with_custom_path(self):
        v = VeraPDFValidator(verapdf_path="/custom/verapdf")
        assert v._verapdf_path == "/custom/verapdf"

    def test_profile_flag_ua2(self, validator):
        flag = validator._profile_flag("ua-2")
        assert flag == "PDFUA-2"

    def test_profile_flag_ua1(self, validator):
        flag = validator._profile_flag("ua-1")
        assert flag == "PDFUA-1"

    def test_profile_flag_default(self, validator):
        flag = validator._profile_flag("unknown")
        assert flag == "PDFUA-2"

    def test_parse_mrr_empty(self, validator):
        """Test parsing empty XML output."""
        result = validator._parse_mrr("", "ua-2")
        assert result.is_compliant is False
        assert result.passed is False

    def test_parse_mrr_compliant(self, validator):
        """Test parsing a compliant validation result."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<validationResult isCompliant="true" totalAssertions="42">
  <details>
    <rules>
      <rule id="6.2.1" status="passed">
        <description>Tagged PDF check passed</description>
      </rule>
    </rules>
  </details>
</validationResult>"""
        result = validator._parse_mrr(xml, "ua-2")
        assert result.is_compliant is True
        assert result.total_checks == 42
        assert len(result.failures) == 0

    def test_parse_mrr_failed(self, validator):
        """Test parsing a failed validation result."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<validationResult isCompliant="false" totalAssertions="50">
  <details>
    <rules>
      <rule id="6.2.1" status="failed">
        <description>Structure tree missing Document element</description>
      </rule>
      <rule id="6.10.1" status="failed">
        <description>Font not embedded: Helvetica</description>
      </rule>
      <rule id="7.1" status="warning">
        <description>DisplayDocTitle not set</description>
      </rule>
    </rules>
  </details>
</validationResult>"""
        result = validator._parse_mrr(xml, "ua-2")
        assert result.is_compliant is False
        assert result.total_checks == 50
        assert len(result.failures) == 2
        assert len(result.warnings) == 1

    def test_parse_mrr_categorizes_by_category(self, validator):
        """Test that failures are categorized correctly."""
        xml = """<?xml version="1.0" encoding="utf-8"?>
<validationResult isCompliant="false" totalAssertions="10">
  <details>
    <rules>
      <rule id="6.15.1" status="failed">
        <description>Figure missing alt text</description>
      </rule>
      <rule id="6.21.1" status="failed">
        <description>Form field not tagged</description>
      </rule>
      <rule id="6.10.1" status="failed">
        <description>Font not embedded</description>
      </rule>
    </rules>
  </details>
</validationResult>"""
        result = validator._parse_mrr(xml, "ua-2")
        assert FailureCategory.ALT_TEXT in result.by_category
        assert FailureCategory.FORMS in result.by_category
        assert FailureCategory.FONTS in result.by_category
        assert len(result.by_category[FailureCategory.ALT_TEXT]) == 1

    def test_parse_mrr_invalid_xml(self, validator):
        """Test parsing invalid XML gracefully."""
        result = validator._parse_mrr("not valid xml at all", "ua-2")
        assert result.is_compliant is False
        # Should not raise, should return a result with empty failures

    def test_categorize_rule_by_id(self, validator):
        """Test rule categorization by ID."""
        assert validator._categorize_rule("6.2.3", "") == FailureCategory.TAG_TREE
        assert validator._categorize_rule("7.1", "") == FailureCategory.METADATA
        assert validator._categorize_rule("6.15", "") == FailureCategory.ALT_TEXT
        assert validator._categorize_rule("6.17.2", "") == FailureCategory.TABLE_STRUCTURE
        assert validator._categorize_rule("6.21", "") == FailureCategory.FORMS
        assert validator._categorize_rule("6.10", "") == FailureCategory.FONTS

    def test_categorize_rule_by_description(self, validator):
        """Test rule categorization by description fallback."""
        assert validator._categorize_rule("unknown.1", "font encoding error") == FailureCategory.FONTS
        assert validator._categorize_rule("unknown.1", "table structure invalid") == FailureCategory.TABLE_STRUCTURE
        assert validator._categorize_rule("unknown.1", "link annotation missing") == FailureCategory.LINKS
        assert validator._categorize_rule("unknown.1", "artifact declaration") == FailureCategory.ARTIFACT

    def test_is_auto_fixable(self, validator):
        """Test auto-fixability determination."""
        assert validator._is_auto_fixable(FailureCategory.METADATA, "metadata issue") is True
        assert validator._is_auto_fixable(FailureCategory.FONTS, "font not embedded") is True
        assert validator._is_auto_fixable(FailureCategory.ARTIFACT, "missing artifact") is True
        assert validator._is_auto_fixable(FailureCategory.TAG_TREE, "structure issue") is False
        assert validator._is_auto_fixable(FailureCategory.TABLE_STRUCTURE, "table issue") is False

    @patch("subprocess.run")
    def test_validate_success(self, mock_run, validator, tmp_path):
        """Test validate with mocked veraPDF success."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='<validationResult isCompliant="true" totalAssertions="42"/>',
            stderr="",
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = validator.validate(pdf_path)
        assert result.is_compliant is True

    @patch("subprocess.run")
    def test_validate_failure(self, mock_run, validator, tmp_path):
        """Test validate with mocked veraPDF failure."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout='<validationResult isCompliant="false" totalAssertions="50"/>',
            stderr="",
        )
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = validator.validate(pdf_path)
        assert result.is_compliant is False

    @patch("subprocess.run")
    def test_validate_verapdf_not_found(self, mock_run, validator, tmp_path):
        """Test validate when veraPDF CLI is not available."""
        mock_run.side_effect = FileNotFoundError("verapdf not found")
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(FileNotFoundError, match="veraPDF CLI not found"):
            validator.validate(pdf_path)

    @patch("subprocess.run")
    def test_validate_timeout(self, mock_run, validator, tmp_path):
        """Test validate with timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["verapdf"], timeout=120)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        result = validator.validate(pdf_path)
        assert result.is_compliant is False
        assert "timed out" in result.raw_stderr.lower()
