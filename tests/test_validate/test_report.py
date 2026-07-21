"""
Tests for ReportGenerator — remediation report creation, saves, confidence scores.
"""
import json
from pathlib import Path

import pytest

from reach.validate.report import (
    FailureCategory,
    FixRecord,
    HumanReviewItem,
    RemediationReport,
    ReportGenerator,
    SectionConfidence,
)
from reach.validate.validator import ValidationFailure, ValidationResult


class TestFixRecord:
    """Tests for FixRecord."""

    def test_create_fix_record(self):
        record = FixRecord(
            component="construct",
            description="Font embedded: TimesNewRoman",
            status="fixed",
        )
        assert record.component == "construct"
        assert record.status == "fixed"

    def test_fix_record_partial(self):
        record = FixRecord(
            component="forms",
            description="3 of 5 form fields tagged",
            status="partial",
            detail="2 fields could not be identified",
        )
        assert record.status == "partial"


class TestSectionConfidence:
    """Tests for SectionConfidence."""

    def test_create_confidence(self):
        sc = SectionConfidence(
            section="tag_tree",
            score=0.85,
            notes=["Heading hierarchy inferred correctly"],
        )
        assert sc.section == "tag_tree"
        assert sc.score == 0.85
        assert len(sc.notes) == 1

    def test_confidence_clamped(self):
        # Clamping is done in set_confidence
        pass


class TestHumanReviewItem:
    """Tests for HumanReviewItem."""

    def test_create_review_item(self):
        item = HumanReviewItem(
            category="alt_text",
            description="Figure on page 3 missing alt text",
            page=3,
            priority="high",
        )
        assert item.category == "alt_text"
        assert item.page == 3
        assert item.priority == "high"


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_record_fix(self, generator):
        generator.record_fix("writer", "Font embedded", "fixed", "Font: Helvetica")
        assert len(generator._fixes) == 1
        assert generator._fixes[0].component == "writer"

    def test_set_confidence(self, generator):
        generator.set_confidence("tag_tree", 0.85, ["Good heading detection"])
        assert len(generator._confidences) == 1
        assert generator._confidences[0].score == 0.85

    def test_set_confidence_clamped(self, generator):
        generator.set_confidence("alt_text", 1.5)  # Should clamp to 1.0
        assert generator._confidences[0].score == 1.0

        generator.set_confidence("fonts", -0.5)  # Should clamp to 0.0
        assert generator._confidences[1].score == 0.0

    def test_generate_pass_report(self, generator, tmp_path):
        """Test generating a report for a fully successful remediation."""
        generator.record_fix("extract", "Extraction complete", "fixed")
        generator.record_fix("construct", "PDF written", "fixed")
        generator.set_confidence("extraction", 0.90)
        generator.set_confidence("construction", 0.95)

        report = generator.generate(
            input_path=Path("/tmp/input.pdf"),
            output_path=Path("/tmp/output.pdf"),
            processing_time=5.2,
        )

        assert report.overall_status == "pass"
        assert report.total_fixes == 2
        assert report.overall_confidence == pytest.approx(0.925)
        assert report.processing_time_seconds == 5.2

    def test_generate_partial_pass_report(self, generator, tmp_path):
        """Test generating a report with partial failures."""
        generator.record_fix("extract", "Extraction", "fixed")
        generator.record_fix("forms", "Form fields", "partial", "2 fields not tagged")
        generator.set_confidence("extraction", 0.85)
        generator.set_confidence("forms", 0.60)

        # Create validation result with failures
        failures = [
            ValidationFailure(
                rule_id="6.21.1",
                category=FailureCategory.FORMS,
                description="Form field missing tag",
                auto_fixable=False,
            )
        ]
        by_category = {FailureCategory.FORMS: failures}
        val_result = ValidationResult(
            passed=False,
            is_compliant=False,
            failures=failures,
            by_category=by_category,
        )

        report = generator.generate(
            input_path=Path("/tmp/input.pdf"),
            output_path=Path("/tmp/output.pdf"),
            validation_result=val_result,
            processing_time=10.0,
        )

        assert report.overall_status == "partial_pass"
        assert report.validation_failures == 1
        assert len(report.review_queue) == 1
        assert report.review_queue[0].category == "forms"

    def test_generate_fail_report(self, generator, tmp_path):
        """Test generating a report for a failed remediation."""
        generator.record_fix("extract", "Extraction", "failed", "PDF encrypted")
        generator.set_confidence("extraction", 0.0)

        report = generator.generate(
            input_path=Path("/tmp/input.pdf"),
            output_path=Path("/tmp/output.pdf"),
        )

        assert report.overall_status == "fail"

    def test_save_report(self, generator, tmp_path):
        """Test saving a report to JSON."""
        generator.record_fix("writer", "Metadata set", "fixed")
        generator.set_confidence("metadata", 1.0)

        report = generator.generate(
            input_path=Path("/tmp/input.pdf"),
            output_path=Path("/tmp/output.pdf"),
        )

        report_path = tmp_path / "report.json"
        result = generator.save(report, report_path)

        assert result == report_path
        assert report_path.exists()

        # Verify JSON structure
        with open(report_path) as f:
            data = json.load(f)

        assert "document" in data
        assert "status" in data
        assert "fixes" in data
        assert "confidence" in data
        assert "human_review" in data
        assert "performance" in data
        assert data["status"]["overall"] == "pass"

    def test_suggest_action(self, generator):
        """Test action suggestions for each category."""
        for category in FailureCategory:
            action = generator._suggest_action(category, "test")
            assert isinstance(action, str)
            assert len(action) > 0

    def test_determine_priority(self, generator):
        """Test priority determination."""
        assert generator._determine_priority(FailureCategory.TAG_TREE) == "critical"
        assert generator._determine_priority(FailureCategory.STRUCTURE) == "critical"
        assert generator._determine_priority(FailureCategory.ALT_TEXT) == "high"
        assert generator._determine_priority(FailureCategory.METADATA) == "high"
        assert generator._determine_priority(FailureCategory.TABLE_STRUCTURE) == "medium"
        assert generator._determine_priority(FailureCategory.UNKNOWN) == "medium"


class TestRemediationReport:
    """Tests for the RemediationReport dataclass."""

    def test_default_report(self):
        report = RemediationReport(
            input_path="/tmp/in.pdf",
            output_path="/tmp/out.pdf",
            timestamp="2026-07-21T00:00:00Z",
            overall_status="pass",
        )
        assert report.total_fixes == 0
        assert report.overall_confidence == 0.0
        assert report.review_queue == []

    def test_report_to_dict_roundtrip(self, tmp_path):
        """Test that report data can be saved and reloaded."""
        generator = ReportGenerator()
        generator.record_fix("extract", "Done", "fixed")
        generator.set_confidence("extraction", 0.9)

        report = generator.generate(
            input_path=Path("/tmp/in.pdf"),
            output_path=Path("/tmp/out.pdf"),
            processing_time=3.5,
        )

        report_path = tmp_path / "report.json"
        generator.save(report, report_path)

        with open(report_path) as f:
            data = json.load(f)

        assert data["total_fixes"] == 1
        assert data["confidence"]["overall"] == 0.9
        assert data["performance"]["processing_time_seconds"] == 3.5
