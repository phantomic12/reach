"""
Output Report — generate remediation report JSON with confidence scores.

The report is the human-facing summary of what remediation did:
    - What was fixed and how.
    - What couldn't be fixed and why.
    - Per-section confidence scores.
    - veraPDF validation summary.
    - Human-review queue for failures requiring manual intervention.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reach.validate.validator import FailureCategory, ValidationFailure, ValidationResult

log = logging.getLogger(__name__)


@dataclass
class SectionConfidence:
    """Confidence score for a specific remediation section."""

    section: str  # e.g., "tag_tree", "reading_order", "alt_text"
    score: float  # 0.0 to 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class FixRecord:
    """Record of a specific fix applied during remediation."""

    component: str  # Which component applied the fix
    description: str
    status: str  # "fixed", "partial", "failed"
    detail: str = ""


@dataclass
class HumanReviewItem:
    """An item flagged for human review."""

    category: str
    description: str
    page: int | None = None
    location: str = ""
    suggested_action: str = ""
    priority: str = "medium"  # low, medium, high, critical


@dataclass
class RemediationReport:
    """Complete remediation report for a single document."""

    # Document info
    input_path: str
    output_path: str
    timestamp: str

    # Pipeline status
    overall_status: str  # "pass", "partial_pass", "fail"
    pipeline_version: str = "0.1.0"

    # Fixes applied
    fixes: list[FixRecord] = field(default_factory=list)
    total_fixes: int = 0

    # Confidence scores
    confidence_scores: list[SectionConfidence] = field(default_factory=list)
    overall_confidence: float = 0.0

    # Validation
    verapdf_compliant: bool = False
    validation_failures: int = 0
    validation_warnings: int = 0
    validation_by_category: dict[str, int] = field(default_factory=dict)

    # Human review
    review_queue: list[HumanReviewItem] = field(default_factory=list)

    # Performance
    processing_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0


class ReportGenerator:
    """Generates remediation reports and manages the human-review queue."""

    def __init__(self) -> None:
        self._fixes: list[FixRecord] = []
        self._confidences: list[SectionConfidence] = []

    def record_fix(
        self,
        component: str,
        description: str,
        status: str = "fixed",
        detail: str = "",
    ) -> None:
        """Record a fix applied during remediation.

        Args:
            component: Component that applied the fix.
            description: What was fixed.
            status: ``"fixed"``, ``"partial"``, or ``"failed"``.
            detail: Additional detail about the fix.
        """
        self._fixes.append(FixRecord(
            component=component,
            description=description,
            status=status,
            detail=detail,
        ))

    def set_confidence(self, section: str, score: float, notes: list[str] | None = None) -> None:
        """Set confidence score for a section.

        Args:
            section: Section name (tag_tree, reading_order, alt_text, forms, fonts, etc.).
            score: Confidence score from 0.0 to 1.0.
            notes: Optional notes explaining the score.
        """
        self._confidences.append(SectionConfidence(
            section=section,
            score=max(0.0, min(1.0, score)),
            notes=notes or [],
        ))

    def generate(
        self,
        input_path: Path,
        output_path: Path,
        validation_result: ValidationResult | None = None,
        processing_time: float = 0.0,
    ) -> RemediationReport:
        """Generate the full remediation report.

        Args:
            input_path: Source PDF path.
            output_path: Remediated PDF path.
            validation_result: Optional veraPDF validation result.
            processing_time: Total processing time in seconds.

        Returns:
            A complete RemediationReport.
        """
        # Determine overall status
        total_fixes = len(self._fixes)
        failed_fixes = sum(1 for f in self._fixes if f.status == "failed")
        partial_fixes = sum(1 for f in self._fixes if f.status == "partial")

        if validation_result and validation_result.is_compliant:
            overall_status = "pass"
        elif failed_fixes > 0:
            overall_status = "fail"
        elif partial_fixes > 0 or (validation_result and validation_result.failures):
            overall_status = "partial_pass"
        else:
            overall_status = "pass"

        # Overall confidence: average of section scores, weighted if some sections are more critical
        if self._confidences:
            overall_confidence = sum(c.score for c in self._confidences) / len(self._confidences)
        else:
            overall_confidence = 0.5  # Default: uncertain

        # Build validation summary
        validation_by_category: dict[str, int] = {}
        review_queue: list[HumanReviewItem] = []

        if validation_result:
            for category, failures in validation_result.by_category.items():
                validation_by_category[category.value] = len(failures)
                # Add non-auto-fixable failures to review queue
                for f in failures:
                    if not f.auto_fixable:
                        review_queue.append(HumanReviewItem(
                            category=category.value,
                            description=f.description,
                            page=f.page,
                            location=f.location,
                            suggested_action=self._suggest_action(category, f.description),
                            priority=self._determine_priority(category),
                        ))

        report = RemediationReport(
            input_path=str(input_path),
            output_path=str(output_path),
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status=overall_status,
            fixes=list(self._fixes),
            total_fixes=total_fixes,
            confidence_scores=list(self._confidences),
            overall_confidence=round(overall_confidence, 3),
            verapdf_compliant=validation_result.is_compliant if validation_result else False,
            validation_failures=len(validation_result.failures) if validation_result else 0,
            validation_warnings=len(validation_result.warnings) if validation_result else 0,
            validation_by_category=validation_by_category,
            review_queue=review_queue,
            processing_time_seconds=round(processing_time, 2),
        )

        log.info(
            "Report generated: status=%s, confidence=%.2f, fixes=%d, review_items=%d",
            overall_status,
            overall_confidence,
            total_fixes,
            len(review_queue),
        )

        return report

    def save(self, report: RemediationReport, output_path: Path) -> Path:
        """Save the report as JSON.

        Args:
            report: The RemediationReport to save.
            output_path: Where to write the JSON file.

        Returns:
            The output path.
        """
        data = self._report_to_dict(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Report saved to %s", output_path)
        return output_path

    def _report_to_dict(self, report: RemediationReport) -> dict[str, Any]:
        """Convert RemediationReport to a JSON-serializable dict."""
        return {
            "document": {
                "input_path": report.input_path,
                "output_path": report.output_path,
                "timestamp": report.timestamp,
            },
            "status": {
                "overall": report.overall_status,
                "pipeline_version": report.pipeline_version,
            },
            "fixes": [
                {
                    "component": f.component,
                    "description": f.description,
                    "status": f.status,
                    "detail": f.detail,
                }
                for f in report.fixes
            ],
            "total_fixes": report.total_fixes,
            "confidence": {
                "overall": report.overall_confidence,
                "sections": [
                    {
                        "section": c.section,
                        "score": c.score,
                        "notes": c.notes,
                    }
                    for c in report.confidence_scores
                ],
            },
            "validation": {
                "verapdf_compliant": report.verapdf_compliant,
                "failures": report.validation_failures,
                "warnings": report.validation_warnings,
                "by_category": report.validation_by_category,
            },
            "human_review": [
                {
                    "category": i.category,
                    "description": i.description,
                    "page": i.page,
                    "location": i.location,
                    "suggested_action": i.suggested_action,
                    "priority": i.priority,
                }
                for i in report.review_queue
            ],
            "performance": {
                "processing_time_seconds": report.processing_time_seconds,
                "peak_memory_mb": report.peak_memory_mb,
            },
        }

    @staticmethod
    def _suggest_action(category: FailureCategory, description: str) -> str:
        """Suggest a human action for a validation failure."""
        suggestions: dict[FailureCategory, str] = {
            FailureCategory.TAG_TREE: "Review the tag tree manually. Check heading hierarchy and element nesting.",
            FailureCategory.METADATA: "Check document metadata: title, language, and XMP information.",
            FailureCategory.ALT_TEXT: "Review and add alternative text for figures and formulas.",
            FailureCategory.TABLE_STRUCTURE: "Verify table headers, scope attributes, and cell structure.",
            FailureCategory.FORMS: "Manually tag form fields and add tooltip descriptions.",
            FailureCategory.FONTS: "Embed missing fonts or address font encoding issues.",
            FailureCategory.UNICODE: "Add ToUnicode CMap entries for affected fonts.",
            FailureCategory.CONTENT_STREAM: "Review content stream for proper marked content markers.",
            FailureCategory.LINKS: "Ensure all links have accessible descriptions.",
            FailureCategory.READING_ORDER: "Manually verify and correct reading order.",
            FailureCategory.ARTIFACT: "Mark decorative elements as artifacts.",
            FailureCategory.ROLE_MAP: "Ensure standard structure types are mapped correctly.",
            FailureCategory.STRUCTURE: "Review document structure for PDF/UA-2 compliance.",
            FailureCategory.UNKNOWN: "Investigate this validation failure manually.",
        }
        return suggestions.get(category, "Manual review required.")

    @staticmethod
    def _determine_priority(category: FailureCategory) -> str:
        """Determine review priority based on failure category."""
        critical = {
            FailureCategory.TAG_TREE,
            FailureCategory.STRUCTURE,
            FailureCategory.READING_ORDER,
        }
        high = {
            FailureCategory.ALT_TEXT,
            FailureCategory.METADATA,
            FailureCategory.UNICODE,
        }
        if category in critical:
            return "critical"
        if category in high:
            return "high"
        return "medium"
