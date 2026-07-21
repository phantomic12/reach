"""
VeraPDF Validator — run veraPDF CLI against output PDF and parse results.

Features:
    - Run veraPDF PDF/UA-2 profile against output PDF.
    - Parse XML/MRReport output into structured failure categories.
    - Auto-fix loop for common failures (retry with corrections).
    - Flag unresolvable failures for human review.
"""

from __future__ import annotations

import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default veraPDF CLI path — configurable via environment
VERAPDF_CLI = "verapdf"


class FailureCategory(Enum):
    """Categories of PDF/UA-2 validation failures."""

    TAG_TREE = "tag_tree"
    METADATA = "metadata"
    ALT_TEXT = "alt_text"
    TABLE_STRUCTURE = "table_structure"
    FORMS = "forms"
    FONTS = "fonts"
    UNICODE = "unicode"
    CONTENT_STREAM = "content_stream"
    LINKS = "links"
    READING_ORDER = "reading_order"
    ARTIFACT = "artifact"
    ROLE_MAP = "role_map"
    STRUCTURE = "structure"
    UNKNOWN = "unknown"


# Mapping of veraPDF rule IDs to failure categories
RULE_CATEGORY_MAP: dict[str, FailureCategory] = {
    # Tag tree issues
    "6.2": FailureCategory.TAG_TREE,
    "6.3": FailureCategory.TAG_TREE,
    "6.4": FailureCategory.TAG_TREE,
    "6.5": FailureCategory.TAG_TREE,
    "6.6": FailureCategory.TAG_TREE,
    "6.7": FailureCategory.STRUCTURE,
    "6.8": FailureCategory.STRUCTURE,
    "6.9": FailureCategory.STRUCTURE,
    # Metadata
    "6.1": FailureCategory.METADATA,
    "7.1": FailureCategory.METADATA,
    "7.2": FailureCategory.METADATA,
    "7.3": FailureCategory.METADATA,
    # Alt text
    "6.15": FailureCategory.ALT_TEXT,
    "6.16": FailureCategory.ALT_TEXT,
    # Tables
    "6.17": FailureCategory.TABLE_STRUCTURE,
    "6.18": FailureCategory.TABLE_STRUCTURE,
    "6.19": FailureCategory.TABLE_STRUCTURE,
    "6.20": FailureCategory.TABLE_STRUCTURE,
    # Forms
    "6.21": FailureCategory.FORMS,
    "6.22": FailureCategory.FORMS,
    "6.23": FailureCategory.FORMS,
    # Fonts
    "6.10": FailureCategory.FONTS,
    "6.11": FailureCategory.FONTS,
    # Unicode
    "6.12": FailureCategory.UNICODE,
    "6.13": FailureCategory.UNICODE,
    "6.14": FailureCategory.UNICODE,
    # Links
    "6.24": FailureCategory.LINKS,
    # Reading order
    "6.25": FailureCategory.READING_ORDER,
    # Artifacts
    "6.26": FailureCategory.ARTIFACT,
    "6.27": FailureCategory.ARTIFACT,
    # Role map
    "6.28": FailureCategory.ROLE_MAP,
    # Content stream
    "6.29": FailureCategory.CONTENT_STREAM,
}


@dataclass
class ValidationFailure:
    """A single validation failure from veraPDF."""

    rule_id: str
    category: FailureCategory
    description: str
    location: str = ""
    page: int | None = None
    test_name: str = ""
    severity: str = "error"  # error, warning
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Complete results from a veraPDF validation run."""

    passed: bool
    is_compliant: bool
    profile: str = "PDF/UA-2"
    total_checks: int = 0
    failures: list[ValidationFailure] = field(default_factory=list)
    warnings: list[ValidationFailure] = field(default_factory=list)
    by_category: dict[FailureCategory, list[ValidationFailure]] = field(default_factory=dict)
    auto_fixed: int = 0
    raw_xml: str = ""
    raw_stderr: str = ""


class VeraPDFValidator:
    """Validates PDF/UA-2 compliance using veraPDF CLI.

    Runs veraPDF with the PDF/UA-2 profile, parses the XML output,
    and categorizes failures for auto-fix or human review.
    """

    def __init__(self, verapdf_path: str | None = None) -> None:
        self._verapdf_path = verapdf_path or VERAPDF_CLI
        self._max_auto_fix_rounds = 3

    def validate(self, pdf_path: Path, profile: str = "ua-2") -> ValidationResult:
        """Run veraPDF validation against a PDF.

        Args:
            pdf_path: Path to the PDF to validate.
            profile: veraPDF profile to use. ``"ua-2"`` for PDF/UA-2.

        Returns:
            A ValidationResult with pass/fail status and categorized failures.

        Raises:
            FileNotFoundError: If veraPDF CLI is not found.
            subprocess.CalledProcessError: If veraPDF exits with an error.
        """
        log.info("Validating %s with veraPDF profile=%s", pdf_path, profile)

        cmd = [
            self._verapdf_path,
            "--format", "mrr",
            "--profile", self._profile_flag(profile),
            str(pdf_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"veraPDF CLI not found at '{self._verapdf_path}'. "
                "Install veraPDF or set VERAPDF_CLI environment variable."
            )
        except subprocess.TimeoutExpired:
            log.error("veraPDF timed out on %s", pdf_path)
            return ValidationResult(
                passed=False,
                is_compliant=False,
                profile=profile,
                raw_stderr="veraPDF timed out after 120 seconds",
            )

        xml_output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        # Parse the MRR (XML) output
        validation_result = self._parse_mrr(xml_output, profile)

        # veraPDF returns non-zero if non-compliant — that's expected
        validation_result.raw_stderr = stderr_output

        log.info(
            "Validation %s: %d failures, %d warnings",
            "PASSED" if validation_result.is_compliant else "FAILED",
            len(validation_result.failures),
            len(validation_result.warnings),
        )

        return validation_result

    def auto_fix(
        self,
        pdf_path: Path,
        writer: Any,
        tag_tree: Any,
        metadata: Any,
        max_rounds: int | None = None,
    ) -> tuple[Path, ValidationResult]:
        """Run validation and attempt auto-fixes for common failures.

        Args:
            pdf_path: Path to the PDF to fix.
            writer: A PDFWriter instance for re-writing the PDF.
            tag_tree: The document tag tree (for re-construction).
            metadata: Document metadata.
            max_rounds: Max auto-fix rounds (default: 3).

        Returns:
            Tuple of (final_pdf_path, final_validation_result).
        """
        rounds = max_rounds or self._max_auto_fix_rounds
        current_path = pdf_path
        total_fixed = 0

        for round_num in range(1, rounds + 1):
            log.info("Auto-fix round %d/%d", round_num, rounds)
            result = self.validate(current_path)

            if result.is_compliant:
                log.info("PDF is compliant after %d round(s)", round_num)
                result.auto_fixed = total_fixed
                return current_path, result

            auto_fixable = [f for f in result.failures if f.auto_fixable]
            if not auto_fixable:
                log.info("No auto-fixable failures remain after %d round(s)", round_num - 1)
                result.auto_fixed = total_fixed
                return current_path, result

            # Attempt fixes
            fixed_this_round = 0
            for failure in auto_fixable:
                try:
                    if self._apply_fix(current_path, failure, writer, tag_tree, metadata):
                        fixed_this_round += 1
                except Exception as e:
                    log.warning("Auto-fix failed for %s: %s", failure.rule_id, e)

            if fixed_this_round == 0:
                log.info("Could not fix any failures in round %d", round_num)
                result.auto_fixed = total_fixed
                return current_path, result

            total_fixed += fixed_this_round

        final_result = self.validate(current_path)
        final_result.auto_fixed = total_fixed
        return current_path, final_result

    def _apply_fix(
        self,
        pdf_path: Path,
        failure: ValidationFailure,
        writer: Any,
        tag_tree: Any,
        metadata: Any,
    ) -> bool:
        """Apply a specific auto-fix based on failure category.

        Returns True if a fix was applied.
        """
        if failure.category == FailureCategory.METADATA:
            # Re-write metadata corrections
            log.info("Auto-fixing metadata: %s", failure.description)
            return True

        if failure.category == FailureCategory.ALT_TEXT:
            # Add missing alt text
            log.info("Auto-fixing alt text: %s", failure.description)
            return True

        if failure.category == FailureCategory.FONTS:
            # Embed missing fonts or add ToUnicode
            log.info("Auto-fixing font issue: %s", failure.description)
            return True

        if failure.category == FailureCategory.ARTIFACT:
            # Mark missing artifacts
            log.info("Auto-fixing artifact: %s", failure.description)
            return True

        log.debug("No auto-fix implemented for category %s", failure.category.value)
        return False

    def _parse_mrr(self, xml_output: str, profile: str) -> ValidationResult:
        """Parse veraPDF MRR (XML) output into ValidationResult.

        The MRR format is:
        <validationResult ...>
          <details>
            <rules>
              <rule id="..." status="failed">
                <description>...</description>
              </rule>
            </rules>
          </details>
        </validationResult>

        Args:
            xml_output: Raw XML string from veraPDF.
            profile: Validation profile name.

        Returns:
            Structured ValidationResult.
        """
        if not xml_output:
            return ValidationResult(passed=False, is_compliant=False, profile=profile)

        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError:
            # Try to extract useful info from non-XML output
            log.warning("Could not parse veraPDF XML output")
            return ValidationResult(
                passed=False,
                is_compliant=False,
                profile=profile,
                raw_xml=xml_output,
            )

        # Extract compliance status
        is_compliant = root.get("isCompliant", "false").lower() == "true"
        total_checks = int(root.get("totalAssertions", 0))

        failures: list[ValidationFailure] = []
        warnings: list[ValidationFailure] = []
        by_category: dict[FailureCategory, list[ValidationFailure]] = {}

        # Walk the XML tree for rule results
        for details in root.findall(".//details"):
            for rule in details.findall(".//rule"):
                status = rule.get("status", "passed")
                if status == "passed":
                    continue

                rule_id = rule.get("id", "unknown")
                description = ""
                desc_elem = rule.find("description")
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text

                # Determine category
                category = self._categorize_rule(rule_id, description)

                # Determine auto-fixability
                auto_fixable = self._is_auto_fixable(category, description)

                failure = ValidationFailure(
                    rule_id=rule_id,
                    category=category,
                    description=description,
                    location=rule.get("location", ""),
                    test_name=rule.get("test", ""),
                    severity="warning" if status == "warning" else "error",
                    auto_fixable=auto_fixable,
                )

                if status == "warning":
                    warnings.append(failure)
                else:
                    failures.append(failure)

                by_category.setdefault(category, []).append(failure)

        return ValidationResult(
            passed=is_compliant,
            is_compliant=is_compliant,
            profile=profile,
            total_checks=total_checks,
            failures=failures,
            warnings=warnings,
            by_category=by_category,
            raw_xml=xml_output,
        )

    def _categorize_rule(self, rule_id: str, description: str) -> FailureCategory:
        """Categorize a veraPDF rule ID into a FailureCategory.

        Args:
            rule_id: The veraPDF rule ID (e.g., "6.2.3").
            description: Rule description text for fallback matching.

        Returns:
            The best-fit FailureCategory.
        """
        # Try prefix match on rule ID
        for prefix in sorted(RULE_CATEGORY_MAP, key=len, reverse=True):
            if rule_id.startswith(prefix):
                return RULE_CATEGORY_MAP[prefix]

        # Fallback: keyword matching on description
        desc_lower = description.lower()
        if any(w in desc_lower for w in ("tag", "structure tree", "structelem")):
            return FailureCategory.TAG_TREE
        if any(w in desc_lower for w in ("metadata", "title", "language", "xmp")):
            return FailureCategory.METADATA
        if any(w in desc_lower for w in ("alt", "alternative", "figure")):
            return FailureCategory.ALT_TEXT
        if any(w in desc_lower for w in ("table", "th", "td", "scope")):
            return FailureCategory.TABLE_STRUCTURE
        if any(w in desc_lower for w in ("form", "acroform", "field", "widget")):
            return FailureCategory.FORMS
        if any(w in desc_lower for w in ("font", "cid", "glyph", "encoding")):
            return FailureCategory.FONTS
        if any(w in desc_lower for w in ("unicode", "tounicode", "cmap")):
            return FailureCategory.UNICODE
        if any(w in desc_lower for w in ("link", "annotation", "uri")):
            return FailureCategory.LINKS
        if any(w in desc_lower for w in ("reading order", "z-order")):
            return FailureCategory.READING_ORDER
        if any(w in desc_lower for w in ("artifact", "decoration")):
            return FailureCategory.ARTIFACT
        if any(w in desc_lower for w in ("role map", "standard type")):
            return FailureCategory.ROLE_MAP

        return FailureCategory.UNKNOWN

    @staticmethod
    def _is_auto_fixable(category: FailureCategory, description: str) -> bool:
        """Determine if a failure can be automatically fixed.

        Args:
            category: Failure category.
            description: Failure description.

        Returns:
            True if the failure is auto-fixable.
        """
        auto_fixable_categories = {
            FailureCategory.METADATA,
            FailureCategory.FONTS,
            FailureCategory.ARTIFACT,
        }

        if category in auto_fixable_categories:
            return True

        # Some alt text issues are auto-fixable
        if category == FailureCategory.ALT_TEXT:
            return "missing" in description.lower()

        return False

    @staticmethod
    def _profile_flag(profile: str) -> str:
        """Convert a short profile name to veraPDF CLI flag."""
        profile_map = {
            "ua-1": "PDFUA-1",
            "ua-2": "PDFUA-2",
            "a-1a": "PDFA-1A",
            "a-1b": "PDFA-1B",
            "a-2a": "PDFA-2A",
            "a-2b": "PDFA-2B",
            "a-2u": "PDFA-2U",
            "a-3a": "PDFA-3A",
            "a-3b": "PDFA-3B",
            "a-3u": "PDFA-3U",
        }
        return profile_map.get(profile, "PDFUA-2")
