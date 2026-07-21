"""
PDF/UA-2 validation — run veraPDF and parse results.

Package:
    validator:   Run veraPDF CLI, parse output, categorize failures, auto-fix loop.
    report:      Generate remediation report JSON with confidence scores.
"""

from reach.validate.validator import VeraPDFValidator, ValidationResult, FailureCategory
from reach.validate.report import RemediationReport, ReportGenerator

__all__ = [
    "VeraPDFValidator",
    "ValidationResult",
    "FailureCategory",
    "RemediationReport",
    "ReportGenerator",
]
