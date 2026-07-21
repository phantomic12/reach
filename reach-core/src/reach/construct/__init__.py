"""
PDF construction — Phase 3 of the Reach pipeline.

Converts a tag tree and extracted content into a valid PDF/UA-2 document.

Modules:
    writer:   Inject tag tree, embed fonts, set metadata via pikepdf.
    forms:    Remediate AcroForm fields with proper tagging and tooltips.
    links:    Preserve and tag internal/external link annotations.
    pipeline: Orchestrate the full extraction → build → construct → validate pipeline.
"""

from reach.construct.writer import PDFWriter
from reach.construct.forms import FormRemediator
from reach.construct.links import LinkPreserver
from reach.construct.pipeline import RemediationPipeline

__all__ = ["PDFWriter", "FormRemediator", "LinkPreserver", "RemediationPipeline"]
