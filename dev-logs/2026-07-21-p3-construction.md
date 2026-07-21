# Task: Phase 3 — PDF Construction & Validation
**Date**: 2026-07-21
**Phase**: 3
**Agent**: Hermes Agent (ocg/deepseek-v4-pro)
**Status**: complete

## What was done

### construct/ package (PDF construction)
- **writer.py**: PDFWriter class using pikepdf
  - Tag tree injection into PDF structure tree with /StructTreeRoot, /ParentTree, /RoleMap
  - Font embedding detection and ToUnicode CMap generation for all fonts
  - Alt text attachment to Figure/Formula tags via /Alt entries
  - Document metadata: title, language, author, DisplayDocTitle flag
  - Content stream rewriting infrastructure (BMC/EMC markers)
  - Artifact marking for page numbers, headers, footers
  - TagNode and DocumentMetadata dataclasses with full type hints
  - WriterConfig for controlling embedding, compression, PDF version

- **forms.py**: FormRemediator class
  - AcroForm field detection with type inference (Btn, Tx, Ch, Sig)
  - Field-to-role mapping: all fields → Form structure element
  - Automatic tooltip (/TU) generation when missing
  - Tab order set to /S (structure order) on all pages
  - to_tag_nodes() for integrating forms into the tag tree

- **links.py**: LinkPreserver class
  - Link annotation detection: URI, GoTo, GoToR, Launch types
  - Internal link destination page resolution
  - /Contents (accessible description) generation for all links
  - Human-readable descriptions: domain for external, page number for internal
  - to_tag_nodes() for integrating links into the tag tree

- **pipeline.py**: RemediationPipeline class
  - Full 5-stage pipeline: extract → build tags → construct → validate → report
  - Minimal extraction from pikepdf (placeholder for Phase 1)
  - Auto-fix loop with veraPDF (up to 3 rounds)
  - Metadata override support via CLI flags
  - Progress logging per stage

### validate/ package
- **validator.py**: VeraPDFValidator class
  - Subprocess execution of veraPDF CLI with MRR (XML) output format
  - XML/MRR parsing into structured ValidationResult
  - FailureCategory enum: 13 categories (tag_tree, metadata, alt_text, etc.)
  - Rule ID → category mapping for veraPDF rule prefixes
  - Description-based fallback categorization
  - Auto-fixability determination per failure
  - Auto-fix loop with configurable max rounds (default 3)
  - Profile flag mapping: ua-2 → PDFUA-2, ua-1 → PDFUA-1, etc.

- **report.py**: ReportGenerator class
  - RemediationReport dataclass with full remediation summary
  - Per-section confidence scores (SectionConfidence)
  - FixRecord tracking per component with fixed/partial/failed status
  - HumanReviewItem queue for non-auto-fixable failures
  - Priority determination (critical/high/medium) per category
  - Suggested human actions per failure category
  - JSON serialization with structured output

### CLI updates
- **cli.py**: Full `remediate` command implementation
  - Flags: --output, --report, --no-validate, --no-auto-fix, --title, --lang, --verbose
  - Error handling for missing files and pipeline failures
  - User-friendly output: status, time, fixes, confidence, veraPDF result
  - Human-review queue display (up to 5 items)

### Test files created
- tests/test_construct/__init__.py
- tests/test_construct/test_writer.py — 16 tests
- tests/test_construct/test_forms.py — 9 tests
- tests/test_construct/test_links.py — 10 tests
- tests/test_construct/test_pipeline.py — 10 tests
- tests/test_validate/__init__.py
- tests/test_validate/test_validator.py — 17 tests
- tests/test_validate/test_report.py — 14 tests

### Design decisions
1. **pikepdf for PDF manipulation**: Chosen per ADR-0001 (PDF 2.0 support). pikepdf wraps QPDF which handles PDF 2.0 natively.
2. **TagNode as intermediate representation**: Decouples tag tree construction from PDF injection, allowing Phase 2 to produce trees independently.
3. **Pluggable extraction**: pipeline._extract() is a placeholder that reads basic structure via pikepdf. Will delegate to Phase 1 modules when ready.
4. **Auto-fix loop architecture**: Validator returns auto_fixable flag per failure; pipeline calls auto_fix() which re-runs veraPDF and re-writes the PDF.
5. **Report as first-class output**: The JSON report is the canonical record of what happened, not just a log. Designed for integration into web review UI (Phase 4).

## Testing
- Total tests: 76 across 6 test files
- All tests use real pikepdf operations with temporary files (tmp_path)
- veraPDF calls mocked in validator tests (subprocess.run patched)
- CLI tests inherited from existing test suite

## Issues encountered
- None significant. pikepdf's API for structure tree manipulation is low-level; TagNode abstraction simplifies this.
- The extraction placeholder reads basic PDF structure only. Full Phase 1 integration will require updating pipeline._extract() and pipeline._build_tag_tree() to use live extraction data.

## Next steps
- Phase 1: Implement actual extraction pipeline (OCR, layout analysis)
- Phase 2: Implement tag tree builder with heading inference, list detection, role mapping
- Phase 4: Wire pipeline into FastAPI server
- veraPDF integration testing with real PDF corpus
