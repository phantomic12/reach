# ADR-0002: Hybrid Extraction Approach

**Status**: Accepted  
**Date**: 2026-07-21

## Context

PDF remediation tools face a fundamental choice: render pages to images and reconstruct from scratch (visual-first), or parse the PDF's internal content stream and only fill gaps with visual analysis (hybrid).

## Decision

**Hybrid approach.** Parse the PDF's existing content stream first, then use visual analysis (OCR, layout models, VLMs) to fill gaps.

## Rationale

1. Pure visual reconstruction produces PDFs that *look* correct but lose interactive elements: form fields, internal links, TOC clickability, embedded metadata, and selectable text.
2. Hybrid preserves everything salvageable from the source PDF and only fills what's missing.
3. For academic PDFs specifically, tables of contents, cross-references, and citation links must remain functional. Pure OCR loses these.
4. The "hybrid" approach is also what OpenDataLoader PDF and the Remedocs pipeline use for best results.

## Consequences

- The extraction stage must handle both structured (PDF content stream) and unstructured (rendered image) inputs.
- PDF parsing libraries (pikepdf, pypdf) are required alongside visual models (Surya, Unlimited OCR).
- Edge case: PDFs with corrupt content streams may fall back to full visual extraction.
- Form field preservation requires special handling in the PDF construction stage.
