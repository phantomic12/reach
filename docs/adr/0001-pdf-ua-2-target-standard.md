# ADR-0001: PDF/UA-2 as Target Standard

**Status**: Accepted  
**Date**: 2026-07-21

## Context

Reach needs a specific accessibility standard to target for output validation. The options were:

- PDF/UA-1 (ISO 14289-1:2012) — older standard, wider tool compatibility
- PDF/UA-2 (ISO 14289-2:2024) — current standard based on PDF 2.0
- WCAG-mapped compliance without formal PDF/UA certification

## Decision

**Target PDF/UA-2 as the primary compliance standard.**

## Rationale

1. PDF/UA-2 is the current ISO standard, published 2024. By the time Reach is mature, PDF/UA-1 will be legacy.
2. PDF/UA-2 maps cleanly to WCAG 2.2, which academic institutions increasingly require under ADA Title II and EAA enforcement.
3. The stricter requirements (artifact decoration, extended tagging for lists/tables/figures, Unicode mapping) produce a better product.
4. veraPDF already supports PDF/UA-2 validation, so automated conformance testing is available.

## Consequences

- Output PDFs must conform to ISO 32000-2 (PDF 2.0), not PDF 1.7. This constrains the PDF construction library choice.
- Validation must use veraPDF's PDF/UA-2 profile specifically.
- Artifact marking, form field tagging, and metadata requirements are more demanding than PDF/UA-1.
