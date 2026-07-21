# ADR-0004: Unlimited OCR + Surya Extraction Stack

**Status**: Accepted  
**Date**: 2026-07-21

## Context

The extraction stage needs two capabilities: layout analysis (detecting block types, positions, reading order) and text recognition (extracting text from rendered pages). The 2026 landscape offers both unified VLM models (Surya 2, PaddleOCR-VL-1.5) and specialized models (Unlimited OCR).

## Decision

**Unlimited OCR for text extraction + Surya for layout analysis.** A two-model pipeline rather than a single unified model.

## Rationale

1. Unlimited OCR (Baidu, MIT license) achieves 93.23% OmniDocBench v1.5 with 3B MoE / 500M active params. Its R-SWA attention keeps KV cache constant regardless of document length — critical for academic PDFs that may be 100+ pages.
2. Surya's layout analysis outputs 18 block types with bounding boxes AND reading-order indices — exactly what tag-tree construction needs. Unlimited OCR does not output structured layout information.
3. MIT license on Unlimited OCR aligns with Reach's open-source goals.
4. DocLayout-YOLO was rejected due to AGPL-3.0 license (incompatible). RT-DocLayout (33M, June 2026) has no weights released yet — watchlist for future.
5. This two-model approach decouples "where" (layout) from "what" (text) — the same pattern proven in the local-llm-pdf-ocr pipeline.

## Consequences

- Two model dependencies rather than one. Sequential loading keeps memory manageable.
- Layout analysis must run first; OCR uses layout regions for targeted extraction.
- Model loading/unloading adds latency — caching strategies needed for batch processing.
- Surya's OpenRAIL-M weights license is compatible with academic non-commercial use.
