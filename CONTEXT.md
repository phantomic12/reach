# Reach — Domain Glossary

## Core Concepts

**Remediation**: The process of transforming an inaccessible PDF into a PDF/UA-2 compliant document by analyzing its structure, extracting content, and reconstructing it with proper accessibility tags, reading order, alt text, and metadata.

**Source PDF**: The original, potentially inaccessible PDF submitted by the user. May be untagged, partially tagged, scanned, or generated from any source (Word, LaTeX, InDesign, web, etc.).

**Remediated PDF**: The output — a new PDF/UA-2 compliant file with proper tag tree, reading order, alt text on all non-decorative images, table structure, form field tagging, and metadata. Functionally equivalent to the source.

**Tag Tree**: The hierarchical structure of semantic elements (headings, paragraphs, lists, tables, figures, artifacts) embedded in a PDF that screen readers use to navigate the document. PDF/UA-2 requires a valid, complete tag tree.

**Reading Order**: The logical sequence in which content should be consumed by assistive technology. Must match the visual flow intended by the document author, not raw content-stream order.

**Artifact**: A decorative or non-content element (page numbers, running headers, decorative borders) that must be marked as an artifact so screen readers skip it. Non-artifact content without proper tagging is a PDF/UA-2 failure.

## Pipeline Stages

**Extraction**: The process of understanding what is on each page — text, images, tables, form fields, vector graphics, and their positions. The Hybrid approach means: (1) parse the PDF's internal content stream for existing structure, then (2) use visual analysis (OCR, layout models) to fill gaps where structure is missing or corrupted.

**Layout Analysis**: Detecting the type, position, and extent of each content block on a page: heading, paragraph, table, figure, list, equation, header, footer. Outputs bounding boxes with class labels and reading-order indices.

**Content Enrichment**: The VLM-driven stage that generates alt text for figures, describes charts/diagrams, interprets table semantics, and resolves ambiguous elements that layout analysis alone cannot classify.

**PDF/UA-2 Export**: Writing the final PDF file with a valid tag tree conforming to ISO 14289-2:2024, embedded metadata (title, language), tagged form fields with tooltips, and all required structural elements.

**Validation**: Running veraPDF against the output to verify PDF/UA-2 conformance. Can be used as a gate (fail if non-conformant) or as a diagnostic (report failures for human review).

## Standards

**PDF/UA-2** (ISO 14289-2:2024): The international standard for accessible PDF based on PDF 2.0 (ISO 32000-2:2020). Defines requirements for tag trees, reading order, alt text, table structure, metadata, and form field accessibility.

**WCAG 2.2** (Web Content Accessibility Guidelines): While a web standard, WCAG success criteria map to PDF/UA-2 requirements. Many institutional accessibility policies reference WCAG levels (A, AA).

**veraPDF**: The industry-reference PDF/A and PDF/UA validator, developed by Dual Lab. Used to programmatically verify conformance.

## Model Vocabulary

**Unlimited OCR**: Baidu's 3B MoE (500M active) end-to-end OCR model. MIT license. Uses Reference Sliding Window Attention (R-SWA) for constant KV cache regardless of document length. Handles text extraction, table recognition, and formula recognition.

**Surya**: Datalab's layout analysis model. Detects 18 block types with bounding boxes and reading-order indices. 650M params, runs on CPU/GPU/Apple Silicon.

**Qwen 3.5 9B**: Alibaba's natively multimodal language model. Used for content enrichment — generating alt text for figures, interpreting charts, and describing complex visual elements. Apache 2.0. Runs via llama.cpp.

**llama.cpp**: The inference runtime for Qwen 3.5 9B and other GGUF-quantized models. Supports CUDA, ROCm, Vulkan, and Metal backends for GPU acceleration across NVIDIA, AMD, Intel, and Apple hardware.

## Deployment

**Reach Server**: The web-hosted deployment. FastAPI backend serving the remediation pipeline. Upload PDFs via browser, download remediated PDFs. All processing server-side.

**Reach Desktop**: The client-side application for Windows and Linux. Same Python pipeline, running locally. Bundles llama.cpp binaries and model files. Auto-installer scripts handle dependency setup.
