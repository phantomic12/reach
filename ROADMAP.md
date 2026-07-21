# Reach — Development Roadmap

Version: 0.1.0-pre | Updated: 2026-07-21

This roadmap is designed for multi-agent parallel development. Each phase produces a testable, verifiable artifact. Agents should read the relevant ADR before starting any task.

---

## Phase 0: Project Scaffold (Week 1)

**Goal**: Repository structure, tooling, CI, and model acquisition. Nothing processes PDFs yet — pure setup.

### 0.1 Repository Setup
- [ ] Initialize Git repo with `reach-core/`, `reach-server/`, `reach-desktop/` package structure
- [ ] `pyproject.toml` with dependencies: pikepdf, pypdf, llama-cpp-python, torch, fastapi, pytest, pillow
- [ ] `.gitignore`, `.editorconfig`, pre-commit hooks
- [ ] CI pipeline (GitHub Actions): lint, type-check, test on push

### 0.2 Model Acquisition
- [ ] Download script for Unlimited OCR (HuggingFace: baidu/Unlimited-OCR)
- [ ] Download script for Surya layout model (pip install surya-ocr + model pull)
- [ ] Download script for Qwen 3.5 9B Q4_K_M GGUF (Unsloth: unsloth/Qwen3.5-9B-GGUF)
- [ ] Model verification: checksum validation, smoke-test inference
- [ ] Model cache directory structure: `~/.reach/models/`

### 0.3 GPU Backend Detection
- [ ] Detect available GPU vendor (NVIDIA/AMD/Intel/None)
- [ ] Verify llama.cpp backend availability for detected vendor
- [ ] Fallback chain: CUDA → ROCm → Vulkan → CPU
- [ ] Startup diagnostic: log detected GPU, backend, VRAM available

### 0.4 Test Harness
- [ ] Pytest configuration with fixtures for model loading
- [ ] Test PDF corpus directory structure
- [ ] veraPDF CLI wrapper for validation assertions
- [ ] Benchmark framework: pages/sec, memory usage, accuracy metrics

**Deliverable**: `reach-core` installable via `pip install -e .`, models downloadable via `reach download-models`, GPU detection working, test suite running (even if mostly empty).

---

## Phase 1: Extraction Pipeline (Week 2-3)

**Goal**: Given a PDF, produce structured JSON with layout blocks + text content + reading order. No PDF output yet.

### 1.1 PDF Parsing
- [ ] PDF content stream parser (pikepdf/pypdf): extract existing text, fonts, positions, images, form fields
- [ ] Page renderer: rasterize each page to image for visual analysis
- [ ] Distinguish digital-born vs scanned pages (heuristic: text coverage + font presence)

### 1.2 Layout Analysis (Surya)
- [ ] Surya model loader with GPU backend selection
- [ ] Layout detection: run Surya on rendered pages, produce block list (type, bbox, reading-order index)
- [ ] Merge: align Surya layout blocks with parsed PDF elements (position-based matching)
- [ ] Edge case: multi-column layouts, irregular flows, rotated text

### 1.3 Text Extraction (Unlimited OCR)
- [ ] Unlimited OCR model loader with vLLM/llama.cpp backend
- [ ] OCR pass: text extraction for scanned pages and text-in-image regions
- [ ] Table extraction: detect table regions from layout, run targeted OCR, output row/column structure
- [ ] Formula extraction: LaTeX/MathML output for equation regions

### 1.4 Structured Output
- [ ] JSON schema for extracted document: `{pages: [{blocks: [{type, bbox, text, reading_order, font_info}]}]}`
- [ ] Validation: every block has type + bbox + reading order. No orphans.
- [ ] Serialize/deserialize for pipeline stage handoff

**Deliverable**: `reach extract input.pdf` → `output.json` with complete structured document representation. Tested against diverse PDF corpus.

---

## Phase 2: Tag Tree Construction (Week 3-4)

**Goal**: Convert structured extraction into a PDF/UA-2 conformant tag tree. No visual PDF output yet — just an in-memory tag structure.

### 2.1 Tag Tree Builder
- [ ] Map layout block types to PDF/UA-2 tag types (SectionHeader → H1-H6, Text → P, Table → Table/TD/TR/TH, etc.)
- [ ] Heading hierarchy inference: font size, font weight, position analysis, numbering patterns
- [ ] List detection: bullet points, numbered lists, indentation patterns
- [ ] Artifact marking: page numbers, running headers/footers, decorative elements

### 2.2 Reading Order Engine
- [ ] Implement reading order from Surya's reading-order indices
- [ ] Multi-column handling: XY-cut algorithm for column-aware ordering
- [ ] Edge cases: sidebars, pull quotes, footnotes, marginalia

### 2.3 Table Structure
- [ ] Header row detection (bold text, top position, distinct formatting)
- [ ] Scope attributes: Row/Column/Both for header cells
- [ ] Merged cell handling: ID/Headers associations
- [ ] Empty cell handling: must still be tagged

### 2.4 Content Enrichment (Qwen 3.5 9B)
- [ ] VLM prompt template for alt text generation: "Describe this [figure/chart/diagram] for a blind reader. Include what type of visual it is and its key information. Be concise (2-3 sentences)."
- [ ] Figure extraction: crop figure regions from rendered pages
- [ ] Batch inference: process all figures in document
- [ ] Alt text quality filter: reject empty, overly generic ("a chart"), or hallucinated descriptions
- [ ] Decorative vs. informational image classification

### 2.5 Metadata
- [ ] Document title extraction (from content, filename, or metadata)
- [ ] Language detection (document-level + per-passage overrides)
- [ ] XMP metadata generation
- [ ] Display title flag setting

**Deliverable**: `reach build-tags output.json` → `tags.json` with complete tag tree ready for PDF construction. Validated against PDF/UA-2 structural requirements (manual review of tag hierarchy).

---

## Phase 3: PDF Construction & Validation (Week 4-5)

**Goal**: Write a PDF/UA-2 compliant file from the tag tree and extracted content. Pass veraPDF validation.

### 3.1 PDF Writer
- [ ] pikepdf-based tag tree injection into source PDF structure
- [ ] Content stream rewriting: ensure text order matches reading order
- [ ] Font embedding: ensure all fonts are embedded (PDF/UA-2 requirement)
- [ ] Unicode mapping: ToUnicode CMap for all fonts
- [ ] Image handling: attach alt text to Figure tags

### 3.2 Form Field Remediation
- [ ] Detect AcroForm fields in source PDF
- [ ] Tag form fields with proper Form/Option/Button roles
- [ ] Add tooltip (TU entry) to each form field
- [ ] Tab order: set logical tab sequence matching reading order

### 3.3 Link Preservation
- [ ] Preserve internal links (TOC, cross-references) with proper Link tags
- [ ] Preserve external hyperlinks
- [ ] Link annotations must include /Contents (alt text equivalent)

### 3.4 Validation (veraPDF)
- [ ] Run veraPDF PDF/UA-2 profile against output
- [ ] Parse validation report: categorize failures by type
- [ ] Auto-fix loop: common failures get automated correction passes
- [ ] Flag unresolvable failures in output report

### 3.5 Output Report
- [ ] Per-document remediation report: what was fixed, what couldn't be fixed, confidence scores
- [ ] veraPDF validation summary: pass/fail with failure breakdown
- [ ] Human-review queue for documents with unresolvable failures

**Deliverable**: `reach remediate input.pdf` → `remediated.pdf` + `report.json`. Verified passing veraPDF PDF/UA-2 on at least 80% of test corpus documents.

---

## Phase 4: Web Server (Week 5-6)

### 4.1 FastAPI Backend
- [ ] Upload endpoint: accept PDF file, queue remediation job
- [ ] Status endpoint: poll job progress
- [ ] Download endpoint: serve remediated PDF + report
- [ ] Batch endpoint: accept multiple PDFs via ZIP upload
- [ ] Job queue: background processing with progress tracking

### 4.2 Web UI
- [ ] Upload page: drag-and-drop PDF, show upload progress
- [ ] Processing page: live status updates (WebSocket or polling)
- [ ] Results page: download remediated PDF, view report summary
- [ ] Batch page: upload ZIP, see per-file status, download all

### 4.3 Deployment
- [ ] Dockerfile: multi-stage build with model caching
- [ ] docker-compose.yml with GPU passthrough
- [ ] Health check endpoint
- [ ] Resource limits and configuration

---

## Phase 5: Desktop Application (Week 6-7)

### 5.1 PyInstaller Bundle
- [ ] Package reach-core + reach-server as single executable
- [ ] Bundle llama.cpp binaries for Windows and Linux
- [ ] Platform-specific GPU backend selection at install time

### 5.2 Auto-Installer Scripts
- [ ] Windows: detect GPU, install CUDA/Vulkan runtime if missing, download models
- [ ] Linux: detect GPU, apt-get ROCm/Vulkan if missing, download models
- [ ] Graceful degradation: CPU-only mode if no GPU detected

### 5.3 Desktop Shell
- [ ] Launch reach-server on localhost with random port
- [ ] Open browser window pointing at localhost
- [ ] System tray icon with status and quit option
- [ ] Auto-update check

---

## Phase 6: Testing & Corpus (Week 7-8+)

### 6.1 PDF Corpus Collection
- [ ] Scrape diverse academic PDFs: ArXiv papers, university course materials, conference proceedings, scanned historical documents
- [ ] Categories: single-column, multi-column, scanned, forms, mixed content, tables-heavy, image-heavy, multilingual
- [ ] Ground truth: manually verified PDF/UA-2 compliance status for each

### 6.2 Automated Testing
- [ ] Extraction accuracy: compare extracted text/layout to ground truth
- [ ] Tag tree correctness: validate tag hierarchy against PDF/UA-2 spec
- [ ] veraPDF pass rate: target >90% on corpus
- [ ] Regression suite: every fix adds a test case
- [ ] Performance benchmarks: pages/sec, memory usage, model load time

### 6.3 Agent-Driven Testing
- [ ] Test generation agents: produce edge-case PDFs programmatically
- [ ] Validation agents: run remediation, check output, report failures
- [ ] Fuzzing: malformed PDFs, encrypted PDFs, extreme page counts, unusual fonts
- [ ] Cross-model comparison: test with different VLMs, OCR models

---

## Phase 7: Polish & Release (Week 8-10)

### 7.1 Interactive Review Workspace (Roadmap Stretch)
- [ ] Visual tag tree editor: review and adjust tags before export
- [ ] Alt text review: approve/edit/reject VLM-generated descriptions
- [ ] Reading order correction: drag-and-drop reordering

### 7.2 Documentation
- [ ] User guide: installation, usage, troubleshooting
- [ ] Developer guide: architecture, contributing, model training
- [ ] API documentation (FastAPI auto-generated)

### 7.3 Release
- [ ] PyPI publication: `pip install reach-core`
- [ ] Docker Hub: reach-server image
- [ ] GitHub Releases: reach-desktop installers for Windows and Linux
- [ ] Academic paper / blog post

---

## Parallel Workstreams (Ongoing)

These can run in parallel with any phase:

- **Corpus Collection** (Phase 6): continuously gather test PDFs. Agents can scrape while other agents build.
- **Model Evaluation**: benchmark alternative models (PaddleOCR-VL-1.6, RT-DocLayout when weights release, Bonsai smaller VLMs if they gain vision).
- **Documentation**: write as you build. Every function gets a docstring. Every ADR gets updated.
- **Dev Logs**: every agent writes to `dev-logs/YYYY-MM-DD-<task>.md` with: what was done, what was tested, what failed, what the fix was.
