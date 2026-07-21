# Reach — Agent Development Guide

> **For AI coding agents.** Read this before starting any task. It tells you what you need, where things are, and how to log your work.

## Quick Start (for agents)

1. Read `CONTEXT.md` for domain vocabulary. If you use a term wrong, your work will be confusing.
2. Read the ADR relevant to your task in `docs/adr/`. They explain *why* decisions were made.
3. Find your task in `ROADMAP.md`. Each task has a checkbox. Update it when you complete something.
4. Log everything in `dev-logs/`. See format below.

## Project Layout

```
reach/
├── CONTEXT.md              # Domain glossary — read this first
├── ROADMAP.md              # Task list — mark things done here
├── DEVGUIDE.md             # This file
├── docs/
│   └── adr/                # Architecture Decision Records
│       ├── 0001-pdf-ua-2-target-standard.md
│       ├── 0002-hybrid-extraction.md
│       ├── 0003-component-based-build.md
│       ├── 0004-extraction-stack.md
│       ├── 0005-enrichment-model.md
│       └── 0006-shared-core-architecture.md
├── dev-logs/               # Per-task development logs
│   └── YYYY-MM-DD-<task-slug>.md
├── reach-core/             # Python package: the remediation pipeline
│   ├── __init__.py
│   ├── extract/            # Phase 1: extraction pipeline
│   ├── tagtree/            # Phase 2: tag tree construction
│   ├── enrich/             # Phase 2: VLM content enrichment
│   ├── construct/          # Phase 3: PDF construction
│   ├── validate/           # Phase 3: veraPDF validation
│   └── models/             # Model loading, GPU detection, caching
├── reach-server/           # Phase 4: FastAPI web application
│   ├── main.py
│   ├── routes/
│   └── templates/
├── reach-desktop/          # Phase 5: Desktop packaging
│   ├── installer/
│   └── scripts/
├── tests/                  # All tests live here
│   ├── corpus/             # Test PDF files
│   ├── test_extract/
│   ├── test_tagtree/
│   ├── test_construct/
│   └── test_validate/
├── scripts/                # Utility scripts
│   └── download_models.py
├── pyproject.toml
└── Dockerfile
```

## Model Stack

| Model | Purpose | Size | License | Source |
|---|---|---|---|---|
| Unlimited OCR | Text/table/formula extraction | 3B MoE / 500M active | MIT | `baidu/Unlimited-OCR` |
| Surya (layout) | Layout detection + reading order | 650M | Apache 2.0 / OpenRAIL-M | `pip install surya-ocr` |
| Qwen 3.5 9B (Q4_K_M) | Alt text + content enrichment | 9B → 5.68 GB GGUF | Apache 2.0 | `unsloth/Qwen3.5-9B-GGUF` |
| pikepdf | PDF read/write/tag injection | N/A | MPL 2.0 | `pip install pikepdf` |
| pypdf | Pure Python PDF construction | N/A | BSD | `pip install pypdf` |
| veraPDF | PDF/UA-2 validation | N/A | Various | CLI tool |

Models are cached at `~/.reach/models/`. Use `scripts/download_models.py` to fetch them.

## GPU Backend Selection

The startup sequence:
1. Detect GPU vendor via `torch.cuda`, `rocm-smi`, or Vulkan capabilities
2. Select llama.cpp backend: CUDA → ROCm → Vulkan → CPU
3. Log the selection: "Reach: using CUDA backend (NVIDIA RTX 4080, 12 GB VRAM)"
4. On CPU-only: "Reach: no GPU detected. Running in CPU mode (slow)."

See `reach-core/models/backend.py` for the detection logic.

## Dev Log Format

Every task gets a log file: `dev-logs/YYYY-MM-DD-<task-slug>.md`

```markdown
# Task: <brief description>
**Date**: 2026-07-21
**Phase**: <1-7>
**Agent**: <agent identifier>
**Status**: complete | in-progress | blocked

## What was done
- Itemized list of changes
- Files created/modified with paths
- Decisions made (reference ADRs if updating one)

## Testing
- What tests were added (file paths)
- Test results: X passed, Y failed, Z skipped
- Performance numbers if relevant (pages/sec, memory)

## Issues encountered
- Error message and context
- Root cause
- Fix applied
- If NOT fixed: flag as blocked with reason

## Next steps
- What should the next agent do
- Dependencies on other tasks
```

## Testing Rules

1. Every new module gets a corresponding `tests/test_<module>/` directory.
2. Every fix gets a regression test. If a bug slipped through, the test gap is as important as the fix.
3. veraPDF assertions: `assert verapdf_validates(output_pdf, profile="ua-2")`
4. Test PDFs go in `tests/corpus/`. Each has a `README.md` describing its edge cases.
5. Run the full suite before marking a task complete: `pytest -x --tb=short`

## Coding Conventions

- Python 3.12+
- Type hints on all public functions
- Docstrings: Google-style, one-line summary + Args/Returns/Raises
- All model inference goes through `reach-core/models/` — no direct llama.cpp/surya imports outside that module
- Config via environment variables + `~/.reach/config.yaml` fallback
- No hardcoded paths

## Key Constraints

- **16GB RAM target**: Desktop must run on machines with 16GB system RAM. Keep peak memory below 14GB.
- **Self-hosted**: No cloud APIs. No telemetry. Everything runs locally.
- **PDF/UA-2 output**: Output MUST pass veraPDF PDF/UA-2 validation. If it doesn't, it's a bug.
- **GPU auto-detection**: Don't ask the user what GPU they have. Detect it.
- **Model download on demand**: Models are large. Download only when needed. Cache aggressively.
- **Agent-testable**: Every pipeline stage must be callable from Python without a web server.
