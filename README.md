# Reach

Automated **PDF/UA-2 remediation** pipeline. Drop in inaccessible PDFs, get back
standards-compliant accessible documents — tags, structure tree, alt text,
forms, links, all wired up.

- **MIT License · Python 3.12+ · Self-hosted · Local-first**
- **PDF/UA-2 compliant output** (validates against veraPDF when installed)
- **Component-based build** (pikepdf + DocLayout-YOLO / Surya layout + Unlimited OCR + Qwen 3.5 9B enrichment) — every stage swappable.

## Quick Start

```bash
pip install "reach-core[all]"
reach download-models
reach remediate input.pdf -o output.pdf -r report.json
```

See [INSTALL.md](INSTALL.md) for per-OS install steps (macOS / Linux /
Windows), veraPDF setup, GPU acceleration, and troubleshooting.

## What it does

Reach runs a five-stage pipeline:

1. **Extract** — parse the PDF, render pages, run layout detection (pikepdf text
   positions by default for digital PDFs; DocLayout-YOLO for scanned/image-only
   PDFs) to get a per-page block map with bounding boxes and reading order.
   Surya v2 is available as an alternate backend but requires Docker.
2. **OCR** — Unlimited OCR (Baidu, MIT) reads text inside each block, with full-page
   fallback for scanned pages.
3. **Enrich** — Qwen 3.5 9B (Q4_K_M GGUF) generates alt text for figures and
   classifies decorative vs. informative images.
4. **Tag tree** — build a PDF/UA-2 structure tree (Document → Sect → P / H1-6 / Table
   / Figure / Formula / Link) with proper parent-child relationships and language metadata.
5. **Construct** — write the remediated PDF with embedded fonts, ToUnicode CMaps,
   marked content IDs (MCIDs), and the structure tree linked into `/StructTreeRoot`.
   Run veraPDF with the `--flavour ua-2` profile to confirm compliance.

## Architecture

```
PDF in
  │
  ▼
┌─────────────────────────────────────────────┐
│ Phase 1: Extract (DocLayout-YOLO / Surya)  │
│ Phase 1: OCR       (Unlimited OCR)         │
│ Phase 2: Build     (tag tree from blocks)  │
│ Phase 2: Enrich    (Qwen 3.5 9B alt text)  │
│ Phase 3: Construct (pikepdf structure tree)│
│ Phase 3: Validate  (veraPDF ua-2 profile)  │
└─────────────────────────────────────────────┘
  │
  ▼
PDF/UA-2 out
```

Each phase is independently callable from Python. See `CONTEXT.md` for the
domain vocabulary and `docs/adr/` for the architecture decision records.

## CLI

```
reach version               # Print version + model status
reach download-models       # Fetch Unlimited OCR + Qwen + Surya
reach extract input.pdf     # Phase 1 only — emits extraction JSON
reach build-tags input.pdf  # Phase 2 only — emits tag tree JSON
reach remediate input.pdf   # Full pipeline: extract + tags + construct
```

## Development

See [DEVGUIDE.md](DEVGUIDE.md) for the agent development guide (project layout,
test conventions, dev log format, coding rules).

```bash
git clone https://github.com/phantomic12/reach
cd reach
pip install -e ".[dev]"
pytest -q --tb=short
```

## Benchmark

Two benchmark scripts ship with the project:

```bash
# Local corpus (digital-born PDFs in tests/corpus/)
scripts/benchmark.py --limit 50 --output report.json

# veraPDF Matterhorn corpus (PDF/UA-1 + PDF/UA-2 test suite)
scripts/setup_matterhorn.sh     # clones ~246MB corpus on first run
scripts/benchmark_matterhorn.py --flavour ua2 --limit 50 --markdown --output report.json
```

The Matterhorn benchmark parses each PDF's expected verdict from its
filename (`<rule>-t<num>-<expected>-<variant>.pdf`), runs Reach over it,
then scores the output with veraPDF. Verdicts: `remediated` (fail→pass),
`preserved` (pass→pass), `regressed` (pass→fail), `failed` (fail→fail).

## Model stack

| Model                  | Purpose                          | License    |
| ---------------------- | -------------------------------- | ---------- |
| DocLayout-YOLO         | Layout detection (scanned PDFs)  | AGPL-3.0*  |
| Surya v2 (alt)         | Layout detection fallback        | Apache 2.0 |
| Unlimited OCR (3B MoE) | Text / tables / formulas         | MIT        |
| Qwen 3.5 9B Q4_K_M     | Alt text + content enrichment    | Apache 2.0 |
| pikepdf / pypdf        | PDF read / write                 | MPL 2.0 / BSD |
| veraPDF                | PDF/UA-2 validation              | Various    |

\* **License note**: DocLayout-YOLO is AGPL-3.0 because that's what the
upstream package ships under. It is the only in-process layout backend that
runs without Docker; Surya v2 (Apache 2.0) requires Docker for inference.
Reach prefers permissive licenses (MIT, Apache 2.0, BSD, MPL) for everything
else. The ML layout path is only invoked for scanned / image-only PDFs — the
production path uses pikepdf text positions (no ML, no Docker) for
digital-born PDFs.

## License

MIT — see [LICENSE](LICENSE).
