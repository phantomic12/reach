#!/usr/bin/env python3
"""
Reach Benchmark Suite — full remediation + WCAG validation + content integrity.

Runs the complete pipeline (Surya layout + OCR + Qwen enrichment + PDF construction)
against every PDF in the corpus and produces a scored benchmark report.

Usage: python scripts/benchmark.py --limit 50 --output benchmark_report.json
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "tests" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "benchmark_output"
sys.path.insert(0, str(PROJECT_ROOT / "reach-core" / "src"))

log = logging.getLogger("benchmark")


def find_corpus(limit: int = 0) -> list[Path]:
    """Find PDFs in the corpus directory."""
    if not CORPUS_DIR.exists():
        return []
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    return pdfs[:limit] if limit > 0 else pdfs


def run_full_remediation(pdf_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the full Reach pipeline with real models."""
    out_pdf = output_dir / f"{pdf_path.stem}-remediated.pdf"
    out_json = output_dir / f"{pdf_path.stem}-extraction.json"
    report_json = output_dir / f"{pdf_path.stem}-report.json"
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = {
        "pdf": pdf_path.name,
        "size_kb": pdf_path.stat().st_size // 1024,
        "remediated": False,
        "extraction_blocks": 0,
        "tag_tree_nodes": 0,
        "alt_texts_generated": 0,
        "content_preserved": False,
        "confidence": 0.0,
        "errors": [],
        "warnings": [],
        "elapsed_s": 0.0,
    }

    # ---- Stage 1: Extraction with Surya + OCR ----
    try:
        from reach.extract import extract_document
        doc = extract_document(str(pdf_path), output_path=out_json, load_models=True, dpi=150)
        result["extraction_blocks"] = len(doc.all_blocks)
        result["page_count"] = len(doc.pages)
    except Exception as e:
        # Models may not be available — fall back to pikepdf-only extraction
        result["warnings"].append(f"Model extraction failed: {e}. Using pikepdf-only extraction.")
        doc = None

    # ---- Stage 2: Build tag tree with enrichment ----
    try:
        from reach.tagtree.builder import build_tag_tree_from_file
        from reach.tagtree.tag_models import TagTreeConfig
        from reach.enrich.enrich import Enricher, LlamaCppEnrichmentModel, MockEnrichmentModel
        from reach.tagtree.metadata import build_metadata
        from reach.tagtree.reading_order import compute_reading_order

        config = TagTreeConfig(enable_enrichment=True)
        if doc is not None and out_json.exists():
            tag_tree = build_tag_tree_from_file(out_json, config=config)
        else:
            tag_tree = build_tag_tree_from_file(None, config=config)

        # Enrichment
        real_model = LlamaCppEnrichmentModel(config)
        if real_model.is_available():
            model = real_model
            model._load_model()
        else:
            model = MockEnrichmentModel()
            result["warnings"].append("Qwen model not available, using mock enrichment")

        enricher = Enricher(model=model, config=config)
        enricher.enrich(tag_tree.document)
        enricher.apply_enrichment(tag_tree.root, enricher._last_report)
        result["alt_texts_generated"] = enricher._last_report.alt_texts_generated if enricher._last_report else 0

        # Reading order
        for page in tag_tree.document.pages:
            compute_reading_order(page)

        # Metadata
        tag_tree = build_metadata(tag_tree, filename=pdf_path.stem)
        result["tag_tree_nodes"] = len(tag_tree.root.children)
    except Exception as e:
        result["warnings"].append(f"Tag tree build failed: {e}")
        tag_tree = None

    # ---- Stage 3: PDF Construction ----
    try:
        from reach.construct.pipeline import RemediationPipeline
        pipeline = RemediationPipeline()
        report = pipeline.remediate(
            input_path=str(pdf_path),
            output=str(out_pdf),
            report_path=str(report_json),
            validate=False,
        )
        result["remediated"] = report.overall_status == "pass"
        result["confidence"] = report.overall_confidence
        result["fixes"] = report.total_fixes
    except Exception as e:
        result["errors"].append(f"Construction failed: {e}")

    # ---- Stage 4: Content Integrity Check ----
    if result["remediated"] and out_pdf.exists():
        result["content_preserved"] = check_content_integrity(pdf_path, out_pdf)

    result["elapsed_s"] = round(time.time() - start, 2)
    return result


def check_content_integrity(original: Path, remediated: Path) -> bool:
    """Verify that the remediated PDF preserves original content."""
    try:
        from pypdf import PdfReader
        orig = PdfReader(str(original))
        remd = PdfReader(str(remediated))

        # Page count preserved
        if len(orig.pages) != len(remd.pages):
            return False

        # Text content preserved (at least 80% overlap)
        orig_text = " ".join(p.extract_text() or "" for p in orig.pages).lower()
        remd_text = " ".join(p.extract_text() or "" for p in remd.pages).lower()

        if not orig_text.strip():
            return True  # Scanned PDF — no text to compare

        orig_words = set(orig_text.split())
        remd_words = set(remd_text.split())
        if not orig_words:
            return True

        overlap = len(orig_words & remd_words) / len(orig_words)
        return overlap >= 0.80

    except Exception:
        return False


def check_wcag_compliance(pdf_path: Path) -> dict[str, Any]:
    """Run veraPDF WCAG/UA-2 validation."""
    result = {"compliant": False, "failures": 0, "output": ""}
    try:
        r = subprocess.run(
            ["verapdf", "--profile", "ua-2", str(pdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        result["compliant"] = "isValid=\"true\"" in r.stdout
        # Count failures
        import re
        fails = re.findall(r"failedRules=\"(\d+)\"", r.stdout)
        result["failures"] = sum(int(f) for f in fails)
        result["output"] = r.stdout[:2000]
    except FileNotFoundError:
        result["output"] = "veraPDF not installed"
    except Exception as e:
        result["output"] = str(e)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reach benchmark suite")
    parser.add_argument("--limit", type=int, default=20, help="PDFs to test")
    parser.add_argument("--output", default="benchmark_report.json", help="Report path")
    parser.add_argument("--fast", action="store_true", help="Skip model loading for speed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = find_corpus(limit=args.limit)
    if not pdfs:
        log.error("No PDFs found. Run scripts/collect_corpus.py first.")
        sys.exit(1)

    log.info("Benchmarking %d PDFs...", len(pdfs))
    results = []

    for i, pdf in enumerate(pdfs, 1):
        log.info("[%d/%d] %s", i, len(pdfs), pdf.name[:60])
        result = run_full_remediation(pdf, OUTPUT_DIR)

        # WCAG check
        out_pdf = OUTPUT_DIR / f"{pdf.stem}-remediated.pdf"
        if out_pdf.exists():
            wcag = check_wcag_compliance(out_pdf)
            result["wcag_compliant"] = wcag["compliant"]
            result["wcag_failures"] = wcag["failures"]
        else:
            result["wcag_compliant"] = False
            result["wcag_failures"] = -1

        results.append(result)

    # Summary
    total = len(results)
    remediated = sum(1 for r in results if r["remediated"])
    content_ok = sum(1 for r in results if r["content_preserved"])
    wcag_ok = sum(1 for r in results if r.get("wcag_compliant", False))
    avg_conf = sum(r["confidence"] for r in results) / total if total else 0
    avg_time = sum(r["elapsed_s"] for r in results) / total if total else 0
    total_alt = sum(r["alt_texts_generated"] for r in results)
    total_blocks = sum(r["extraction_blocks"] for r in results)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total": total,
        "remediated": remediated,
        "remediation_rate": round(remediated / total * 100, 1),
        "content_preserved": content_ok,
        "content_integrity_rate": round(content_ok / total * 100, 1),
        "wcag_compliant": wcag_ok,
        "wcag_rate": round(wcag_ok / total * 100, 1),
        "avg_confidence": round(avg_conf, 3),
        "avg_time_s": round(avg_time, 2),
        "total_alt_texts": total_alt,
        "total_extraction_blocks": total_blocks,
        "results": results,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Report: %s", out_path)

    print(f"\n{'='*60}")
    print(f"  Reach Benchmark Results")
    print(f"{'='*60}")
    print(f"  PDFs tested:        {total}")
    print(f"  Remediated:          {remediated}/{total} ({report['remediation_rate']}%)")
    print(f"  Content preserved:   {content_ok}/{total} ({report['content_integrity_rate']}%)")
    print(f"  WCAG compliant:      {wcag_ok}/{total} ({report['wcag_rate']}%)")
    print(f"  Avg confidence:      {avg_conf:.1%}")
    print(f"  Avg time:            {avg_time:.1f}s")
    print(f"  Alt texts generated: {total_alt}")
    print(f"  Extraction blocks:   {total_blocks}")
    print(f"{'='*60}")

    if remediated < total:
        failed = [r for r in results if not r["remediated"]]
        print(f"\n  Failed ({len(failed)}):")
        for r in failed[:5]:
            print(f"    {r['pdf'][:50]}: {r['errors'][:80] if r['errors'] else 'unknown'}")

    sys.exit(0 if remediated == total else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
