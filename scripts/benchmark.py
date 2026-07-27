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

    # ---- Stage 1-3: Full remediation via CLI (handles extraction, tags, construction) ----
    try:
        r = subprocess.run(
            [sys.executable, "-m", "reach.cli", "remediate",
             str(pdf_path), "-o", str(out_pdf), "-r", str(report_json), "--no-validate"],
            capture_output=True, text=True, timeout=120,
            env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "reach-core" / "src")},
            cwd=str(PROJECT_ROOT),
        )
        result["remediated"] = r.returncode == 0

        # Count extraction blocks from the pipeline log
        import re
        blocks_match = re.search(r"PDF text layout: (\d+) pages, (\d+) blocks", r.stderr or "")
        if blocks_match:
            result["extraction_blocks"] = int(blocks_match.group(2))
            result["page_count"] = int(blocks_match.group(1))
        ml_match = re.search(r"ML extraction: (\d+) pages, (\d+) blocks", r.stderr or "")
        if ml_match:
            result["extraction_blocks"] = max(result["extraction_blocks"], int(ml_match.group(2)))

        # Parse the final report for confidence/status
        report_match = re.search(r"status=(\w+), confidence=([\d.]+), fixes=(\d+)", r.stderr or "")
        if report_match:
            result["confidence"] = float(report_match.group(2))

        if r.returncode != 0:
            result["errors"].append(r.stderr[-200:] if r.stderr else "Unknown error")
    except Exception as e:
        result["errors"].append(f"Remediation failed: {e}")

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
        "timestamp": datetime.now(datetime.UTC).isoformat(),  # type: ignore[attr-defined]
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
