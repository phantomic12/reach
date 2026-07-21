#!/usr/bin/env python3
"""
Agent-driven comprehensive test runner for Reach.

Runs the full pipeline against every PDF in the corpus and generates a test report.
Designed to be invoked by AI agents or CI.

Usage: python scripts/run_tests.py [--full] [--parallel N]
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "tests" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"
REPORT_PATH = PROJECT_ROOT / "tests" / "test_report.json"


def find_corpus_pdfs() -> list[Path]:
    """Find all PDFs in the corpus directory."""
    if not CORPUS_DIR.exists():
        log.warning(f"Corpus directory not found: {CORPUS_DIR}")
        return []
    return sorted(CORPUS_DIR.glob("*.pdf"))


def run_remediate(pdf_path: Path, output_path: Path) -> dict:
    """Run `reach remediate` on a single PDF and return results."""
    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "reach.cli", "remediate",
                str(pdf_path),
                "--output", str(output_path / f"{pdf_path.stem}-remediated.pdf"),
                "--report", str(output_path / f"{pdf_path.stem}-report.json"),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min per PDF
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - start
        return {
            "pdf": pdf_path.name,
            "status": "success" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "stderr": result.stderr[-500:] if result.stderr else None,
            "elapsed_seconds": round(elapsed, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "pdf": pdf_path.name,
            "status": "timeout",
            "elapsed_seconds": 300,
        }
    except Exception as e:
        return {
            "pdf": pdf_path.name,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": time.time() - start,
        }


def run_verapdf(remediated_pdf: Path) -> Optional[dict]:
    """Run veraPDF validation on a remediated PDF."""
    try:
        result = subprocess.run(
            ["verapdf", "--profile", "ua-2", str(remediated_pdf)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        passed = "isValid=\"true\"" in result.stdout or "nonCompliant=\"0\"" in result.stdout
        return {
            "passed": passed,
            "output": result.stdout[:1000] if result.stdout else result.stderr[:1000],
        }
    except FileNotFoundError:
        return {"passed": None, "error": "veraPDF not installed"}
    except subprocess.TimeoutExpired:
        return {"passed": None, "error": "veraPDF validation timed out"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reach test suite against PDF corpus")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (slower, more thorough)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs to test")
    parser.add_argument("--verapdf", action="store_true", help="Also run veraPDF validation")
    args = parser.parse_args()

    pdfs = find_corpus_pdfs()
    if args.limit > 0:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        log.warning("No PDFs in corpus. Run scripts/collect_corpus.py first.")
        # Generate quick test PDFs
        log.info("Generating test PDFs...")
        from scripts.collect_corpus import generate_test_pdfs

        generate_test_pdfs(CORPUS_DIR)
        pdfs = find_corpus_pdfs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Testing {len(pdfs)} PDFs...")
    results = []

    for i, pdf_path in enumerate(pdfs, 1):
        log.info(f"[{i}/{len(pdfs)}] {pdf_path.name}...")
        result = run_remediate(pdf_path, OUTPUT_DIR)

        if args.verapdf and result["status"] == "success":
            remediated = OUTPUT_DIR / f"{pdf_path.stem}-remediated.pdf"
            if remediated.exists():
                result["verapdf"] = run_verapdf(remediated)

        results.append(result)

    # Summary
    passed = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] == "error")
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    total_time = sum(r.get("elapsed_seconds", 0) for r in results)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "timeouts": timeouts,
        "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
        "total_seconds": round(total_time, 1),
        "avg_seconds_per_pdf": round(total_time / len(results), 1) if results else 0,
        "results": results,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info(f"Report: {REPORT_PATH}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Reach Test Report")
    print(f"{'='*50}")
    print(f"  Total:    {len(results)}")
    print(f"  Passed:   {passed} ({report['pass_rate']}%)")
    print(f"  Failed:   {failed}")
    print(f"  Errors:   {errors}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Time:     {total_time:.1f}s ({report['avg_seconds_per_pdf']}s/PDF)")
    print(f"{'='*50}")

    # Exit with failure if anything failed
    if failed > 0 or errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
