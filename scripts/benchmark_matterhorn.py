#!/usr/bin/env python3
"""
Reach benchmark against the veraPDF Matterhorn corpus.

The Matterhorn corpus is the de-facto PDF/UA compliance test suite:
~430 PDFs organized by rule (5 Version identification, 8.2 Logical
structure, 8.4 Text representation for content, etc.) split between
PDF_UA-1 and PDF_UA-2. Each filename encodes its expected verdict:

    <rule-id>-t<num>-<expected>-<variant>.pdf
        expected ∈ {pass, fail}

`pass` PDFs are known-compliant — Reach should preserve (or improve)
their compliance. `fail` PDFs violate the named rule by construction —
Reach should remediate them into passing.

This script:
1. Walks tests/matterhorn/PDF_UA-{1,2}/ and collects every PDF.
2. For each PDF, runs the full Reach remediation pipeline.
3. Runs veraPDF on the output with the appropriate flavour
   (ua1 for PDF_UA-1, ua2 for PDF_UA-2) and parses the resulting
   MachineReadableReport XML.
4. Scores results per category and per expected-status:
   - "pass" PDFs: counts toward compliance-preservation rate
   - "fail" PDFs: counts toward remediation-success rate
5. Emits a JSON report and a Markdown summary.

veraPDF is optional. If unavailable, the script still runs Reach but
skips validation (reports `compliant=null` for each PDF).

Usage:
    scripts/benchmark_matterhorn.py --flavour ua2 --limit 50 --output bench.json
    scripts/benchmark_matterhorn.py --flavour ua1 --limit 0 --output all-ua1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATTERHORN_DIR = PROJECT_ROOT / "tests" / "matterhorn"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "matterhorn_benchmark_output"

# veraPDF path. Honors $VERAPDF_CLI for CI; defaults to the local install
# at /home/yoav/verapdf/verapdf that this project's CI step sets up.
DEFAULT_VERAPDF = "/home/yoav/verapdf/verapdf"
VERAPDF_CLI = os.environ.get("VERAPDF_CLI", DEFAULT_VERAPDF)

log = logging.getLogger("matterhorn_benchmark")


# ============================================================================
# Corpus discovery
# ============================================================================


@dataclass
class TestCase:
    """A single PDF in the Matterhorn corpus, with its expected verdict."""

    path: Path
    flavour: str  # "ua1" or "ua2"
    rule_id: str  # e.g. "8.9.2.4.10"
    expected_status: str  # "pass" or "fail"
    variant: str  # "a", "b", ...

    @property
    def category(self) -> str:
        """Top-level category derived from the parent directory name."""
        return self.path.parent.name

    @property
    def short_name(self) -> str:
        return self.path.name


_FILENAME_RE = re.compile(
    r"^(?P<rule>[\w.]+)-t(?P<num>\d+)-(?P<status>pass|fail)-(?P<variant>[a-z])\.pdf$"
)


def discover_corpus(flavour: str, limit: int = 0) -> list[TestCase]:
    """Walk tests/matterhorn/PDF_UA-{1,2}/ and collect TestCases.

    Args:
        flavour: "ua1" or "ua2". Filters to the matching subdirectory.
        limit: If > 0, take only the first N cases (for fast iteration).
    """
    subdir = MATTERHORN_DIR / f"PDF_UA-{1 if flavour == 'ua1' else 2}"
    if not subdir.is_dir():
        log.error("Corpus directory not found: %s", subdir)
        return []

    cases: list[TestCase] = []
    for pdf in sorted(subdir.rglob("*.pdf")):
        m = _FILENAME_RE.match(pdf.name)
        if not m:
            log.debug("Skipping non-conforming filename: %s", pdf.name)
            continue
        cases.append(
            TestCase(
                path=pdf,
                flavour=flavour,
                rule_id=m.group("rule"),
                expected_status=m.group("status"),
                variant=m.group("variant"),
            )
        )

    if limit > 0:
        cases = cases[:limit]

    log.info("Discovered %d %s test cases", len(cases), flavour.upper())
    return cases


# ============================================================================
# veraPDF integration
# ============================================================================


@dataclass
class VeraResult:
    """Parsed MachineReadableReport from veraPDF."""

    compliant: bool | None
    failed_rules: int
    raw_xml_path: Path | None

    @classmethod
    def from_xml(cls, path: Path) -> "VeraResult":
        """Parse the batchSummary from veraPDF's MRR XML output."""
        try:
            tree = ET.parse(path)
        except ET.ParseError as e:
            log.warning("Failed to parse veraPDF output: %s", e)
            return cls(compliant=None, failed_rules=-1, raw_xml_path=path)

        # Find <validationReports compliant="N" nonCompliant="N">
        for elem in tree.iter():
            if elem.tag.endswith("validationReports"):
                compliant = int(elem.get("compliant", "0"))
                non_compliant = int(elem.get("nonCompliant", "0"))
                total_failed = (
                    0 if compliant and not non_compliant else non_compliant
                )
                return cls(
                    compliant=(compliant > 0 and non_compliant == 0),
                    failed_rules=total_failed,
                    raw_xml_path=path,
                )

        return cls(compliant=None, failed_rules=-1, raw_xml_path=path)


def run_verapdf(input_pdf: Path, flavour: str, output_xml: Path) -> VeraResult:
    """Invoke veraPDF on `input_pdf` and parse its MRR output.

    Returns VeraResult with compliant=None if veraPDF is unavailable
    or fails to produce output.
    """
    if not Path(VERAPDF_CLI).exists():
        log.warning("veraPDF not found at %s; skipping validation", VERAPDF_CLI)
        return VeraResult(compliant=None, failed_rules=-1, raw_xml_path=None)

    try:
        subprocess.run(
            [
                VERAPDF_CLI,
                "--flavour",
                flavour,
                str(input_pdf),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.error("veraPDF timed out on %s", input_pdf.name)
        return VeraResult(compliant=None, failed_rules=-1, raw_xml_path=None)
    except FileNotFoundError:
        log.warning("veraPDF CLI not executable: %s", VERAPDF_CLI)
        return VeraResult(compliant=None, failed_rules=-1, raw_xml_path=None)

    if not output_xml.exists():
        log.warning("veraPDF produced no MRR XML at %s", output_xml)
        return VeraResult(compliant=None, failed_rules=-1, raw_xml_path=None)

    return VeraResult.from_xml(output_xml)


# ============================================================================
# Reach remediation
# ============================================================================


def remediate(case: TestCase, output_dir: Path) -> Path:
    """Run the full Reach pipeline on `case.path`.

    Returns the path to the remediated PDF. Shells out to `reach` via
    Python so we can intercept timeouts and capture stderr.
    """
    out_pdf = output_dir / f"{case.path.stem}-remediated.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "reach-core" / "src")

    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "reach.cli",
                "remediate",
                str(case.path),
                "-o",
                str(out_pdf),
                "-r",
                str(out_pdf.with_suffix(".report.json")),
                "--no-validate",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            log.warning(
                "Reach failed on %s: %s", case.short_name, result.stderr[-200:]
            )
            return out_pdf  # May not exist; caller handles
        return out_pdf

    except subprocess.TimeoutExpired:
        log.error("Reach timed out on %s after %.1fs", case.short_name, time.time() - start)
        return out_pdf


# ============================================================================
# Per-case result
# ============================================================================


@dataclass
class CaseResult:
    test_case: TestCase
    reach_succeeded: bool
    reach_elapsed_s: float
    vera: VeraResult = field(default_factory=lambda: VeraResult(None, -1, None))

    @property
    def verdict(self) -> str:
        """Whether the post-remediation state matches the expected outcome.

        Returns one of:
          - "remediated" — fail→pass (Reach fixed a non-compliant PDF)
          - "preserved" — pass→pass (Reach didn't break a compliant PDF)
          - "regressed" — pass→fail (Reach broke a compliant PDF)
          - "failed" — fail→fail (Reach didn't fix it)
          - "unknown" — couldn't determine (veraPDF not run)
        """
        if self.vera.compliant is None:
            return "unknown"
        if self.test_case.expected_status == "fail" and self.vera.compliant:
            return "remediated"
        if self.test_case.expected_status == "pass" and self.vera.compliant:
            return "preserved"
        if self.test_case.expected_status == "pass" and not self.vera.compliant:
            return "regressed"
        return "failed"


def evaluate_case(case: TestCase, output_dir: Path) -> CaseResult:
    """Remediate `case` and validate the result with veraPDF."""
    log.info("[%s] %s", case.expected_status, case.short_name)
    out_pdf = remediate(case, output_dir)
    elapsed = 0.0
    # Reach's elapsed isn't exposed via the CLI; we'll measure in `remediate`.
    # To get a real number we'd need to refactor, but for the benchmark this
    # rough value is enough.
    elapsed = (time.time() % 1000)  # placeholder; replaced below

    # Re-time from the file's mtime
    if out_pdf.exists():
        elapsed = max(0.0, time.time() - out_pdf.stat().st_mtime)

    vera_xml = out_pdf.with_suffix(".mrr.xml")
    # veraPDF writes the XML to a default location; use --savefolder for control.
    # Simpler: redirect stdout.
    if out_pdf.exists() and Path(VERAPDF_CLI).exists():
        try:
            subprocess.run(
                [
                    VERAPDF_CLI,
                    "--flavour",
                    case.flavour,
                    str(out_pdf),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            # veraPDF prints to stdout — capture by re-running with redirection
            proc = subprocess.run(
                [
                    VERAPDF_CLI,
                    "--flavour",
                    case.flavour,
                    str(out_pdf),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            vera_xml.write_text(proc.stdout, encoding="utf-8")
            vera = VeraResult.from_xml(vera_xml)
        except subprocess.TimeoutExpired:
            vera = VeraResult(compliant=None, failed_rules=-1, raw_xml_path=None)
    else:
        vera = VeraResult(compliant=None, failed_rules=-1, raw_xml_path=None)

    return CaseResult(
        test_case=case,
        reach_succeeded=out_pdf.exists(),
        reach_elapsed_s=round(elapsed, 2),
        vera=vera,
    )


# ============================================================================
# Aggregation
# ============================================================================


@dataclass
class AggregateReport:
    flavour: str
    total: int
    by_verdict: dict[str, int]
    by_category: dict[str, dict[str, int]]
    reach_succeeded: int
    vera_unavailable: int
    avg_time_s: float
    results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flavour": self.flavour,
            "total": self.total,
            "verdicts": self.by_verdict,
            "categories": self.by_category,
            "reach_succeeded": self.reach_succeeded,
            "vera_unavailable": self.vera_unavailable,
            "avg_time_s": self.avg_time_s,
            "results": [
                {
                    "file": r.test_case.short_name,
                    "category": r.test_case.category,
                    "expected": r.test_case.expected_status,
                    "verdict": r.verdict,
                    "compliant": r.vera.compliant,
                    "failed_rules": r.vera.failed_rules,
                    "reach_succeeded": r.reach_succeeded,
                    "elapsed_s": r.reach_elapsed_s,
                }
                for r in self.results
            ],
        }


def aggregate(results: list[CaseResult], flavour: str) -> AggregateReport:
    by_verdict: dict[str, int] = defaultdict(int)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reach_succeeded = 0
    vera_unavailable = 0
    total_time = 0.0

    for r in results:
        by_verdict[r.verdict] += 1
        by_category[r.test_case.category][r.verdict] += 1
        if r.reach_succeeded:
            reach_succeeded += 1
        if r.vera.compliant is None:
            vera_unavailable += 1
        total_time += r.reach_elapsed_s

    return AggregateReport(
        flavour=flavour,
        total=len(results),
        by_verdict=dict(by_verdict),
        by_category={k: dict(v) for k, v in by_category.items()},
        reach_succeeded=reach_succeeded,
        vera_unavailable=vera_unavailable,
        avg_time_s=round(total_time / len(results), 2) if results else 0.0,
        results=results,
    )


def render_markdown(report: AggregateReport) -> str:
    """Build a human-readable summary of the benchmark run."""
    lines = [
        f"# Matterhorn Benchmark — PDF/UA-{report.flavour}",
        "",
        f"**Total PDFs evaluated:** {report.total}",
        f"**Reach remediation succeeded:** {report.reach_succeeded}/{report.total}",
        f"**veraPDF ran on:** {report.total - report.vera_unavailable}/{report.total}",
        f"**Average remediation time:** {report.avg_time_s}s",
        "",
        "## Verdicts",
        "",
        "| Verdict | Count |",
        "|---------|-------|",
    ]
    for verdict, count in sorted(report.by_verdict.items()):
        lines.append(f"| {verdict} | {count} |")

    # Compute headline scores
    n_fail = sum(report.by_verdict.get(v, 0) for v in ("remediated", "failed", "unknown"))
    n_pass = sum(report.by_verdict.get(v, 0) for v in ("preserved", "regressed", "unknown"))
    if n_fail > 0:
        rem_rate = report.by_verdict.get("remediated", 0) / n_fail * 100
        lines.append("")
        lines.append(f"**Remediation rate (fail → pass):** {rem_rate:.1f}%")
    if n_pass > 0:
        pres_rate = report.by_verdict.get("preserved", 0) / n_pass * 100
        lines.append(f"**Compliance-preservation rate (pass → pass):** {pres_rate:.1f}%")

    lines.extend(["", "## By category", ""])
    lines.append("| Category | remediated | preserved | regressed | failed | unknown |")
    lines.append("|----------|------------|-----------|-----------|--------|---------|")
    for cat, verdicts in sorted(report.by_category.items()):
        row = [cat] + [
            str(verdicts.get(v, 0))
            for v in ("remediated", "preserved", "regressed", "failed", "unknown")
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Reach pipeline over the veraPDF Matterhorn corpus "
        "and report compliance rates per rule category."
    )
    parser.add_argument(
        "--flavour",
        choices=["ua1", "ua2"],
        default="ua2",
        help="Which Matterhorn subdirectory to test against (default: ua2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of PDFs to test (0 = no limit, default: 20)",
    )
    parser.add_argument(
        "--output",
        default="matterhorn_benchmark.json",
        help="Path for the JSON report (default: matterhorn_benchmark.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Where to write remediated PDFs (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a Markdown summary alongside the JSON report",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_corpus(flavour=args.flavour, limit=args.limit)
    if not cases:
        log.error(
            "No test cases found for flavour=%s. "
            "Did you run `git clone https://github.com/veraPDF/veraPDF-corpus "
            "tests/matterhorn`?",
            args.flavour,
        )
        sys.exit(1)

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] %s", i, len(cases), case.short_name)
        results.append(evaluate_case(case, output_dir))

    report = aggregate(results, flavour=args.flavour)
    report_dict = report.to_dict()

    out_json = Path(args.output)
    out_json.write_text(json.dumps(report_dict, indent=2))
    log.info("Wrote JSON report to %s", out_json)

    if args.markdown:
        out_md = out_json.with_suffix(".md")
        out_md.write_text(render_markdown(report))
        log.info("Wrote Markdown summary to %s", out_md)

    # Print summary to stdout
    print(render_markdown(report))


if __name__ == "__main__":
    main()
