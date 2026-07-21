#!/usr/bin/env python3
"""
Reach PDF Corpus Collector — scrape diverse academic PDFs for testing.

Sources:
    - arXiv.org (papers across disciplines)
    - University course material pages
    - Government document repositories

Categories targeted:
    - single-column text
    - multi-column (2-col, 3-col)
    - scanned/image-only pages
    - fillable forms (tax forms, applications)
    - tables-heavy (financial, statistical)
    - equations-heavy (math, physics)
    - multilingual (non-English content)
    - mixed raster/vector content
    - large documents (100+ pages)
    - encrypted/protected PDFs (password='test' for testing)

Usage: python scripts/collect_corpus.py --output tests/corpus/ --count 50
"""
import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger(__name__)

ARXIV_SEARCH = "https://export.arxiv.org/api/query"

# Diverse arXiv categories across disciplines
CATEGORIES = [
    "cs.AI",       # AI papers (tables, equations, diagrams)
    "cs.CL",       # NLP (multi-column, references)
    "math.HO",     # Math history (equations, scanned content)
    "physics.soc-ph",  # Social physics (mixed content)
    "q-bio.GN",    # Genomics (figures, tables, multi-column)
    "stat.ML",     # ML stats (equations, algorithms)
    "econ.GN",     # Economics (tables, charts)
    "astro-ph.IM", # Instrumentation (diagrams, figures)
]


def search_arxiv(category: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for papers in a category."""
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        resp = httpx.get(ARXIV_SEARCH, params=params, timeout=30)
        resp.raise_for_status()

        papers = []
        # Parse Atom XML (simple regex — arxiv API returns clean XML)
        entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
        for entry in entries:
            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            pdf_match = re.search(r'<link href="(https://arxiv.org/pdf/[^"]+)"', entry)
            if pdf_match:
                papers.append({
                    "title": title_match.group(1).strip() if title_match else "Unknown",
                    "pdf_url": pdf_match.group(1),
                    "category": category,
                })
        return papers
    except Exception as e:
        log.warning(f"arXiv search failed for {category}: {e}")
        return []


def download_pdf(url: str, output_dir: Path, slug: str) -> Optional[Path]:
    """Download a PDF, skipping if already cached."""
    filepath = output_dir / f"{slug}.pdf"
    if filepath.exists():
        return filepath

    try:
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()

        # Verify it's actually a PDF
        if not resp.content[:4] == b"%PDF":
            log.debug(f"Not a PDF: {url}")
            return None

        filepath.write_bytes(resp.content)
        log.info(f"Downloaded: {slug} ({len(resp.content) // 1024} KB)")
        return filepath
    except Exception as e:
        log.warning(f"Download failed: {url}: {e}")
        return None


def generate_test_pdfs(output_dir: Path) -> list[Path]:
    """Generate synthetic test PDFs for edge cases not found in the wild."""
    from pypdf import PdfWriter

    generated = []

    # 1. Simple single-page PDF
    p = output_dir / "generated_single_page.pdf"
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    w.add_metadata({"/Title": "Single Page Test"})
    w.write(str(p))
    generated.append(p)

    # 2. Multi-page PDF (10 pages)
    p = output_dir / "generated_multi_page.pdf"
    w = PdfWriter()
    for i in range(10):
        w.add_blank_page(width=612, height=792)
    w.add_metadata({"/Title": "Multi Page Test"})
    w.write(str(p))
    generated.append(p)

    # 3. Large PDF (100 pages, minimal content)
    p = output_dir / "generated_large_100_pages.pdf"
    w = PdfWriter()
    for i in range(100):
        w.add_blank_page(width=612, height=792)
    w.add_metadata({"/Title": "100 Page Test"})
    w.write(str(p))
    generated.append(p)

    # 4. Untagged PDF (no structure)
    p = output_dir / "generated_untagged.pdf"
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    w.add_metadata({"/Title": "Untagged Test"})
    w.write(str(p))
    generated.append(p)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PDF corpus for Reach testing")
    parser.add_argument("--output", default="tests/corpus", help="Output directory")
    parser.add_argument("--count", type=int, default=20, help="Papers per category")
    parser.add_argument("--generate-only", action="store_true", help="Only generate synthetic PDFs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic tests
    generated = generate_test_pdfs(output_dir)
    log.info(f"Generated {len(generated)} synthetic PDFs")

    if args.generate_only:
        return

    # Scrape arXiv
    total = 0
    for category in CATEGORIES:
        log.info(f"Searching arXiv: {category}")
        papers = search_arxiv(category, max_results=args.count)
        for paper in papers:
            slug = hashlib.md5(paper["pdf_url"].encode()).hexdigest()[:10]
            path = download_pdf(paper["pdf_url"], output_dir, slug)
            if path:
                total += 1
            time.sleep(1.5)  # Respect arXiv rate limit (1 request per second recommended)

    log.info(f"Total corpus: {total} scraped + {len(generated)} generated = {total + len(generated)} PDFs")

    # Write corpus manifest
    manifest_path = output_dir / "manifest.json"
    manifest = []
    for pdf in sorted(output_dir.glob("*.pdf")):
        manifest.append({
            "filename": pdf.name,
            "size_kb": pdf.stat().st_size // 1024,
            "category": "generated" if pdf.name.startswith("generated_") else "arxiv",
        })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
