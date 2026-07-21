"""
Extraction pipeline — Phase 1 of the Reach PDF/UA-2 remediation pipeline.

Orchestrates: PDF parsing → layout analysis (Surya) → text extraction
(Unlimited OCR) → structured JSON output.

Usage:
    from reach.extract import extract_document

    doc = extract_document("input.pdf")
    doc.write_json("output.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from reach.extract.layout import (
    SuryaLayoutModel,
    analyze_layout,
    convert_surya_blocks,
    load_surya,
    merge_text_with_layout,
)
from reach.extract.models import (
    BlockType,
    DocumentMetadata,
    EquationBlock,
    ExtractedDocument,
    ImageBlock,
    Page,
    PageMetadata,
    TableBlock,
    TextBlock,
)
from reach.extract.ocr import (
    UnlimitedOCREngine,
    extract_page_text_with_ocr,
    load_unlimited_ocr,
)
from reach.extract.parser import ParsedDocument

log = logging.getLogger(__name__)

# Re-exports for convenience
__all__ = [
    # Orchestrator
    "extract_document",
    "ExtractionPipeline",
    # Models
    "ExtractedDocument",
    "Page",
    "PageMetadata",
    "DocumentMetadata",
    "TextBlock",
    "TableBlock",
    "ImageBlock",
    "EquationBlock",
    "BlockType",
    # Parser
    "ParsedDocument",
    # Layout
    "load_surya",
    "SuryaLayoutModel",
    # OCR
    "load_unlimited_ocr",
    "UnlimitedOCREngine",
]


# ============================================================================
# Extraction pipeline
# ============================================================================


class ExtractionPipeline:
    """Full Phase 1 extraction pipeline.

    Encapsulates model loading/unloading, orchestrates the parse→layout→OCR
    flow, and produces an ExtractedDocument with structured JSON serialization.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        backend: Optional[str] = None,
        dpi: int = 200,
        load_models: bool = True,
    ):
        """Initialize the extraction pipeline.

        Args:
            models_dir: Base directory for model weights. If None, uses defaults.
            backend: Override GPU backend ("cuda", "mps", "cpu").
            dpi: Rendering resolution for page images.
            load_models: If True, loads Surya and OCR models on init.
                Set False for tests that mock models.
        """
        self.models_dir = models_dir
        self.backend = backend
        self.dpi = dpi

        self._surya: Optional[SuryaLayoutModel] = None
        self._ocr: Optional[UnlimitedOCREngine] = None

        if load_models:
            self._surya = load_surya(models_dir=models_dir, backend=backend)
            self._ocr = load_unlimited_ocr(models_dir=models_dir, backend=backend)

    @property
    def surya(self) -> Optional[SuryaLayoutModel]:
        return self._surya

    @surya.setter
    def surya(self, model: Optional[SuryaLayoutModel]) -> None:
        self._surya = model

    @property
    def ocr(self) -> Optional[UnlimitedOCREngine]:
        return self._ocr

    @ocr.setter
    def ocr(self, engine: Optional[UnlimitedOCREngine]) -> None:
        self._ocr = engine

    def run(self, pdf_path: Path | str) -> ExtractedDocument:
        """Run the full extraction pipeline on a PDF.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            An ExtractedDocument with all pages and blocks populated.
        """
        pdf_path = Path(pdf_path)
        log.info("Starting extraction: %s", pdf_path.name)

        # 1. Parse PDF
        parsed = ParsedDocument.from_path(pdf_path, dpi=self.dpi)
        log.info(
            "Parsed %d pages (%d scanned, %d digital)",
            len(parsed.page_images),
            sum(parsed.scanned_flags),
            len(parsed.page_images) - sum(parsed.scanned_flags),
        )

        # 2. Layout analysis (Surya)
        layout_blocks: list[list[TextBlock | TableBlock | ImageBlock | EquationBlock]]
        if self._surya and self._surya.is_available:
            layout_blocks = analyze_layout(
                images=parsed.page_images,
                model=self._surya,
                page_texts=parsed.page_texts,
            )
        else:
            log.warning("Surya not available — layout will be empty.")
            layout_blocks = [[] for _ in parsed.page_images]

        # 3. Text extraction (Unlimited OCR)
        pages: list[Page] = []
        for page_num, blocks in enumerate(layout_blocks):
            is_scanned = (
                page_num < len(parsed.scanned_flags) and parsed.scanned_flags[page_num]
            )
            page_image = parsed.page_images[page_num]
            page_text = (
                parsed.page_texts[page_num]
                if page_num < len(parsed.page_texts)
                else ""
            )

            if self._ocr and self._ocr.is_available:
                blocks = extract_page_text_with_ocr(
                    image=page_image,
                    blocks=blocks,
                    engine=self._ocr,
                    is_scanned=is_scanned,
                    content_stream_text=page_text,
                )

            # Calculate page dimensions
            if page_num < len(parsed.pdf.pages):
                mb = parsed.pdf.pages[page_num].MediaBox
                w_pt = float(mb[2] - mb[0])
                h_pt = float(mb[3] - mb[1])
            else:
                w_pt, h_pt = 612.0, 792.0

            # Calculate text coverage
            text_chars = sum(
                len(b.text) if isinstance(b, TextBlock) else 0 for b in blocks
            )
            page_area = w_pt * h_pt
            text_coverage = (text_chars * 60) / page_area if page_area > 0 else 0.0

            metadata = PageMetadata(
                page_number=page_num + 1,  # 1-indexed for human readability
                width_pt=w_pt,
                height_pt=h_pt,
                is_scanned=is_scanned,
                text_coverage=min(text_coverage, 1.0),
            )

            pages.append(Page(metadata=metadata, blocks=blocks))

        # 4. Assemble document
        doc = ExtractedDocument(metadata=parsed.metadata, pages=pages)

        log.info("Extraction complete: %d pages, %d blocks", len(pages), len(doc.all_blocks))
        return doc

    def unload(self) -> None:
        """Free all model memory."""
        if self._surya:
            self._surya.unload()
        if self._ocr:
            self._ocr.unload()


# ============================================================================
# Convenience function
# ============================================================================


def extract_document(
    pdf_path: Path | str,
    output_path: Optional[Path | str] = None,
    *,
    models_dir: Optional[Path] = None,
    backend: Optional[str] = None,
    dpi: int = 200,
    load_models: bool = True,
    pipeline: Optional[ExtractionPipeline] = None,
) -> ExtractedDocument:
    """Extract a PDF and optionally write structured JSON output.

    This is the main entry point for Phase 1. It runs the full
    parse → layout → OCR pipeline and produces an ExtractedDocument.

    Args:
        pdf_path: Path to the input PDF.
        output_path: If provided, write JSON output to this file.
        models_dir: Base directory for model weights.
        backend: GPU backend override.
        dpi: Page rendering resolution.
        load_models: If True, load Surya and OCR models.
        pipeline: Optional pre-configured pipeline. If provided, uses this
            instead of creating a new one.

    Returns:
        An ExtractedDocument.

    Raises:
        FileNotFoundError: If pdf_path doesn't exist.
        ValueError: If the PDF is malformed or has no pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pipeline is None:
        pipeline = ExtractionPipeline(
            models_dir=models_dir,
            backend=backend,
            dpi=dpi,
            load_models=load_models,
        )

    try:
        doc = pipeline.run(pdf_path)

        if output_path:
            output_path = Path(output_path)
            doc.write_json(output_path)

        return doc
    finally:
        if pipeline is not None and pipeline._surya:
            pipeline.unload()


# ============================================================================
# Extend ExtractedDocument with serialization
# ============================================================================


def _doc_write_json(self: ExtractedDocument, path: Path) -> None:
    """Write the extracted document to a JSON file with schema validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = self.model_dump(mode="json", exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Output written: %s (%d bytes)", path, path.stat().st_size)


def _doc_from_json(cls: type[ExtractedDocument], path: Path) -> ExtractedDocument:
    """Load an extracted document from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return cls.model_validate(data)


# Monkey-patch serialization methods onto ExtractedDocument
ExtractedDocument.write_json = _doc_write_json  # type: ignore[attr-defined]
ExtractedDocument.from_json = classmethod(_doc_from_json)  # type: ignore[attr-defined]
