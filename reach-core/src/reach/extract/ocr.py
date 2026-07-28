"""
Unlimited OCR integration for text extraction.

Uses Baidu's Unlimited-OCR (3B MoE / 500M active) for:
- Full-page OCR on scanned pages
- Targeted text extraction within layout blocks
- Table structure recognition (row/column)
- Formula extraction (LaTeX output)

ADR-0004: Unlimited OCR for text ("what"), Surya for layout ("where").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from reach.extract.models import (
    BBox,
    BlockType,
    EquationBlock,
    TableBlock,
    TableCell,
    TextBlock,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model path (configurable via env)
# ---------------------------------------------------------------------------

import os

DEFAULT_MODELS_DIR = Path(
    os.environ.get("REACH_MODELS_DIR", Path.home() / ".reach" / "models")
)
DEFAULT_OCR_DIR = DEFAULT_MODELS_DIR / "unlimited-ocr"


# ============================================================================
# Model loading
# ============================================================================


def _get_ocr_backend() -> str:
    """Determine the best available backend for Unlimited OCR."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_unlimited_ocr(
    models_dir: Optional[Path] = None,
    backend: Optional[str] = None,
) -> "UnlimitedOCREngine":
    """Load the Unlimited OCR model.

    Args:
        models_dir: Directory containing the Unlimited-OCR model weights.
            Defaults to ~/.reach/models/unlimited-ocr/.
        backend: Override backend selection.

    Returns:
        A loaded UnlimitedOCREngine wrapper.
    """
    backend = backend or _get_ocr_backend()
    models_dir = models_dir or DEFAULT_OCR_DIR
    log.info("Loading Unlimited OCR (backend=%s, dir=%s) ...", backend, models_dir)

    engine = UnlimitedOCREngine(backend=backend, models_dir=models_dir)
    engine.load()
    return engine


# ============================================================================
# UnlimitedOCREngine wrapper
# ============================================================================


class UnlimitedOCREngine:
    """Wrapper around the Unlimited OCR model.

    Handles loading, inference, and memory management. When the model
    is not installed, returns empty results so tests can run without it.
    """

    def __init__(
        self,
        backend: str = "cpu",
        models_dir: Optional[Path] = None,
    ):
        self.backend = backend
        self.models_dir = models_dir or DEFAULT_OCR_DIR
        self._model: Optional[object] = None
        self._processor: Optional[object] = None
        self._available = False

    def load(self) -> None:
        """Load Unlimited OCR model weights.

        Honors ``REACH_NO_ML=1`` to skip the actual model download/load —
        useful for tests and environments where the OCR backend isn't
        available. The model is still loaded into memory even on CPU since
        transformers' from_pretrained runs device-agnostic init.
        """
        import os
        if os.environ.get("REACH_NO_ML"):
            log.info("REACH_NO_ML set — skipping OCR model loading")
            return

        # Unlimited OCR is typically loaded via transformers or a custom
        # loader. We check for the model directory and attempt to load.
        if not self.models_dir.exists():
            log.warning(
                "Unlimited OCR model not found at %s. "
                "Run `reach download-models` first.",
                self.models_dir,
            )
            self._available = False
            return

        try:
            # Unlimited OCR requires transformers==4.57.1 with AutoModel +
            # AutoTokenizer (the AutoModelForCausalLM path doesn't route
            # through this model's auto_map in any transformers version).
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.models_dir), trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                str(self.models_dir),
                trust_remote_code=True,
                torch_dtype="auto",
            )

            if self.backend == "cuda":
                self._model = self._model.cuda()
            elif self.backend == "mps":
                self._model = self._model.to("mps")

            self._available = True
            log.info("Unlimited OCR model loaded successfully.")
        except ImportError:
            log.warning(
                "transformers not installed. OCR will return empty results. "
                "Install with: pip install transformers"
            )
            self._available = False
        except Exception:
            log.exception("Failed to load Unlimited OCR model")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def recognize(self, image: Image.Image) -> str:
        """Run OCR on a single image and return extracted text.

        Args:
            image: A PIL Image (full page or cropped region).

        Returns:
            Extracted text as a string.
        """
        if not self._available or not self._model or not self._tokenizer:
            log.warning("Unlimited OCR not loaded — returning empty text.")
            return ""

        import tempfile
        from pathlib import Path

        # Resize the image to Unlimited OCR's expected size range. The model
        # uses base_size=1024 internally and crop_mode=True; oversized images
        # get rejected with "no text found" because the encoder downsamples
        # too aggressively. Cap the longest side at 2048 (≈ 2x base_size) for
        # the gundam config which is what we use.
        max_side = 2048
        if max(image.size) > max_side:
            ratio = max_side / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            log.debug("Resizing OCR input from %s to %s", image.size, new_size)
            image = image.resize(new_size, Image.LANCZOS)

        try:
            with tempfile.TemporaryDirectory(prefix="reach_ocr_") as tmp:
                tmp_path = Path(tmp)
                image_file = tmp_path / "input.png"
                image.save(image_file)
                output_path = tmp_path / "out"
                output_path.mkdir()

                # Unlimited OCR exposes a custom .infer() method that handles
                # image preprocessing + cropping + generation. It writes
                # results to <output_path>/result.md when save_results=True.
                #
                # The 'gundam' config (crop_mode=True, image_size=640) is the
                # default per the README. It first runs figure detection, then
                # OCRs each figure region. For full-page scanned images this
                # can degenerate to "one figure, no text" — the model treats
                # the whole page as a single image and emits just a bounding
                # box placeholder. Fall back to the 'base' config
                # (crop_mode=False, image_size=1024) which OCRs the whole page
                # directly when the gundam result looks like a single image
                # with no real text.
                self._model.infer(
                    self._tokenizer,
                    prompt="<image>Free OCR.",
                    image_file=str(image_file),
                    output_path=str(output_path),
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=True,
                )

                result_text = ""
                result_file = output_path / "result.md"
                if result_file.exists():
                    result_text = self._strip_ocr_artifacts(
                        result_file.read_text(encoding="utf-8")
                    )

                # If gundam only produced a placeholder ("one image, no text"),
                # retry with the base config (whole-page OCR, no crop).
                if not result_text or result_text == self._strip_ocr_artifacts(
                    "<|det|>image [0, 0, 999, 999]<|/det|>"
                ):
                    log.debug("Gundam OCR produced no text; retrying with base config")
                    self._model.infer(
                        self._tokenizer,
                        prompt="<image>document parsing.",
                        image_file=str(image_file),
                        output_path=str(output_path),
                        base_size=1024,
                        image_size=1024,
                        crop_mode=False,
                        save_results=True,
                    )
                    if result_file.exists():
                        result_text = self._strip_ocr_artifacts(
                            result_file.read_text(encoding="utf-8")
                        )

                return result_text
        except Exception:
            log.exception("OCR inference failed")
            return ""

    @staticmethod
    def _strip_ocr_artifacts(text: str) -> str:
        """Remove Unlimited OCR placeholder tokens from recognized text.

        Also deduplicates consecutive identical lines — the model occasionally
        emits overlapping detection boxes (e.g., one box for "that should be
        detected" and a larger one for "that should be detected\nas separate
        regions by OCR.") which leaves the inner text appearing twice.
        """
        import re
        # Detection boxes like `<|det|>title [43, 109, 999, 202]<|/det|>`
        text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
        # Image markdown placeholders `![](images/0.jpg)`
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        # Page break markers `<page>` / `<|end▁of▁sentence|>`
        text = re.sub(r"<page>|<\|end▁of▁sentence\|>", "", text)

        # Collapse runs of 3+ newlines (the model emits double newlines between
        # paragraphs) and dedup consecutive identical lines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = UnlimitedOCREngine._dedup_consecutive_lines(text)
        return text.strip()

    @staticmethod
    def _dedup_consecutive_lines(text: str) -> str:
        """Remove consecutive duplicate lines from OCR output.

        The Unlimited OCR model occasionally emits the same text twice when
        it produces overlapping detection boxes. This collapses those dupes
        while preserving intentional repetition in different paragraphs.
        """
        lines = text.split("\n")
        deduped: list[str] = []
        prev: str | None = None
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev:
                # Skip this duplicate line but keep the rest
                continue
            deduped.append(line)
            prev = stripped
        return "\n".join(deduped)

    def recognize_region(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> str:
        """Run OCR on a specific region of an image.

        Args:
            image: Full page image.
            bbox: (x1, y1, x2, y2) in image pixel coordinates.

        Returns:
            Extracted text from the cropped region.
        """
        cropped = image.crop(tuple(int(v) for v in bbox))
        return self.recognize(cropped)

    def extract_table(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> TableBlock:
        """Extract table structure from a table region.

        Uses Unlimited OCR's table-recognition capability to identify
        row/column boundaries and cell contents.

        Args:
            image: Full page image.
            bbox: (x1, y1, x2, y2) of the table region.

        Returns:
            A TableBlock with cells populated.
        """
        from reach.extract.models import BBox

        text = self.recognize_region(image, bbox)

        # Table structure extraction: Unlimited OCR can output structured
        # table data. For now, we use a heuristic approach that splits
        # the recognized text into cells based on newlines and tabs/spaces.
        rows = [row.strip() for row in text.split("\n") if row.strip()]
        if not rows:
            return TableBlock(
                bbox=BBox.from_list(list(bbox)),
                rows=0,
                columns=0,
                cells=[],
            )

        # Detect columns from first row (header)
        sample_cells = _split_table_row(rows[0])
        cols = len(sample_cells)
        if cols == 0:
            cols = 1

        cells: list[TableCell] = []
        x1, y1, x2, y2 = bbox
        row_h = (y2 - y1) / max(1, len(rows))
        col_w = (x2 - x1) / max(1, cols)

        for ri, row_text in enumerate(rows):
            row_cells = _split_table_row(row_text)
            for ci, cell_text in enumerate(row_cells[:cols]):
                cell_bbox = BBox(
                    x1=x1 + ci * col_w,
                    y1=y1 + ri * row_h,
                    x2=x1 + (ci + 1) * col_w,
                    y2=y1 + (ri + 1) * row_h,
                )
                cells.append(
                    TableCell(
                        text=cell_text.strip(),
                        row=ri,
                        col=ci,
                        bbox=cell_bbox,
                        is_header=(ri == 0),
                    )
                )

        return TableBlock(
            bbox=BBox.from_list(list(bbox)),
            rows=len(rows),
            columns=cols,
            cells=cells,
        )

    def extract_formula(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> str:
        """Extract a formula as LaTeX from an equation region.

        Args:
            image: Full page image.
            bbox: (x1, y1, x2, y2) of the equation region.

        Returns:
            LaTeX string.
        """
        text = self.recognize_region(image, bbox)
        # Unlimited OCR can output LaTeX for formulas when available.
        # For now, return raw recognized text; LaTeX conversion is a
        # future enhancement.
        return text.strip()

    def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._processor = None
        self._available = False


# ============================================================================
# Helpers
# ============================================================================


def _split_table_row(text: str) -> list[str]:
    """Split a table row text into cells using common delimiters."""
    # Try tab-separated
    if "\t" in text:
        return [c.strip() for c in text.split("\t")]

    # Try multiple spaces (2+ spaces as delimiter)
    import re

    cells = re.split(r"\s{2,}", text)
    if len(cells) > 1:
        return [c.strip() for c in cells]

    # Fall back to single space
    return [c.strip() for c in text.split(" ") if c.strip()]


# ============================================================================
# High-level OCR entry points
# ============================================================================


def ocr_scanned_page(
    image: Image.Image,
    engine: UnlimitedOCREngine,
) -> str:
    """Run full-page OCR on a scanned page image.

    Args:
        image: Rendered page image.
        engine: Loaded OCR engine.

    Returns:
        Full recognized text for the page.
    """
    return engine.recognize(image)


def ocr_layout_blocks(
    image: Image.Image,
    blocks: list[TextBlock | TableBlock | ImageBlock | EquationBlock],
    engine: UnlimitedOCREngine,
) -> list[TextBlock | TableBlock | ImageBlock | EquationBlock]:
    """Run targeted OCR on each block in a page.

    For TextBlock and EquationBlock: extract text/LaTeX from the cropped region.
    For TableBlock: extract full table structure.
    For ImageBlock: not OCR'd (handled by VLM enrichment in Phase 2).

    Args:
        image: Rendered page image.
        blocks: Layout blocks from Surya.
        engine: Loaded OCR engine.

    Returns:
        The blocks with text/tables/formulas populated.
    """
    for block in blocks:
        bbox_tuple = (block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2)

        if isinstance(block, TextBlock) and not block.text:
            block.text = engine.recognize_region(image, bbox_tuple)

        elif isinstance(block, TableBlock) and not block.cells:
            table = engine.extract_table(image, bbox_tuple)
            block.cells = table.cells
            block.rows = table.rows
            block.columns = table.columns

        elif isinstance(block, EquationBlock) and not block.latex:
            block.latex = engine.extract_formula(image, bbox_tuple)

    return blocks


def extract_page_text_with_ocr(
    image: Image.Image,
    blocks: list[TextBlock | TableBlock | ImageBlock | EquationBlock],
    engine: UnlimitedOCREngine,
    is_scanned: bool = False,
    content_stream_text: str = "",
) -> list[TextBlock | TableBlock | ImageBlock | EquationBlock]:
    """Determine strategy and extract text for a single page.

    Scanned pages: full-page OCR → distribute text across blocks.
    Digital-born pages: if content-stream text is sparse, use targeted OCR
    on each block; otherwise, content-stream text is already assigned.

    Args:
        image: Rendered page image.
        blocks: Layout blocks.
        engine: Loaded OCR engine.
        is_scanned: Whether the page is classified as scanned.
        content_stream_text: Text from the PDF content stream.

    Returns:
        Blocks with text populated.
    """
    if is_scanned and engine.is_available:
        full_text = ocr_scanned_page(image, engine)
        # Distribute full-text across text blocks by reading order
        text_blocks = [b for b in blocks if isinstance(b, TextBlock)]
        if text_blocks and full_text:
            from reach.extract.layout import _naive_text_distribution

            segments = _naive_text_distribution(full_text, len(text_blocks))
            for block, seg in zip(text_blocks, segments):
                block.text = seg
        elif full_text and not text_blocks:
            # Layout failed (no blocks) but OCR recovered text. Wrap the
            # full OCR'd text in a single page-spanning TextBlock so the
            # downstream pipeline has something to tag.
            blocks = [
                TextBlock(
                    block_type=BlockType.TEXT,
                    bbox=BBox(x1=0, y1=0, x2=612, y2=792),
                    text=full_text,
                    reading_order=0,
                    confidence=1.0,
                )
            ]

    elif engine.is_available:
        # Digital-born: run OCR only on blocks that don't have text
        blocks = ocr_layout_blocks(image, blocks, engine)

    return blocks
