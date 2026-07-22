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
        """Load Unlimited OCR model weights."""
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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.models_dir), trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
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
        if not self._available or not self._model or not self._processor:
            log.warning("Unlimited OCR not loaded — returning empty text.")
            return ""

        try:
            # Preprocess using the already-loaded processor
            inputs = self._processor(images=image, return_tensors="pt")
            if self.backend == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.backend == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}

            # Generate
            with __import__("torch").no_grad():
                outputs = self._model.generate(**inputs)

            # Decode
            text = self._processor.batch_decode(outputs, skip_special_tokens=True)
            return text[0] if text else ""
        except Exception:
            log.exception("OCR inference failed")
            return ""

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

    elif engine.is_available:
        # Digital-born: run OCR only on blocks that don't have text
        blocks = ocr_layout_blocks(image, blocks, engine)

    return blocks
