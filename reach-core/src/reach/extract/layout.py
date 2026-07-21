"""
Surya layout analysis integration.

Loads Surya's layout detection model and runs it on rendered page images
to produce a list of classified blocks with bounding boxes and reading-order
indices. Maps Surya's 18 block types to Reach's BlockType enum.

ADR-0004: Surya for layout ("where"), Unlimited OCR for text ("what").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from reach.extract.models import (
    SURYA_TO_REACH,
    BaseBlock,
    BBox,
    BlockType,
    EquationBlock,
    ImageBlock,
    TableBlock,
    TextBlock,
    map_surya_block_type,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type stub for Surya results (avoids hard import of surya-ocr in tests)
# ---------------------------------------------------------------------------


class SuryaBlock:
    """Stub for a Surya detection result block.

    Mirrors the shape of data returned by Surya's batch_detection()
    without requiring the surya-ocr package at import time.
    """

    def __init__(
        self,
        label: str,
        bbox: list[float],
        reading_order: int = 0,
        confidence: float = 0.0,
    ):
        self.label = label
        self.bbox = bbox
        self.reading_order = reading_order
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"SuryaBlock(label={self.label!r}, bbox={self.bbox}, "
            f"order={self.reading_order}, conf={self.confidence:.2f})"
        )


# ============================================================================
# Model loading
# ============================================================================


def _get_surya_backend() -> str:
    """Determine the best available backend for Surya.

    Surya supports: cuda, mps, cpu. We detect what's available."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_surya(
    models_dir: Optional[Path] = None,
    backend: Optional[str] = None,
) -> "SuryaLayoutModel":
    """Load the Surya layout detection model.

    Args:
        models_dir: Custom directory for model weights. If None, uses Surya's
            default cache (~/.cache/surya/).
        backend: Override backend selection ("cuda", "mps", or "cpu").
            If None, auto-detects.

    Returns:
        A loaded SuryaLayoutModel wrapper.
    """
    backend = backend or _get_surya_backend()
    log.info("Loading Surya layout model (backend=%s) ...", backend)

    model = SuryaLayoutModel(backend=backend, models_dir=models_dir)
    model.load()
    return model


# ============================================================================
# SuryaLayoutModel wrapper
# ============================================================================


class SuryaLayoutModel:
    """Thin wrapper around surya-ocr's layout model.

    Encapsulates model loading and inference so the rest of the
    pipeline doesn't need to know surya-ocr internals.

    When surya-ocr is not installed, all methods return empty results
    and log a warning — this lets tests run without heavy dependencies.
    """

    def __init__(
        self,
        backend: str = "cpu",
        models_dir: Optional[Path] = None,
    ):
        self.backend = backend
        self.models_dir = models_dir
        self._model: Optional[object] = None
        self._loaded = False
        self._available = False

    def load(self) -> None:
        """Load Surya model weights into memory."""
        try:
            from surya.detection import batch_detection
            from surya.model.detection.model import (
                load_model as surya_load_model,
            )
            from surya.model.detection.processor import (
                load_processor as surya_load_processor,
            )
            from surya.settings import settings

            if self.models_dir:
                settings.DETECTOR_MODEL_CHECKPOINT = str(self.models_dir)

            self._model = surya_load_model()
            self._processor = surya_load_processor()
            self._batch_func = batch_detection
            self._loaded = True
            self._available = True
            log.info("Surya layout model loaded successfully.")
        except ImportError:
            log.warning(
                "surya-ocr not installed. Layout analysis will return empty results. "
                "Install with: pip install reach-core[surya]"
            )
            self._available = False
        except Exception:
            log.exception("Failed to load Surya layout model")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, images: list[Image.Image]) -> list[list[SuryaBlock]]:
        """Run layout detection on a list of page images.

        Args:
            images: One PIL Image per page.

        Returns:
            A list (per page) of lists of SuryaBlock objects.
        """
        if not self._available or not self._model:
            log.warning("Surya model not loaded — returning empty layout.")
            return [[] for _ in images]

        try:
            results = self._batch_func(images, [self._model], [self._processor])
            # results is a list of DetectionResult per page
            output: list[list[SuryaBlock]] = []
            for page_result in results:
                page_blocks: list[SuryaBlock] = []
                for b in page_result.bboxes:
                    # b is a Surya PolygonBox with label, bbox, etc.
                    block = SuryaBlock(
                        label=getattr(b, "label", "Text"),
                        bbox=list(getattr(b, "bbox", [0, 0, 100, 100])),
                        reading_order=getattr(b, "position", 0),
                        confidence=getattr(b, "confidence", 0.0),
                    )
                    page_blocks.append(block)
                # Sort by reading order
                page_blocks.sort(key=lambda x: x.reading_order)
                output.append(page_blocks)
            return output
        except Exception:
            log.exception("Surya detection failed")
            return [[] for _ in images]

    def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._processor = None
        self._loaded = False


# ============================================================================
# Layout block conversion
# ============================================================================


def convert_surya_blocks(
    surya_blocks: list[SuryaBlock],
    page_text: str = "",
    page_num: int = 0,
) -> list[TextBlock | TableBlock | ImageBlock | EquationBlock]:
    """Convert raw Surya detection output to Reach Pydantic block models.

    Args:
        surya_blocks: List of SuryaBlock objects from layout detection.
        page_text: Raw text extracted from the PDF content stream (for merging).
        page_num: Page number (for logging).

    Returns:
        List of Reach block models sorted by reading_order.
    """
    reach_blocks: list[TextBlock | TableBlock | ImageBlock | EquationBlock] = []

    for i, sb in enumerate(surya_blocks):
        block_type = map_surya_block_type(sb.label)
        bbox = BBox.from_list(sb.bbox)

        if block_type == BlockType.TABLE:
            block = TableBlock(
                block_type=BlockType.TABLE,
                bbox=bbox,
                reading_order=sb.reading_order,
                confidence=sb.confidence,
            )
        elif block_type == BlockType.EQUATION:
            block = EquationBlock(
                block_type=BlockType.EQUATION,
                bbox=bbox,
                reading_order=sb.reading_order,
                confidence=sb.confidence,
            )
        elif block_type == BlockType.FIGURE:
            block = ImageBlock(
                block_type=BlockType.FIGURE,
                bbox=bbox,
                reading_order=sb.reading_order,
                confidence=sb.confidence,
            )
        else:
            block = TextBlock(
                block_type=block_type,
                bbox=bbox,
                reading_order=sb.reading_order,
                confidence=sb.confidence,
                text="",  # text filled in by OCR or content-stream merge
            )

        reach_blocks.append(block)

    return sorted(reach_blocks, key=lambda b: b.reading_order)


def merge_text_with_layout(
    blocks: list[TextBlock | TableBlock | ImageBlock | EquationBlock],
    page_text: str,
) -> list[TextBlock | TableBlock | ImageBlock | EquationBlock]:
    """Merge extracted content-stream text into layout blocks.

    For digital-born pages: takes the raw text from the PDF content
    stream and distributes it across the TextBlocks that Surya detected.
    This is a naive approach — a production merge would use character-level
    position data to map text spans to blocks precisely.

    Args:
        blocks: Layout blocks from Surya (sorted by reading_order).
        page_text: Raw text extracted from the page's content stream.

    Returns:
        The same blocks with text populated where possible.
    """
    text_blocks = [b for b in blocks if isinstance(b, TextBlock)]

    if not text_blocks or not page_text.strip():
        return blocks

    # Simple approach: split text evenly among text blocks.
    # A production implementation would use the block bboxes to
    # find overlapping text runs from the content stream.
    text_segments = _naive_text_distribution(page_text, len(text_blocks))

    for block, seg in zip(text_blocks, text_segments):
        block.text = seg

    return blocks


def _naive_text_distribution(text: str, num_blocks: int) -> list[str]:
    """Distribute text across N blocks by paragraph boundaries.

    Falls back to character-level splitting for single-paragraph text.
    """
    if num_blocks <= 0:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) >= num_blocks:
        # Assign one paragraph per block (first N blocks get paragraphs)
        result = [""] * num_blocks
        for i in range(min(num_blocks, len(paragraphs))):
            result[i] = paragraphs[i]
        return result

    # Fewer paragraphs than blocks: split paragraphs into sentences
    import re

    sentences: list[str] = []
    for para in paragraphs:
        para_sents = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend(s.strip() for s in para_sents if s.strip())

    if len(sentences) >= num_blocks:
        result = [""] * num_blocks
        for i in range(min(num_blocks, len(sentences))):
            result[i] = sentences[i]
        return result

    # Fallback: character-level split
    if len(text) < num_blocks:
        result = [""] * num_blocks
        for i, ch in enumerate(text):
            result[i % num_blocks] += ch
        return result

    chunk_size = max(1, len(text) // num_blocks)
    result = []
    for i in range(num_blocks):
        start = i * chunk_size
        end = start + chunk_size if i < num_blocks - 1 else len(text)
        result.append(text[start:end])
    return result


# ============================================================================
# High-level layout analysis entry point
# ============================================================================


def analyze_layout(
    images: list[Image.Image],
    model: Optional[SuryaLayoutModel] = None,
    page_texts: Optional[list[str]] = None,
) -> list[list[TextBlock | TableBlock | ImageBlock | EquationBlock]]:
    """Run full layout analysis on rendered page images.

    This is the high-level entry point: load the model, detect blocks,
    convert to Reach types, merge content-stream text.

    Args:
        images: One PIL Image per page.
        model: An already-loaded SuryaLayoutModel. If None, loads a new one.
        page_texts: Optional list of per-page raw text from the PDF content
            stream. Used for merging on digital-born pages.

    Returns:
        A list (per page) of lists of Reach block objects.
    """
    if model is None:
        model = load_surya()

    if not model.is_available:
        log.warning("Surya not available — returning empty layout.")
        return [[] for _ in images]

    page_texts = page_texts or [""] * len(images)

    # Run detection
    surya_results = model.detect(images)

    # Convert and merge
    output: list[list[TextBlock | TableBlock | ImageBlock | EquationBlock]] = []
    for page_num, (surya_blocks, page_text) in enumerate(zip(surya_results, page_texts)):
        blocks = convert_surya_blocks(surya_blocks, page_text, page_num)
        blocks = merge_text_with_layout(blocks, page_text)
        output.append(blocks)

    return output
