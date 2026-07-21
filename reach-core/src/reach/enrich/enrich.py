"""
Content enrichment using Qwen 3.5 9B VLM via llama.cpp.

Handles:
    - Alt text generation for figures (charts, diagrams, photos)
    - Decorative vs. informational image classification
    - Low-quality alt text filtering (empty, generic, hallucinated)
    - Figure region cropping and VLM inference

The VLM call is isolated behind the `EnrichmentModel` interface so tests
can mock it without downloading the 5.68 GB GGUF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

from reach.tagtree.tag_models import (
    AltTextResult,
    EnrichmentReport,
    ExtractionBlock,
    ExtractionDocument,
    TagNode,
    TagTreeConfig,
)

log = logging.getLogger(__name__)

# ── Prompt Templates ─────────────────────────────────────────────────────────

ALT_TEXT_SYSTEM_PROMPT = (
    "You are an accessibility assistant. Your task is to describe visual content "
    "for a blind reader. Be factual, concise, and focus on information content. "
    "Never speculate about what an image might show — only describe what you actually see. "
    "If an image is purely decorative (borders, icons, logos without info), say 'DECORATIVE'."
)

ALT_TEXT_USER_TEMPLATE = (
    "Describe this {figure_type} for a blind reader. "
    "Include what type of visual it is and its key information. "
    "Be concise (2-3 sentences)."
)

DECORATIVE_CLASSIFICATION_TEMPLATE = (
    "Is this image purely decorative (has no informational content important for "
    "understanding the document)? Answer only 'YES' or 'NO'."
)

# Generic phrases that indicate low-quality alt text
GENERIC_ALT_TEXT_PATTERNS: list[str] = [
    "a chart",
    "a diagram",
    "an image",
    "a figure",
    "a photo",
    "a picture",
    "a graph",
    "a table",
    "a screenshot",
    "a drawing",
    "a plot",
    "an illustration",
]

# Hallucination indicators — phrases that suggest fabrications
HALLUCINATION_INDICATORS: list[str] = [
    "I'm not sure",
    "I cannot see",
    "this might be",
    "possibly",
    "unclear image",
    "appears to be",
    "could be a",
    "what appears to be",
]


def classify_figure_type(block_type: str) -> str:
    """Map Surya block type to descriptive figure type for prompt.

    Args:
        block_type: Surya block type string.

    Returns:
        Descriptive string for VLM prompt.
    """
    type_map: dict[str, str] = {
        "Figure": "figure",
        "Picture": "picture",
        "Table": "table",
        "Formula": "mathematical formula",
        "Chart": "chart",
        "Diagram": "diagram",
        "Graph": "graph",
        "Photo": "photograph",
    }
    return type_map.get(block_type, "figure")


# ── EnrichmentModel Interface ────────────────────────────────────────────────


class EnrichmentModel(Protocol):
    """Interface for the VLM enrichment model (Qwen 3.5 9B via llama.cpp).

    This protocol allows the real llama.cpp model and mock implementations
    to be used interchangeably.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        image_data: Optional[bytes] = None,
        image_path: Optional[Path] = None,
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from the VLM.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            image_data: Raw image bytes (for vision input).
            image_path: Path to image file (alternative to image_data).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        ...

    def is_available(self) -> bool:
        """Check if the VLM model is loaded and ready.

        Returns:
            True if the model is available for inference.
        """
        ...


# ── LlamaCppEnrichmentModel ──────────────────────────────────────────────────


class LlamaCppEnrichmentModel:
    """Enrichment model using llama-cpp-python with Qwen 3.5 9B GGUF.

    This is the real inference implementation. Requires the GGUF model file
    at the configured path.
    """

    def __init__(self, config: TagTreeConfig) -> None:
        self.config = config
        self._model: Optional[object] = None
        self._loaded: bool = False

    def is_available(self) -> bool:
        """Check if the model file exists and llama.cpp is importable."""
        if self._loaded:
            return True

        model_path = Path(self.config.vlm_model_path).expanduser()
        if not model_path.exists():
            log.debug("VLM model not found at %s", model_path)
            return False

        try:
            import llama_cpp  # noqa: F401, I001
            return True
        except ImportError:
            log.debug("llama-cpp-python not installed")
            return False

    def _load_model(self) -> Optional[object]:
        """Lazy-load the llama.cpp model.

        Returns:
            Llama model instance or None on failure.
        """
        if self._loaded and self._model is not None:
            return self._model

        try:
            from llama_cpp import Llama

            model_path = Path(self.config.vlm_model_path).expanduser()
            log.info("Loading Qwen 3.5 9B from %s ...", model_path)

            self._model = Llama(
                model_path=str(model_path),
                n_ctx=8192,
                n_threads=4,
                verbose=False,
            )
            self._loaded = True
            log.info("Qwen 3.5 9B loaded successfully")
            return self._model
        except Exception as e:
            log.error("Failed to load VLM model: %s", e)
            return None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        image_data: Optional[bytes] = None,
        image_path: Optional[Path] = None,
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> str:
        """Generate text with optional image input.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            image_data: Raw image bytes.
            image_path: Path to image file.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        model = self._load_model()
        if model is None:
            return ""

        # Build the full prompt with optional system prompt
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|system|>\n{system_prompt}\n"
        full_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"

        # Note: llama-cpp-python supports image inputs via `image_data` parameter
        # in newer versions. For Qwen 3.5 native multimodal, images are passed
        # as base64 in the chat format or via the mmproj (no longer needed).
        kwargs: dict = {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "echo": False,
            "stop": ["<|user|>", "<|system|>", "<|endoftext|>"],
        }

        # Pass image data if available
        if image_data is not None:
            kwargs["image_data"] = image_data

        try:
            result = model(**kwargs)
            output = result.get("choices", [{}])[0].get("text", "").strip()
            return output
        except Exception as e:
            log.error("VLM generation failed: %s", e)
            return ""


# ── Alt Text Quality Filter ──────────────────────────────────────────────────


def filter_alt_text(
    alt_text: str,
    config: TagTreeConfig,
) -> tuple[str, bool, str]:
    """Filter alt text for quality — reject low-quality or hallucinated output.

    Args:
        alt_text: Raw alt text from the VLM.
        config: Enrichment configuration.

    Returns:
        (filtered_text, was_rejected, reason) tuple.
    """
    text = alt_text.strip()

    # Empty or too short
    if not text:
        return text, True, "empty"

    # Reject generic descriptions FIRST (before length checking,
    # since generic phrases like "a chart" are intentionally short)
    text_lower = text.lower()

    for generic in config.reject_generic_phrases:
        if text_lower.strip().rstrip(".") == generic:
            return text, True, f"generic: '{generic}'"

    if len(text) < config.min_alt_text_length:
        return text, True, f"too short ({len(text)} < {config.min_alt_text_length})"

    # Too long (likely rambling / hallucination)
    if len(text) > config.max_alt_text_length:
        return text, True, f"too long ({len(text)} > {config.max_alt_text_length})"

    # Reject hallucination indicators
    for indicator in HALLUCINATION_INDICATORS:
        if indicator.lower() in text_lower:
            return text, True, f"hallucination: contains '{indicator}'"

    return text, False, ""


# ── Figure Classification ────────────────────────────────────────────────────


def classify_decorative(
    alt_text: str,
) -> bool:
    """Classify whether an image is decorative based on VLM alt text.

    Args:
        alt_text: Alt text from the VLM (or classification output).

    Returns:
        True if the image is decorative.
    """
    text_lower = alt_text.lower().strip()

    # VLM explicitly marked as decorative
    if text_lower == "decorative" or text_lower.startswith("decorative"):
        return True

    # Check if the VLM's classification response indicates decorative
    if text_lower in ("yes", "yes.", "yes, it is decorative"):
        return True

    if text_lower in ("no", "no.", "no, it is not decorative"):
        return False

    # Heuristic: very short descriptions of simple graphics
    decorative_indicators = [
        "decorative border",
        "decorative line",
        "logo",
        "icon",
        "separator",
        "background",
        "watermark",
    ]

    for indicator in decorative_indicators:
        if indicator in text_lower:
            return True

    return False


# ── Enrichment Pipeline ──────────────────────────────────────────────────────


class Enricher:
    """Content enrichment pipeline for Phase 2.

    Processes figures in the tag tree:
        1. Crop figure regions from rendered pages.
        2. Send to Qwen VLM for alt text generation.
        3. Filter low-quality alt text.
        4. Classify decorative vs. informational.
    """

    def __init__(
        self,
        model: EnrichmentModel,
        config: Optional[TagTreeConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TagTreeConfig()

    def enrich(
        self,
        document: ExtractionDocument,
        page_images: Optional[dict[int, bytes]] = None,
    ) -> EnrichmentReport:
        """Run enrichment on all figures in the document.

        Args:
            document: Phase 1 extraction document.
            page_images: Optional dict of page_number → raw page image bytes
                         for cropping figure regions.

        Returns:
            EnrichmentReport with results for all figures.
        """
        report = EnrichmentReport()

        if not self.config.enable_enrichment:
            log.info("Enrichment disabled — skipping VLM processing")
            return report

        if not self.model.is_available():
            log.warning("VLM model not available — skipping enrichment")
            report.errors.append("VLM model not available")
            return report

        figures = self._collect_figures(document)

        for block in figures:
            result = self._process_figure(block, page_images)
            report.results.append(result)
            report.figures_processed += 1

            if result.rejected:
                report.alt_texts_rejected += 1
            else:
                report.alt_texts_generated += 1

            if result.is_decorative:
                report.decorative_classified += 1

        log.info(
            "Enrichment complete: %d figures, %d alt texts, %d rejected, %d decorative",
            report.figures_processed,
            report.alt_texts_generated,
            report.alt_texts_rejected,
            report.decorative_classified,
        )

        return report

    def _collect_figures(
        self, document: ExtractionDocument
    ) -> list[ExtractionBlock]:
        """Collect all figure blocks from the document.

        Args:
            document: Extraction document.

        Returns:
            List of figure-type blocks.
        """
        figures: list[ExtractionBlock] = []
        figure_types = {"Figure", "Picture", "Chart", "Diagram", "Photo", "Graph"}

        for page in document.pages:
            for block in page.blocks:
                if block.type in figure_types:
                    figures.append(block)

        return figures

    def _process_figure(
        self,
        block: ExtractionBlock,
        page_images: Optional[dict[int, bytes]] = None,
    ) -> AltTextResult:
        """Process a single figure block through the VLM pipeline.

        Args:
            block: Figure extraction block.
            page_images: Optional page image data for cropping.

        Returns:
            AltTextResult with alt text and quality info.
        """
        node_id = f"fig_{block.page_number}_{block.bbox.x0:.0f}_{block.bbox.y0:.0f}"

        # Determine figure type for prompt
        figure_type = classify_figure_type(block.type)

        # Crop the figure region if page image is available
        image_data: Optional[bytes] = None
        if page_images and block.page_number in page_images:
            image_data = self._crop_region(
                page_images[block.page_number],
                block.bbox,
            )

        # Generate alt text via VLM
        prompt = ALT_TEXT_USER_TEMPLATE.format(figure_type=figure_type)
        raw_alt_text = self.model.generate(
            prompt=prompt,
            system_prompt=ALT_TEXT_SYSTEM_PROMPT,
            image_data=image_data,
        )

        # Filter for quality
        filtered_text, rejected, reason = filter_alt_text(raw_alt_text, self.config)

        # Classify decorative
        is_decorative = classify_decorative(raw_alt_text)

        return AltTextResult(
            node_id=node_id,
            alt_text=filtered_text if not rejected else "",
            is_decorative=is_decorative,
            confidence=1.0 if not rejected else 0.0,
            rejected=rejected,
            rejection_reason=reason,
            model_used="qwen3.5-9b",
        )

    @staticmethod
    def _crop_region(page_image: bytes, bbox) -> Optional[bytes]:
        """Crop a region from a page image.

        Args:
            page_image: Raw page image bytes (PNG/JPEG).
            bbox: Bounding box in page coordinates.

        Returns:
            Cropped region as PNG bytes, or None on failure.
        """
        try:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(page_image))
            # BBox uses top-left origin; Pillow uses (left, upper, right, lower)
            cropped = img.crop((
                int(bbox.x0),
                int(bbox.y0),
                int(bbox.x1),
                int(bbox.y1),
            ))

            buf = BytesIO()
            cropped.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            log.warning("Failed to crop figure region: %s", e)
            return None

    def apply_enrichment(
        self,
        root: TagNode,
        report: EnrichmentReport,
    ) -> None:
        """Apply enrichment results to tag tree nodes.

        Walks the tag tree and sets alt_text and is_decorative on Figure nodes
        based on the enrichment report.

        Args:
            root: Root tag tree node (mutated in place).
            report: Enrichment report with results.
        """
        # Build lookup from node_id → result
        result_map: dict[str, AltTextResult] = {
            r.node_id: r for r in report.results
        }

        self._walk_and_apply(root, result_map)

    def _walk_and_apply(
        self, node: TagNode, result_map: dict[str, AltTextResult]
    ) -> None:
        """Recursively walk tree and apply enrichment results.

        Args:
            node: Current node.
            result_map: Map of node_id → AltTextResult.
        """
        # Generate a node_id matching the pattern from _process_figure
        if node.bbox:
            node_id = f"fig_{node.page_number}_{node.bbox.x0:.0f}_{node.bbox.y0:.0f}"
        else:
            node_id = ""

        if node_id in result_map:
            result = result_map[node_id]
            node.alt_text = result.alt_text
            node.is_decorative = result.is_decorative

        for child in node.children:
            self._walk_and_apply(child, result_map)


# ── Mock for Testing ─────────────────────────────────────────────────────────


class MockEnrichmentModel:
    """Mock VLM model for testing — returns deterministic responses."""

    def __init__(self, responses: Optional[dict[str, str]] = None) -> None:
        self.responses = responses or {}
        self.call_count: int = 0
        self.last_prompt: str = ""

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        image_data: Optional[bytes] = None,
        image_path: Optional[Path] = None,
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> str:
        """Return a mock response based on prompt content.

        Args:
            prompt: User prompt.
            system_prompt: System prompt (unused in mock).
            image_data: Image bytes (unused in mock).
            image_path: Image path (unused in mock).
            max_tokens: Max tokens (unused).
            temperature: Temperature (unused).

        Returns:
            Mock response string.
        """
        self.call_count += 1
        self.last_prompt = prompt

        # Look up in configured responses
        for key, value in self.responses.items():
            if key in prompt:
                return value

        # Default mock response
        if "chart" in prompt.lower() or "graph" in prompt.lower():
            return "A bar chart showing quarterly sales data for 2023. Sales increased from $10M in Q1 to $15M in Q4."
        elif "diagram" in prompt.lower():
            return "A flowchart diagram showing the software development lifecycle with stages: Plan, Design, Implement, Test, Deploy."
        elif "formula" in prompt.lower() or "mathematical" in prompt.lower():
            return "A mathematical formula showing the quadratic equation: x = (-b ± √(b² - 4ac)) / 2a."
        elif "photo" in prompt.lower() or "picture" in prompt.lower():
            return "A photograph of a laboratory setup with beakers and test tubes on a bench."
        elif "decorative" in prompt.lower():
            return "DECORATIVE"
        else:
            return "A figure containing visual information."
