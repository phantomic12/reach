"""Content enrichment using Qwen 3.5 9B VLM.

Modules:
    enrich: VLM alt text generation, quality filtering, figure classification.
"""

from reach.enrich.enrich import (
    ALT_TEXT_SYSTEM_PROMPT,
    ALT_TEXT_USER_TEMPLATE,
    DECORATIVE_CLASSIFICATION_TEMPLATE,
    Enricher,
    EnrichmentModel,
    LlamaCppEnrichmentModel,
    MockEnrichmentModel,
    classify_decorative,
    classify_figure_type,
    filter_alt_text,
)

__all__ = [
    "ALT_TEXT_SYSTEM_PROMPT",
    "ALT_TEXT_USER_TEMPLATE",
    "DECORATIVE_CLASSIFICATION_TEMPLATE",
    "Enricher",
    "EnrichmentModel",
    "LlamaCppEnrichmentModel",
    "MockEnrichmentModel",
    "classify_decorative",
    "classify_figure_type",
    "filter_alt_text",
]
