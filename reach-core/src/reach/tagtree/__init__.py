"""Tag tree construction — Phase 2 of the Reach pipeline.

Converts Phase 1 extraction JSON into a PDF/UA-2 conformant tag tree,
with reading order, table structure, metadata, and content enrichment.

Modules:
    tag_models:    Pydantic models for tag tree, extraction input, enrichment.
    builder:       Tag tree construction from extraction blocks.
    reading_order: XY-cut algorithm for multi-column reading order.
    tables:        Table structure detection (headers, scope, merged cells).
    metadata:      Title extraction, language detection, XMP generation.
"""

from reach.tagtree.builder import TagTreeBuilder, build_tag_tree, build_tag_tree_from_file
from reach.tagtree.metadata import build_metadata, detect_document_language, extract_title, generate_xmp_metadata
from reach.tagtree.reading_order import (
    ReadingOrderConfig,
    classify_special_regions,
    compute_reading_order,
    xy_cut_sort,
)
from reach.tagtree.tables import (
    TableCell,
    TableStructure,
    assign_scope_attributes,
    build_table_tag_node,
    detect_merged_cells,
    detect_table_structure,
)
from reach.tagtree.tag_models import (
    BBox,
    EnrichmentReport,
    ExtractionBlock,
    ExtractionDocument,
    ExtractionPage,
    FontInfo,
    LanguageSpan,
    ReadingOrderItem,
    SURYA_TAG_MAP,
    TagNode,
    TagTree,
    TagTreeConfig,
    TagType,
)

__all__ = [
    # Models
    "BBox",
    "EnrichmentReport",
    "ExtractionBlock",
    "ExtractionDocument",
    "ExtractionPage",
    "FontInfo",
    "LanguageSpan",
    "ReadingOrderItem",
    "ReadingOrderConfig",
    "SURYA_TAG_MAP",
    "TableCell",
    "TableStructure",
    "TagNode",
    "TagTree",
    "TagTreeConfig",
    "TagType",
    # Builder
    "TagTreeBuilder",
    "build_tag_tree",
    "build_tag_tree_from_file",
    # Reading Order
    "compute_reading_order",
    "xy_cut_sort",
    "classify_special_regions",
    # Tables
    "detect_table_structure",
    "build_table_tag_node",
    "detect_merged_cells",
    "assign_scope_attributes",
    # Metadata
    "build_metadata",
    "extract_title",
    "detect_document_language",
    "generate_xmp_metadata",
]
