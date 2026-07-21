"""
Tag tree builder — converts Phase 1 extraction JSON into a PDF/UA-2 tag tree.

Handles:
    - Mapping Surya block types to PDF/UA-2 tag types
    - Heading hierarchy inference from font size/weight/position
    - List detection from indentation and bullet patterns
    - Artifact marking (page numbers, running headers)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from reach.tagtree.tag_models import (
    BBox,
    ExtractionBlock,
    ExtractionDocument,
    ExtractionPage,
    FontInfo,
    SURYA_TAG_MAP,
    ARTIFACT_TYPES,
    HEADING_TYPES,
    TagNode,
    TagTree,
    TagTreeConfig,
    TagType,
)

log = logging.getLogger(__name__)

# ── Bullet / list detection patterns ─────────────────────────────────────────

BULLET_PATTERNS = re.compile(
    r"^\s*([\u2022\u2023\u25E6\u2043\u2219\u25CB\u25CF\u2713\u2714\-•◦▪▸►❖◆◇○●])\s"
)

NUMBERED_LIST_PATTERN = re.compile(
    r"^\s*(\d+[.)]\s|[a-zA-Z][.)]\s|[ivxlcdm]+[.)]\s)"
)


# ── Builder ──────────────────────────────────────────────────────────────────


class TagTreeBuilder:
    """Builds a PDF/UA-2 TagTree from Phase 1 extraction output."""

    def __init__(self, config: Optional[TagTreeConfig] = None) -> None:
        self.config = config or TagTreeConfig()

    def build(self, document: ExtractionDocument) -> TagTree:
        """Build a complete tag tree from an extraction document.

        Args:
            document: Phase 1 extraction output.

        Returns:
            Fully populated TagTree ready for enrichment and export.
        """
        root = TagNode(tag=TagType.DOCUMENT)
        reading_order: list = []

        detected_headings: list[TagNode] = []

        for page in document.pages:
            for block in page.blocks:
                node = self._block_to_tag(block, page)
                if node is None:
                    continue

                root.children.append(node)

                # Track headings for hierarchy inference
                if node.tag in HEADING_TYPES:
                    detected_headings.append(node)

        # Infer heading hierarchy
        if detected_headings:
            self._infer_heading_levels(detected_headings)

        # Build reading order from node sequence
        seq = 0
        for i, child in enumerate(root.children):
            reading_order.append(
                {
                    "node_id": str(i),
                    "page_number": child.page_number,
                    "sequence": seq,
                }
            )
            seq += 1

        return TagTree(
            document=document,
            root=root,
            language=self.config.default_language,
        )

    def _block_to_tag(
        self, block: ExtractionBlock, page: ExtractionPage
    ) -> Optional[TagNode]:
        """Convert a single extraction block to a tag tree node.

        Args:
            block: Phase 1 extraction block.
            page: The source page (for coordinate normalization).

        Returns:
            TagNode or None if the block should be skipped.
        """
        # Map Surya type to TagType
        tag_type = SURYA_TAG_MAP.get(block.type, TagType.P)

        node = TagNode(
            tag=tag_type,
            text=block.text,
            page_number=block.page_number,
            bbox=block.bbox,
        )

        # Handle special cases based on Surya block type
        if block.type == "SectionHeader":
            node = self._handle_heading(block, page)
        elif block.type in ("List", "List-item"):
            node = self._handle_list_item(block, tag_type)
        elif block.type == "Table":
            node = self._handle_table_block(block, tag_type)
        elif block.type in ("PageHeader", "PageFooter", "PageNumber"):
            node.tag = TagType.ARTIFACT
        elif block.type == "Title":
            node.tag = TagType.H1

        return node

    def _handle_heading(
        self, block: ExtractionBlock, page: ExtractionPage
    ) -> TagNode:
        """Handle a SectionHeader block — starts as generic H, level inferred later.

        Args:
            block: Extraction block for a heading.
            page: Source page.

        Returns:
            TagNode with H tag (level inferred later).
        """
        font = block.font_info or FontInfo()

        node = TagNode(
            tag=TagType.H,
            text=block.text,
            page_number=block.page_number,
            bbox=block.bbox,
            attributes={
                "font_size": str(font.size),
                "font_family": font.family,
                "bold": str(font.bold),
            },
        )

        return node

    def _handle_list_item(
        self, block: ExtractionBlock, tag_type: TagType
    ) -> TagNode:
        """Handle a list item — wrap in L/LI structure.

        Detects if this is a list item by checking for bullets or numbering,
        and creates the L → LI → (Lbl + LBody) hierarchy.

        Args:
            block: Extraction block.
            tag_type: Raw mapped tag type.

        Returns:
            TagNode with L tag containing LI children.
        """
        # Create the list container (L)
        list_node = TagNode(tag=TagType.L)

        # Create the list item (LI)
        li_node = TagNode(tag=TagType.LI)

        # Try to detect and extract bullet/number label
        bullet_match = BULLET_PATTERNS.match(block.text)
        number_match = NUMBERED_LIST_PATTERN.match(block.text)

        if bullet_match:
            label = TagNode(tag=TagType.LBL, text=bullet_match.group(0).strip())
            body = TagNode(
                tag=TagType.LBODY,
                text=block.text[bullet_match.end() :].strip(),
            )
            li_node.children = [label, body]
        elif number_match:
            label = TagNode(tag=TagType.LBL, text=number_match.group(0).strip())
            body = TagNode(
                tag=TagType.LBODY,
                text=block.text[number_match.end() :].strip(),
            )
            li_node.children = [label, body]
        else:
            # No explicit bullet — wrap whole text in LBody
            body = TagNode(tag=TagType.LBODY, text=block.text)
            li_node.children = [body]

        list_node.children = [li_node]
        list_node.page_number = block.page_number
        list_node.bbox = block.bbox

        return list_node

    def _handle_table_block(
        self, block: ExtractionBlock, tag_type: TagType
    ) -> TagNode:
        """Handle a Table block — basic Table tag.

        Detailed table structure (TR/TD/TH, headers, scope) is handled
        by tables.py enrichment step.

        Args:
            block: Extraction block for a table.
            tag_type: Tag type (TagType.TABLE).

        Returns:
            TagNode with Table tag.
        """
        return TagNode(
            tag=tag_type,
            text=block.text,
            page_number=block.page_number,
            bbox=block.bbox,
        )

    # ── Heading Hierarchy Inference ───────────────────────────────────────

    def _infer_heading_levels(self, headings: list[TagNode]) -> None:
        """Infer H1-H6 levels from heading font sizes and positions.

        Strategy:
            1. Group headings by font size (descending).
            2. Assign levels starting from H1 for the largest size.
            3. Cap at H6.

        Args:
            headings: List of heading TagNodes (mutated in place).
        """
        if not headings:
            return

        # Collect font sizes
        sizes: list[float] = []
        for h in headings:
            font_size_str = h.attributes.get("font_size", "0")
            try:
                sizes.append(float(font_size_str))
            except ValueError:
                sizes.append(0.0)

        if not sizes or max(sizes) == 0:
            return

        # Get unique sizes sorted descending
        unique_sizes = sorted(set(s for s in sizes if s > 0), reverse=True)

        # Build size → level mapping
        size_to_level: dict[float, int] = {}
        for i, size in enumerate(unique_sizes[: self.config.max_heading_depth]):
            size_to_level[size] = i + 1  # 1-indexed

        # Detect numbering patterns (e.g., "1.", "1.2", "1.2.3")
        numbering_depths = self._detect_numbering_depths(headings)

        # Assign levels — use numbering depth if available, otherwise font size
        for h in headings:
            font_size = float(h.attributes.get("font_size", "12.0"))
            assigned_level = size_to_level.get(font_size, 1)

            # If numbering depth suggests a deeper level, use it
            numbering_depth = numbering_depths.get(h.text.strip(), 0)
            if numbering_depth > assigned_level:
                assigned_level = min(numbering_depth, self.config.max_heading_depth)

            level_tag = getattr(TagType, f"H{assigned_level}", TagType.H)
            h.tag = level_tag
            h.attributes["level"] = str(assigned_level)

    @staticmethod
    def _detect_numbering_depths(headings: list[TagNode]) -> dict[str, int]:
        """Detect heading depth from numbering patterns (1, 1.1, 1.1.1 etc.).

        Args:
            headings: List of heading nodes.

        Returns:
            Mapping of heading text → detected depth.
        """
        depths: dict[str, int] = {}
        number_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s")

        for h in headings:
            m = number_pattern.match(h.text.strip())
            if m:
                parts = m.group(1).split(".")
                depths[h.text.strip()] = len(parts)
        return depths


# ── Convenience function ─────────────────────────────────────────────────────


def build_tag_tree(
    extraction_json: dict,
    config: Optional[TagTreeConfig] = None,
) -> TagTree:
    """Build a tag tree from Phase 1 extraction JSON.

    Args:
        extraction_json: Parsed JSON from Phase 1 extraction.
        config: Optional configuration overrides.

    Returns:
        Completed TagTree.
    """
    doc = ExtractionDocument.model_validate(extraction_json)
    builder = TagTreeBuilder(config=config)
    return builder.build(doc)


def build_tag_tree_from_file(
    input_path: str,
    config: Optional[TagTreeConfig] = None,
) -> TagTree:
    """Build a tag tree from a Phase 1 extraction JSON file.

    Args:
        input_path: Path to extraction JSON file.
        config: Optional configuration overrides.

    Returns:
        Completed TagTree.
    """
    import json
    from pathlib import Path

    data = json.loads(Path(input_path).read_text())
    return build_tag_tree(data, config)
