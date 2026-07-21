"""
Reading order engine — determines logical reading sequence for multi-column layouts.

Implements XY-cut recursive projection profile algorithm for column-aware ordering.
Also handles sidebars, pull quotes, footnotes as separate flow items.

Reference: "XY-cut Algorithm for Document Layout" (Nagy, 2000).
The algorithm recursively projects bounding boxes onto the X or Y axis,
finding gaps (valleys in the projection histogram) that indicate column/row boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from reach.tagtree.tag_models import (
    BBox,
    ExtractionBlock,
    ExtractionPage,
    ReadingOrderItem,
)

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ReadingOrderConfig:
    """Configuration for reading order detection."""

    column_gap_threshold: float = 20.0  # Minimum gap to consider a column boundary
    sidebar_max_width_ratio: float = 0.35  # Max width ratio for sidebar detection
    footnote_min_y_ratio: float = 0.85  # Bottom of page threshold for footnotes
    header_max_y_ratio: float = 0.08  # Top of page threshold for headers
    enable_xy_cut: bool = True


# ── XY-Cut Implementation ────────────────────────────────────────────────────


@dataclass
class LayoutRegion:
    """A rectangular region of the page containing content blocks."""

    x0: float
    y0: float
    x1: float
    y1: float
    blocks: list[ExtractionBlock] = field(default_factory=list)
    regions: list["LayoutRegion"] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def is_leaf(self) -> bool:
        return not self.regions


def xy_cut_sort(
    blocks: list[ExtractionBlock],
    page_width: float,
    page_height: float,
    config: Optional[ReadingOrderConfig] = None,
) -> list[ExtractionBlock]:
    """Sort blocks into reading order using XY-cut recursive projection.

    The algorithm:
        1. Project all block bounding boxes onto the Y axis.
        2. Find gaps larger than `column_gap_threshold` — these are row separators.
        3. For each row, recursively project onto the X axis to find columns.
        4. Read columns left-to-right, rows top-to-bottom.

    Args:
        blocks: Unsorted extraction blocks for a single page.
        page_width: Page width in points.
        page_height: Page height in points.
        config: Reading order configuration.

    Returns:
        Blocks sorted in reading order with updated `reading_order` fields.
    """
    if config is None:
        config = ReadingOrderConfig()

    if not blocks:
        return []

    if not config.enable_xy_cut:
        # Simple top-to-bottom, left-to-right fallback
        sorted_blocks = sorted(
            blocks, key=lambda b: (b.bbox.y0, b.bbox.x0)
        )
        for i, b in enumerate(sorted_blocks):
            b.reading_order = i
        return sorted_blocks

    root = LayoutRegion(
        x0=0.0,
        y0=0.0,
        x1=page_width,
        y1=page_height,
        blocks=list(blocks),
    )

    _xy_cut_recursive(root, config, depth=0)

    # Extract blocks in DFS order (top-down, left-right)
    ordered: list[ExtractionBlock] = []
    _collect_blocks(root, ordered)

    for i, b in enumerate(ordered):
        b.reading_order = i

    return ordered


def _xy_cut_recursive(
    region: LayoutRegion,
    config: ReadingOrderConfig,
    depth: int,
    max_depth: int = 10,
) -> None:
    """Recursively partition a region using XY-cut.

    Alternates between vertical (Y) and horizontal (X) cuts based on depth.
    Even depth → vertical cuts (row separators).
    Odd depth → horizontal cuts (column separators).

    Args:
        region: The region to partition.
        config: Reading order configuration.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth.
    """
    if depth >= max_depth or len(region.blocks) <= 1:
        return

    # Alternate cut direction: even depth = vertical (rows), odd = horizontal (columns)
    if depth % 2 == 0:
        _cut_vertical(region, config, depth)
    else:
        _cut_horizontal(region, config, depth)


def _cut_vertical(
    region: LayoutRegion,
    config: ReadingOrderConfig,
    depth: int,
) -> None:
    """Cut region into rows by projecting onto Y axis.

    Args:
        region: Region to cut.
        config: Reading order configuration.
        depth: Current recursion depth.
    """
    blocks = region.blocks
    if len(blocks) <= 1:
        return

    # Sort by y0 for projection
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox.y0)

    # Build Y-axis projection: find gaps between block rows
    gaps = _find_y_gaps(sorted_blocks, config.column_gap_threshold)

    if not gaps:
        # No significant gaps → proceed to X-cut
        _cut_horizontal(region, config, depth)
        return

    # Partition blocks into rows based on gaps
    row_regions = _partition_by_gaps(sorted_blocks, gaps, axis="y", parent=region)

    if len(row_regions) <= 1:
        _cut_horizontal(region, config, depth)
        return

    region.regions = row_regions
    region.blocks = []

    for row in row_regions:
        _xy_cut_recursive(row, config, depth + 1)


def _cut_horizontal(
    region: LayoutRegion,
    config: ReadingOrderConfig,
    depth: int,
) -> None:
    """Cut region into columns by projecting onto X axis.

    Args:
        region: Region to cut.
        config: Reading order configuration.
        depth: Current recursion depth.
    """
    blocks = region.blocks
    if len(blocks) <= 1:
        return

    # Sort by x0 for projection
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox.x0)

    # Build X-axis projection: find gaps between block columns
    gaps = _find_x_gaps(sorted_blocks, config.column_gap_threshold)

    if not gaps:
        return

    # Partition blocks into columns based on gaps
    col_regions = _partition_by_gaps(sorted_blocks, gaps, axis="x", parent=region)

    if len(col_regions) <= 1:
        return

    region.regions = col_regions
    region.blocks = []

    for col in col_regions:
        _xy_cut_recursive(col, config, depth + 1)


def _find_y_gaps(
    blocks: list[ExtractionBlock],
    threshold: float,
) -> list[float]:
    """Find Y-axis gaps (row separators) between blocks.

    A gap exists where the top (y0) of block B is significantly below
    the bottom (y1) of block A.

    Args:
        blocks: Blocks sorted by y0.
        threshold: Minimum gap size in points.

    Returns:
        List of Y coordinates where gaps occur (midpoints between rows).
    """
    if len(blocks) < 2:
        return []

    gaps: list[float] = []
    current_bottom = blocks[0].bbox.y1

    for i in range(1, len(blocks)):
        b = blocks[i]
        if b.bbox.y0 - current_bottom >= threshold:
            # Gap found at midpoint
            gap_mid = (current_bottom + b.bbox.y0) / 2
            gaps.append(gap_mid)
        current_bottom = max(current_bottom, b.bbox.y1)

    return gaps


def _find_x_gaps(
    blocks: list[ExtractionBlock],
    threshold: float,
) -> list[float]:
    """Find X-axis gaps (column separators) between blocks.

    A gap exists where the left (x0) of block B is significantly right of
    the right (x1) of block A.

    Args:
        blocks: Blocks sorted by x0.
        threshold: Minimum gap size in points.

    Returns:
        List of X coordinates where gaps occur.
    """
    if len(blocks) < 2:
        return []

    gaps: list[float] = []
    current_right = blocks[0].bbox.x1

    for i in range(1, len(blocks)):
        b = blocks[i]
        if b.bbox.x0 - current_right >= threshold:
            gap_mid = (current_right + b.bbox.x0) / 2
            gaps.append(gap_mid)
        current_right = max(current_right, b.bbox.x1)

    return gaps


def _partition_by_gaps(
    blocks: list[ExtractionBlock],
    gaps: list[float],
    axis: str,
    parent: LayoutRegion,
) -> list[LayoutRegion]:
    """Partition blocks into regions based on gap coordinates.

    Args:
        blocks: Sorted blocks.
        gaps: Gap coordinates (Y or X midpoints).
        axis: "y" for rows, "x" for columns.
        parent: Parent region for boundary inheritance.

    Returns:
        List of child LayoutRegions.
    """
    regions: list[LayoutRegion] = []
    gaps = sorted(gaps)

    # Add start and end boundaries
    if axis == "y":
        boundaries = [parent.y0] + gaps + [parent.y1]
    else:
        boundaries = [parent.x0] + gaps + [parent.x1]

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]

        # Find blocks that fall in this band
        region_blocks: list[ExtractionBlock] = []
        for b in blocks:
            if axis == "y":
                if lo <= b.bbox.y0 < hi or (i == len(boundaries) - 2 and lo <= b.bbox.y0 <= hi):
                    region_blocks.append(b)
            else:
                if lo <= b.bbox.x0 < hi or (i == len(boundaries) - 2 and lo <= b.bbox.x0 <= hi):
                    region_blocks.append(b)

        if region_blocks:
            if axis == "y":
                region = LayoutRegion(
                    x0=parent.x0,
                    y0=lo,
                    x1=parent.x1,
                    y1=hi,
                    blocks=region_blocks,
                )
            else:
                region = LayoutRegion(
                    x0=lo,
                    y0=parent.y0,
                    x1=hi,
                    y1=parent.y1,
                    blocks=region_blocks,
                )
            regions.append(region)

    return regions


def _collect_blocks(region: LayoutRegion, output: list[ExtractionBlock]) -> None:
    """DFS traversal: collect blocks from leaf regions in order.

    Args:
        region: Current region.
        output: Output list (mutated).
    """
    if region.is_leaf:
        output.extend(region.blocks)
    else:
        for child in region.regions:
            _collect_blocks(child, output)


# ── Sidebar / Footnote / Header Detection ────────────────────────────────────


def classify_special_regions(
    blocks: list[ExtractionBlock],
    page_width: float,
    page_height: float,
    config: Optional[ReadingOrderConfig] = None,
) -> dict[str, list[ExtractionBlock]]:
    """Classify blocks into special categories: headers, footnotes, sidebars.

    Args:
        blocks: Page blocks.
        page_width: Page width.
        page_height: Page height.
        config: Configuration.

    Returns:
        Dict with keys "headers", "footnotes", "sidebars", "main_content".
    """
    if config is None:
        config = ReadingOrderConfig()

    result: dict[str, list[ExtractionBlock]] = {
        "headers": [],
        "footnotes": [],
        "sidebars": [],
        "main_content": [],
    }

    for b in blocks:
        x_rel = b.bbox.x0 / page_width if page_width else 0
        y_rel = b.bbox.y0 / page_height if page_height else 0
        width_rel = b.bbox.width / page_width if page_width else 1

        # Headers: top of page
        if y_rel < config.header_max_y_ratio:
            result["headers"].append(b)
        # Footnotes: bottom of page
        elif y_rel > config.footnote_min_y_ratio:
            result["footnotes"].append(b)
        # Sidebars: narrow column on left or right
        elif width_rel < config.sidebar_max_width_ratio and (
            x_rel < 0.15 or x_rel > 0.75
        ):
            result["sidebars"].append(b)
        else:
            result["main_content"].append(b)

    return result


# ── Public API ────────────────────────────────────────────────────────────────


def compute_reading_order(
    page: ExtractionPage,
    config: Optional[ReadingOrderConfig] = None,
) -> ExtractionPage:
    """Compute reading order for a single page using XY-cut.

    Args:
        page: Extraction page with unsorted blocks.
        config: Reading order configuration.

    Returns:
        Same page with blocks sorted and `reading_order` fields updated.
    """
    if config is None:
        config = ReadingOrderConfig()

    if not page.blocks:
        return page

    # Classify special regions
    special = classify_special_regions(page.blocks, page.width, page.height, config)

    # Sort main content with XY-cut
    main_blocks = special["main_content"]
    if main_blocks:
        ordered_main = xy_cut_sort(main_blocks, page.width, page.height, config)
        special["main_content"] = ordered_main

    # Reassemble in reading order: headers → main → footnotes → sidebars
    ordered_blocks: list[ExtractionBlock] = []
    ordered_blocks.extend(special["headers"])
    ordered_blocks.extend(special["main_content"])
    ordered_blocks.extend(special["footnotes"])
    ordered_blocks.extend(special["sidebars"])

    # Update reading order indices
    for i, b in enumerate(ordered_blocks):
        b.reading_order = i

    page.blocks = ordered_blocks
    return page
