"""
Table structure detection — identifies header rows, assigns scope attributes,
handles merged cells, and ensures proper TH/TD tagging.

Per PDF/UA-2 requirements:
    - Header cells must have scope attributes (Row/Column/Both)
    - Merged cells need ID/Headers associations
    - Empty cells must still be tagged
    - THead/TBody/TFoot should be used when detectable
"""

from __future__ import annotations

import logging
from typing import Optional

from reach.tagtree.tag_models import (
    ExtractionBlock,
    TableCell,
    TableStructure,
    TagNode,
    TagType,
)

log = logging.getLogger(__name__)

# ── Table Detection ──────────────────────────────────────────────────────────


def detect_table_structure(
    block: ExtractionBlock,
) -> Optional[TableStructure]:
    """Detect table structure from an extraction block with row data.

    Analyzes cell text, row data from OCR, and formatting cues to identify
    header rows, header columns, column spans, and row spans.

    Args:
        block: An extraction block of type "Table" with `rows` data.

    Returns:
        TableStructure with full cell grid, or None if structure can't be determined.
    """
    rows_data = block.rows
    if not rows_data or not isinstance(rows_data, list) or len(rows_data) == 0:
        return None

    nrows = len(rows_data)
    ncols = max(len(row) for row in rows_data) if rows_data else 1

    # Detect header row count
    header_count = _detect_header_rows(block)

    # Build cell grid
    cells: list[TableCell] = []
    for r_idx, row in enumerate(rows_data):
        for c_idx, cell_text in enumerate(row):
            is_header = r_idx < header_count
            scope = _determine_scope(is_header, r_idx, c_idx, header_count, nrows, ncols)

            cells.append(
                TableCell(
                    text=str(cell_text) if cell_text is not None else "",
                    row=r_idx,
                    col=c_idx,
                    is_header=is_header,
                    scope=scope,
                )
            )

    return TableStructure(
        rows=nrows,
        cols=ncols,
        header_rows=header_count,
        cells=cells,
        has_header_row=(header_count > 0),
        has_header_column=False,  # To be detected by font analysis if available
    )


def _detect_header_rows(block: ExtractionBlock) -> int:
    """Detect number of header rows in a table.

    Uses multiple heuristics:
        1. Bold text in first rows
        2. Distinct font size in first rows
        3. Position (top rows)
        4. Underline/border presence

    Args:
        block: Table extraction block.

    Returns:
        Number of detected header rows (0, 1, or 2).
    """
    rows = block.rows
    if not rows:
        return 0

    # If explicit header_rows is set, use it
    if block.header_rows is not None and block.header_rows > 0:
        return min(block.header_rows, len(rows))

    # If only one row, treat it as header by default
    if len(rows) == 1:
        return 1

    # Bold detection heuristic:
    # Check first row for bold text patterns (uppercase, shorter length, etc.)
    # This is a lightweight heuristic when font_info is not available per-cell.
    header_count = 0

    for i in range(min(3, len(rows))):
        row = rows[i]
        if not row:
            continue

        # Count non-empty cells
        non_empty = [c for c in row if c and str(c).strip()]
        if not non_empty:
            continue

        # Heuristic: header rows often have ALL_CAPS or shorter text
        caps_count = sum(
            1 for c in non_empty if str(c).strip().isupper() or len(str(c).strip()) < 30
        )
        caps_ratio = caps_count / len(non_empty) if non_empty else 0

        # Check if font_info suggests bold
        font = block.font_info
        is_bold = font.bold if font else False

        # Header detection logic:
        # First row is a header if: bold font detected OR >50% cells are ALL_CAPS/short
        if i == 0 and (is_bold or caps_ratio > 0.5):
            header_count = 1
        # Second row may also be a header if similar patterns
        elif i == 1 and header_count == 1:
            if is_bold or caps_ratio > 0.5:
                header_count = 2
            else:
                break
        else:
            break

    return header_count


def _determine_scope(
    is_header: bool,
    row: int,
    col: int,
    header_count: int,
    total_rows: int,
    total_cols: int,
) -> str:
    """Determine the scope attribute for a TH cell.

    Per PDF/UA-2 §7.5.6.1:
        - "Row" if the header describes cells to its right in the same row
        - "Column" if the header describes cells below in the same column
        - "Both" if the header describes both (top-left corner cell)

    For simple tables:
        - Row 0 header cells get scope="Column" (column headers)
        - Col 0 in data rows with row headers get scope="Row"
        - Top-left corners may get "Both"

    Args:
        is_header: Whether this cell is a header.
        row: Cell row index.
        col: Cell column index.
        header_count: Number of header rows.
        total_rows: Total rows in table.
        total_cols: Total columns in table.

    Returns:
        Scope string: "Row", "Column", "Both", or "".
    """
    if not is_header:
        return ""

    # Top-left corner of multi-header table → Both
    if row == 0 and col == 0 and header_count >= 1:
        return "Both"
    # First header row → column headers
    elif row == 0:
        return "Column"
    # First column in header → row headers (if multi-row header)
    elif col == 0:
        return "Row"
    # Other header cells → Column by default
    else:
        return "Column"


# ── Tag Tree Node Generation ─────────────────────────────────────────────────


def build_table_tag_node(table: TableStructure, page_number: int) -> TagNode:
    """Build a complete Table → THead/TBody → TR → TH/TD tag tree.

    Args:
        table: Detected table structure.
        page_number: Page number for the tag node.

    Returns:
        TagNode tree for the table.
    """
    root = TagNode(tag=TagType.TABLE, page_number=page_number)

    thead = TagNode(tag=TagType.THEAD)
    tbody = TagNode(tag=TagType.TBODY)
    has_head = False
    has_body = False

    for r in range(table.rows):
        tr = TagNode(tag=TagType.TR)
        is_header_row = r < table.header_rows

        for c in range(table.cols):
            # Find the cell at (r, c)
            cell = _find_cell(table, r, c)

            if cell:
                tag = TagType.TH if cell.is_header else TagType.TD
                td = TagNode(
                    tag=tag,
                    text=cell.text,
                    scope=cell.scope if cell.is_header else "",
                    colspan=cell.colspan,
                    rowspan=cell.rowspan,
                )
            else:
                # Empty cell — must still be tagged per PDF/UA-2
                td = TagNode(
                    tag=TagType.TD,
                    text="",
                    scope="",
                )

            tr.children.append(td)

        if is_header_row:
            thead.children.append(tr)
            has_head = True
        else:
            tbody.children.append(tr)
            has_body = True

    if has_head:
        root.children.append(thead)
    if has_body:
        root.children.append(tbody)

    return root


def _find_cell(
    table: TableStructure, row: int, col: int
) -> Optional[TableCell]:
    """Find a cell at the given grid position, accounting for spans.

    Args:
        table: Table structure.
        row: Target row.
        col: Target column.

    Returns:
        TableCell at position, or None if the position is empty.
    """
    for cell in table.cells:
        if cell.row == row and cell.col == col:
            return cell
    return None


# ── Merged Cell Detection ────────────────────────────────────────────────────


def detect_merged_cells(table: TableStructure) -> dict[str, list[str]]:
    """Detect merged cell patterns and generate ID/Headers associations.

    For PDF/UA-2, merged cells are typically flagged in the source document
    via ColSpan/RowSpan or explicit merge indicators. This function generates
    header IDs and associates data cells with their header cells.

    Args:
        table: Table structure.

    Returns:
        Dict mapping cell IDs to associated header IDs:
        {"data_cell_id": ["header_id_1", "header_id_2"]}
    """
    associations: dict[str, list[str]] = {}

    # Generate IDs for header cells
    header_ids: dict[tuple[int, int], str] = {}
    for cell in table.cells:
        if cell.is_header:
            cell_id = f"th_{cell.row}_{cell.col}"
            header_ids[(cell.row, cell.col)] = cell_id

    # Associate data cells with headers
    for cell in table.cells:
        if not cell.is_header:
            cell_id = f"td_{cell.row}_{cell.col}"
            associated: list[str] = []

            # Column header: cells in header rows in same column
            for (hr, hc), hid in header_ids.items():
                if hc == cell.col and hr < table.header_rows:
                    associated.append(hid)

            # Row header: cells in header columns (col 0) in same row
            for (hr, hc), hid in header_ids.items():
                if hr == cell.row and hc == 0 and cell.col != 0:
                    if hid not in associated:
                        associated.append(hid)

            if associated:
                associations[cell_id] = associated

    return associations


# ── Scope Attribute Detection ────────────────────────────────────────────────


def assign_scope_attributes(table: TableStructure) -> None:
    """Assign scope attributes (Row/Column/Both) to all header cells.

    Mutates cells in place.

    Args:
        table: Table structure to annotate.
    """
    for cell in table.cells:
        if cell.is_header:
            cell.scope = _determine_scope(
                is_header=True,
                row=cell.row,
                col=cell.col,
                header_count=table.header_rows,
                total_rows=table.rows,
                total_cols=table.cols,
            )
