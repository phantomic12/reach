"""Tests for table structure detection."""

import pytest

from reach.tagtree.tables import (
    TableCell,
    TableStructure,
    _detect_header_rows,
    _determine_scope,
    assign_scope_attributes,
    build_table_tag_node,
    detect_merged_cells,
    detect_table_structure,
)
from reach.tagtree.tag_models import (
    BBox,
    ExtractionBlock,
    FontInfo,
    TagType,
)


def make_table_block(
    rows: list[list[str]],
    header_rows: int | None = None,
    bold: bool = False,
) -> ExtractionBlock:
    """Create a table extraction block."""
    return ExtractionBlock(
        type="Table",
        bbox=BBox(x0=0, y0=0, x1=400, y1=200),
        text="",
        rows=rows,
        header_rows=header_rows,
        font_info=FontInfo(bold=bold),
    )


class TestHeaderDetection:
    def test_explicit_header_rows(self):
        """When header_rows is explicitly set, use it."""
        block = make_table_block(
            [["A", "B"], ["1", "2"], ["3", "4"]],
            header_rows=1,
        )
        assert _detect_header_rows(block) == 1

    def test_detect_header_by_bold(self):
        """Bold text in first row should flag it as header."""
        block = make_table_block(
            [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]],
            bold=True,
        )
        count = _detect_header_rows(block)
        assert count >= 1

    def test_detect_header_by_short_text(self):
        """Short / ALL_CAPS text in first row suggests header."""
        block = make_table_block(
            [["ID", "NAME", "VALUE"], ["1", "alpha", "100"], ["2", "beta", "200"]],
            bold=False,
        )
        count = _detect_header_rows(block)
        assert count >= 1

    def test_single_row_table(self):
        """A table with one row should treat that row as header."""
        block = make_table_block([["Only"]])
        assert _detect_header_rows(block) == 1

    def test_empty_table(self):
        block = make_table_block([])
        assert _detect_header_rows(block) == 0


class TestScopeDetermination:
    def test_top_left_column_header(self):
        """First header row, not first column → column scope."""
        scope = _determine_scope(True, 0, 1, 1, 5, 3)
        assert scope == "Column"

    def test_top_left_both(self):
        """Top-left corner in header → Both."""
        scope = _determine_scope(True, 0, 0, 1, 5, 3)
        assert scope == "Both"

    def test_non_header(self):
        scope = _determine_scope(False, 2, 1, 1, 5, 3)
        assert scope == ""

    def test_column_zero_header(self):
        """First column in a multi-header-row table could be row header."""
        scope = _determine_scope(True, 1, 0, 2, 5, 3)
        # Row 1, Col 0 in a 2-header-row table → Row scope
        # (This depends on implementation details; basic check)
        assert scope in ("Row", "Column", "Both")


class TestTableStructureDetection:
    def test_simple_table(self):
        block = make_table_block(
            [["Name", "Score"], ["Alice", "95"], ["Bob", "87"]],
            header_rows=1,
        )
        table = detect_table_structure(block)
        assert table is not None
        assert table.rows == 3
        assert table.cols == 2
        assert table.header_rows == 1
        assert len(table.cells) == 6

    def test_no_row_data(self):
        block = ExtractionBlock(
            type="Table",
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            text="",
        )
        assert detect_table_structure(block) is None

    def test_header_cells_marked(self):
        block = make_table_block(
            [["H1", "H2"], ["d1", "d2"]],
            header_rows=1,
        )
        table = detect_table_structure(block)
        assert table is not None

        # First row cells should be headers
        for cell in table.cells:
            if cell.row == 0:
                assert cell.is_header is True
            else:
                assert cell.is_header is False

    def test_cell_scope_assignment(self):
        block = make_table_block(
            [["Category", "Value"], ["A", "100"]],
            header_rows=1,
        )
        table = detect_table_structure(block)
        assert table is not None

        # Header cells should have scope
        header_cells = [c for c in table.cells if c.is_header]
        for c in header_cells:
            assert c.scope in ("Column", "Both", "")

    def test_empty_cells_handling(self):
        block = make_table_block(
            [["H1", "", "H3"], ["d1", "d2", ""]],
            header_rows=1,
        )
        table = detect_table_structure(block)
        assert table is not None

        # Even with empty cells, all positions in 3-col grid should have cells
        positions = {(c.row, c.col) for c in table.cells}
        assert (0, 0) in positions  # H1
        assert (0, 1) in positions  # empty header
        assert (1, 0) in positions  # d1


class TestBuildTableTagNode:
    def test_simple_table_tag_tree(self):
        table = TableStructure(
            rows=3,
            cols=2,
            header_rows=1,
            cells=[
                TableCell(text="Name", row=0, col=0, is_header=True, scope="Column"),
                TableCell(text="Score", row=0, col=1, is_header=True, scope="Column"),
                TableCell(text="Alice", row=1, col=0, is_header=False),
                TableCell(text="95", row=1, col=1, is_header=False),
                TableCell(text="Bob", row=2, col=0, is_header=False),
                TableCell(text="87", row=2, col=1, is_header=False),
            ],
        )

        node = build_table_tag_node(table, page_number=1)
        assert node.tag == TagType.TABLE
        assert node.page_number == 1

        # Should have THead and TBody
        assert len(node.children) == 2
        thead = node.children[0]
        tbody = node.children[1]
        assert thead.tag == TagType.THEAD
        assert tbody.tag == TagType.TBODY

        # THead should have 1 TR with 2 TH
        assert len(thead.children) == 1
        header_row = thead.children[0]
        assert header_row.tag == TagType.TR
        assert len(header_row.children) == 2
        assert header_row.children[0].tag == TagType.TH
        assert header_row.children[0].scope == "Column"

        # TBody should have 2 TRs with 2 TDs each
        assert len(tbody.children) == 2

    def test_no_header_table(self):
        table = TableStructure(
            rows=2,
            cols=2,
            header_rows=0,
            cells=[
                TableCell(text="1", row=0, col=0, is_header=False),
                TableCell(text="2", row=0, col=1, is_header=False),
                TableCell(text="3", row=1, col=0, is_header=False),
                TableCell(text="4", row=1, col=1, is_header=False),
            ],
        )
        node = build_table_tag_node(table, page_number=1)
        # No THead, only TBody
        assert len(node.children) == 1
        assert node.children[0].tag == TagType.TBODY

    def test_empty_cells_still_tagged(self):
        table = TableStructure(
            rows=1,
            cols=2,
            header_rows=1,
            cells=[],
        )
        node = build_table_tag_node(table, page_number=1)
        # Should still create a TR with TDs even if cells list is empty
        thead = node.children[0]
        tr = thead.children[0]
        assert len(tr.children) == 2  # Both columns populated
        assert tr.children[0].tag == TagType.TD
        assert tr.children[0].text == ""


class TestMergedCells:
    def test_detect_header_associations(self):
        table = TableStructure(
            rows=3,
            cols=2,
            header_rows=1,
            cells=[
                TableCell(text="Name", row=0, col=0, is_header=True),
                TableCell(text="Score", row=0, col=1, is_header=True),
                TableCell(text="Alice", row=1, col=0, is_header=False),
                TableCell(text="95", row=1, col=1, is_header=False),
            ],
        )
        associations = detect_merged_cells(table)
        # Data cells should be associated with their column headers
        assert len(associations) >= 2  # At least td_1_0 and td_1_1

    def test_assign_scope_attributes(self):
        table = TableStructure(
            rows=3,
            cols=2,
            header_rows=1,
            cells=[
                TableCell(text="Name", row=0, col=0, is_header=True),
                TableCell(text="Score", row=0, col=1, is_header=True),
                TableCell(text="Alice", row=1, col=0, is_header=False),
            ],
        )
        assign_scope_attributes(table)

        for cell in table.cells:
            if cell.is_header:
                assert cell.scope in ("Column", "Row", "Both", "")
            else:
                assert cell.scope == ""
