"""
Pydantic models for the extraction pipeline output.

Defines the structured representation of an extracted PDF document:
pages, blocks, text, tables, images, and metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Block Types — mapped from Surya's 18 layout categories
# ---------------------------------------------------------------------------


class BlockType(str, Enum):
    """Semantic block types for Reach's extraction output.

    These are mapped from Surya's 18 layout categories to a smaller,
    PDF/UA-2-relevant set. Types that Reach doesn't need to distinguish
    (e.g. Stamp, Signature) are collapsed into Artifact.
    """

    # Structural
    SECTION_HEADER = "SectionHeader"  # H1-H6 headings
    TEXT = "Text"  # Body paragraphs, footnotes, references
    LIST = "List"  # Bulleted or numbered lists
    PAGE_HEADER = "PageHeader"  # Running page header
    PAGE_FOOTER = "PageFooter"  # Running page footer

    # Content
    TABLE = "Table"  # Tabular data
    FIGURE = "Figure"  # Images, pictures, charts
    EQUATION = "Equation"  # Display or inline math

    # Special
    ARTIFACT = "Artifact"  # Stamps, signatures, decorative elements


# ---- Surya-to-Reach mapping ------------------------------------------------

SURYA_TO_REACH: dict[str, BlockType] = {
    "Section-header": BlockType.SECTION_HEADER,
    "Title": BlockType.SECTION_HEADER,
    "Plain text": BlockType.TEXT,
    "Text": BlockType.TEXT,
    "List-item": BlockType.LIST,
    "Table": BlockType.TABLE,
    "Picture": BlockType.FIGURE,
    "Figure": BlockType.FIGURE,
    "Formula": BlockType.EQUATION,
    "Page-header": BlockType.PAGE_HEADER,
    "Page-footer": BlockType.PAGE_FOOTER,
    "Footnote": BlockType.TEXT,
    "Caption": BlockType.TEXT,
    "Handwriting": BlockType.TEXT,
    "Text-inline-math": BlockType.TEXT,
    "Reference": BlockType.TEXT,
    "Table-of-contents": BlockType.TEXT,
    "Stamp": BlockType.ARTIFACT,
    "Signature": BlockType.ARTIFACT,
}


def map_surya_block_type(raw: str) -> BlockType:
    """Map a raw Surya block label to a Reach BlockType.

    Args:
        raw: Surya's block label (e.g. "Section-header", "Table").

    Returns:
        The corresponding Reach BlockType. Falls back to TEXT.
    """
    return SURYA_TO_REACH.get(raw, BlockType.TEXT)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned bounding box — [x1, y1, x2, y2] in page coordinates (origin top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x2_must_be_ge_x1(cls, v: float, info: Any) -> float:
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError("x2 must be >= x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_must_be_ge_y1(cls, v: float, info: Any) -> float:
        if "y1" in info.data and v < info.data["y1"]:
            raise ValueError("y2 must be >= y1")
        return v

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_list(cls, bbox: list[float]) -> BBox:
        """Create from a [x1, y1, x2, y2] list."""
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    def to_list(self) -> list[float]:
        """Convert back to a plain list."""
        return [self.x1, self.y1, self.x2, self.y2]


class FontInfo(BaseModel):
    """Font metadata for a text run."""

    name: str = "Unknown"
    size: float = 0.0
    bold: bool = False
    italic: bool = False
    color_hex: str = "#000000"


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class BaseBlock(BaseModel):
    """Shared fields for all block types."""

    block_type: BlockType
    bbox: BBox
    reading_order: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TextBlock(BaseBlock):
    """A block containing textual content."""

    text: str = ""
    font_info: Optional[FontInfo] = None


class TableCell(BaseModel):
    """A single cell in a table."""

    text: str = ""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    bbox: BBox
    is_header: bool = False


class TableBlock(BaseBlock):
    """A table with full row/column structure."""

    block_type: BlockType = BlockType.TABLE
    rows: int = 0
    columns: int = 0
    cells: list[TableCell] = Field(default_factory=list)
    caption: Optional[str] = None


class ImageBlock(BaseBlock):
    """An image or figure with optional extracted text."""

    block_type: BlockType = BlockType.FIGURE
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    image_bytes: Optional[bytes] = Field(default=None, exclude=True)
    image_mime: str = "image/png"


class EquationBlock(BaseBlock):
    """A mathematical equation, stored as LaTeX."""

    block_type: BlockType = BlockType.EQUATION
    latex: str = ""


# ---------------------------------------------------------------------------
# Page & Document
# ---------------------------------------------------------------------------


class PageMetadata(BaseModel):
    """Metadata about a single page."""

    page_number: int
    width_pt: float
    height_pt: float
    rotation: int = 0
    is_scanned: bool = False
    text_coverage: float = 0.0  # fraction of page area covered by text


class Page(BaseModel):
    """A single page of extracted content."""

    metadata: PageMetadata
    blocks: list[TextBlock | TableBlock | ImageBlock | EquationBlock] = Field(
        default_factory=list
    )

    @property
    def text_blocks(self) -> list[TextBlock]:
        return [b for b in self.blocks if isinstance(b, TextBlock)]

    @property
    def table_blocks(self) -> list[TableBlock]:
        return [b for b in self.blocks if isinstance(b, TableBlock)]

    @property
    def figure_blocks(self) -> list[ImageBlock]:
        return [b for b in self.blocks if isinstance(b, ImageBlock)]

    @property
    def equation_blocks(self) -> list[EquationBlock]:
        return [b for b in self.blocks if isinstance(b, EquationBlock)]


class DocumentMetadata(BaseModel):
    """Document-level metadata extracted from the PDF."""

    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: str = ""
    language: str = "en"
    page_count: int = 0
    pdf_version: str = ""


class ExtractedDocument(BaseModel):
    """Top-level extraction output: the complete structured representation of a PDF."""

    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    pages: list[Page] = Field(default_factory=list)

    @property
    def all_blocks(self) -> list:
        """Flat list of every block across all pages."""
        result: list = []
        for page in self.pages:
            result.extend(page.blocks)
        return result
