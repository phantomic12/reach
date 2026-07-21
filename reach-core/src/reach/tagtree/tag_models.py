"""
Tag tree models for PDF/UA-2 structure.

Defines the Pydantic models used throughout Phase 2:
    - Extraction input models (Phase 1 output)
    - Tag tree output models (Phase 2 output → Phase 3 input)
    - PDF/UA-2 standard tag types and role mappings

Tag type reference (ISO 32000-2, §14.8.4):
    Document, Part, Sect, Div, H, H1-H6, P, L, LI, Lbl, LBody,
    Table, TR, TH, TD, THead, TBody, TFoot,
    Figure, Formula, Form, Artifact, Caption, Quote, Note,
    Reference, Link, Annot, BibEntry, BlockQuote, TOC, TOCI,
    Span, Code, Em, Strong, NonStruct, Private
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── PDF Tag Types ────────────────────────────────────────────────────────────


class TagType(str, enum.Enum):
    """Standard PDF/UA-2 structural element types."""

    # Document structure
    DOCUMENT = "Document"
    PART = "Part"
    SECT = "Sect"
    DIV = "Div"

    # Headings
    H = "H"  # Generic heading (auto-level)
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"
    H5 = "H5"
    H6 = "H6"

    # Text
    P = "P"

    # Lists
    L = "L"
    LI = "LI"
    LBL = "Lbl"
    LBODY = "LBody"

    # Tables
    TABLE = "Table"
    TR = "TR"
    TH = "TH"
    TD = "TD"
    THEAD = "THead"
    TBODY = "TBody"
    TFOOT = "TFoot"

    # Figures & Graphics
    FIGURE = "Figure"
    FORMULA = "Formula"
    FORM = "Form"

    # Special
    ARTIFACT = "Artifact"
    CAPTION = "Caption"
    QUOTE = "Quote"
    NOTE = "Note"
    REFERENCE = "Reference"
    LINK = "Link"
    ANNOT = "Annot"
    BIBENTRY = "BibEntry"
    BLOCKQUOTE = "BlockQuote"
    TOC = "TOC"
    TOCI = "TOCI"

    # Inline
    SPAN = "Span"
    CODE = "Code"
    EM = "Em"
    STRONG = "Strong"

    # Fallback
    NONSTRUCT = "NonStruct"


SURYA_TAG_MAP: dict[str, TagType] = {
    "SectionHeader": TagType.H,  # Level inferred later
    "Title": TagType.H1,
    "Text": TagType.P,
    "List": TagType.L,
    "ListItem": TagType.LI,
    "Table": TagType.TABLE,
    "Figure": TagType.FIGURE,
    "Formula": TagType.FORMULA,
    "Caption": TagType.CAPTION,
    "PageHeader": TagType.ARTIFACT,
    "PageFooter": TagType.ARTIFACT,
    "Footnote": TagType.NOTE,
    "PageNumber": TagType.ARTIFACT,
    "Picture": TagType.FIGURE,
    "TableOfContents": TagType.TOC,
    "Equation": TagType.FORMULA,
    "Code": TagType.CODE,
    "BlockQuote": TagType.BLOCKQUOTE,
    "Reference": TagType.REFERENCE,
}

HEADING_TYPES: set[TagType] = {TagType.H, TagType.H1, TagType.H2, TagType.H3, TagType.H4, TagType.H5, TagType.H6}

ARTIFACT_TYPES: set[TagType] = {TagType.ARTIFACT}


# ── Phase 1 Extraction Models (Input) ────────────────────────────────────────


class FontInfo(BaseModel):
    """Font information from extraction."""

    family: str = "Unknown"
    size: float = 12.0
    bold: bool = False
    italic: bool = False


class BBox(BaseModel):
    """Bounding box [x0, y0, x1, y1] in page coordinates (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height


class ExtractionBlock(BaseModel):
    """A single content block from Phase 1 extraction."""

    type: str  # Surya block type label
    bbox: BBox
    text: str = ""
    reading_order: int = 0
    font_info: Optional[FontInfo] = None
    page_number: int = 0
    confidence: float = 1.0

    # Table-specific fields (from Unlimited OCR)
    rows: Optional[list[list[str]]] = None  # Row-major cell text matrix
    header_rows: Optional[int] = None  # Number of header rows


class ExtractionPage(BaseModel):
    """A single page from Phase 1 extraction."""

    page_number: int
    width: float
    height: float
    blocks: list[ExtractionBlock] = []
    is_scanned: bool = False


class ExtractionDocument(BaseModel):
    """Full Phase 1 extraction output."""

    filename: str = ""
    pages: list[ExtractionPage] = []
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Phase 2 Tag Tree Models (Output) ─────────────────────────────────────────


class TagNode(BaseModel):
    """A single node in the PDF/UA-2 tag tree."""

    tag: TagType
    attributes: dict[str, str] = Field(default_factory=dict)
    text: str = ""
    children: list[TagNode] = Field(default_factory=list)
    page_number: int = 0
    bbox: Optional[BBox] = None

    # Enrichment fields
    alt_text: str = ""
    is_decorative: bool = False
    language_override: str = ""  # BCP-47 code for per-passage language

    # Table-specific
    scope: str = ""  # "Row", "Column", "Both" for TH
    colspan: int = 1
    rowspan: int = 1
    header_ids: list[str] = Field(default_factory=list)  # ID/Headers for merged cells

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for Phase 2→3 handoff."""
        return self.model_dump(exclude_none=True, mode="json")


class ReadingOrderItem(BaseModel):
    """One item in the document-level reading order."""

    node_id: str  # References a TagNode
    page_number: int
    sequence: int  # Position in reading order


class LanguageSpan(BaseModel):
    """A per-passage language override."""

    page_number: int
    bbox: BBox
    language: str  # BCP-47 code
    node_ids: list[str] = Field(default_factory=list)


class TagTree(BaseModel):
    """Complete PDF/UA-2 tag tree ready for PDF construction."""

    document: ExtractionDocument = Field(default_factory=ExtractionDocument)
    root: TagNode
    reading_order: list[ReadingOrderItem] = Field(default_factory=list)
    language: str = "en-US"  # Default document language
    language_spans: list[LanguageSpan] = Field(default_factory=list)
    title: str = ""
    display_title: bool = True  # PDF/UA-2: title must be displayed

    xmp_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Write tag tree to JSON file."""
        import json

        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)
        )

    @classmethod
    def from_json(cls, path: Path) -> "TagTree":
        """Read tag tree from JSON file."""
        import json

        return cls.model_validate(json.loads(path.read_text()))


# ── Table Structure Models ───────────────────────────────────────────────────


class TableCell(BaseModel):
    """A single table cell."""

    text: str
    row: int
    col: int
    is_header: bool = False
    scope: str = ""  # "Row", "Column", "Both"
    colspan: int = 1
    rowspan: int = 1
    col_span: int = 1  # Alias for colspan
    row_span: int = 1  # Alias for rowspan


class TableStructure(BaseModel):
    """Detected table structure with headers and cell grid."""

    rows: int
    cols: int
    header_rows: int
    cells: list[TableCell] = Field(default_factory=list)
    has_header_row: bool = True
    has_header_column: bool = False


# ── Enrichment Models ────────────────────────────────────────────────────────


class AltTextResult(BaseModel):
    """Result of VLM alt text generation for a single figure."""

    node_id: str
    alt_text: str
    is_decorative: bool
    confidence: float = 0.0
    rejected: bool = False  # Quality filter rejection
    rejection_reason: str = ""
    model_used: str = "qwen3.5-9b"


class EnrichmentReport(BaseModel):
    """Report from the enrichment stage."""

    figures_processed: int = 0
    alt_texts_generated: int = 0
    alt_texts_rejected: int = 0
    decorative_classified: int = 0
    results: list[AltTextResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ── Configuration ────────────────────────────────────────────────────────────


class TagTreeConfig(BaseModel):
    """Configuration for Phase 2 tag tree construction."""

    # Enrichment
    vlm_model_path: str = "~/.reach/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf"
    vlm_batch_size: int = 4
    vlm_max_tokens: int = 128
    vlm_temperature: float = 0.1
    enable_enrichment: bool = True

    # Alt text quality
    min_alt_text_length: int = 10  # Characters
    max_alt_text_length: int = 500  # Characters
    reject_generic_phrases: list[str] = Field(
        default_factory=lambda: ["a chart", "a diagram", "an image", "a figure", "a photo"]
    )

    # Reading order
    column_gap_threshold: float = 20.0  # Pixels gap for column detection
    enable_xy_cut: bool = True

    # Heading inference
    heading_font_size_threshold_ratio: float = 1.2  # h1 > h2 > body
    max_heading_depth: int = 6

    # Language detection
    default_language: str = "en-US"
