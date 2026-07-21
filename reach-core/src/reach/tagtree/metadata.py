"""
Metadata extraction and generation for PDF/UA-2.

Handles:
    - Document title extraction (from content, filename, or source metadata)
    - Language detection (document-level + per-passage overrides)
    - XMP metadata generation
    - DisplayTitle flag setting per PDF/UA-2 requirements
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from reach.tagtree.tag_models import (
    ExtractionDocument,
    LanguageSpan,
    TagNode,
    TagTree,
)

log = logging.getLogger(__name__)

# ── Language Detection ───────────────────────────────────────────────────────


# Common words for language identification (stopwords for detection)
LANGUAGE_MARKERS: dict[str, list[str]] = {
    "en": ["the", "of", "and", "a", "an", "in", "to", "is", "that", "for"],
    "fr": ["le", "la", "les", "des", "est", "une", "dans", "que", "pas", "pour", "de", "un", "sur", "avec"],
    "de": ["der", "die", "das", "und", "ist", "ein", "eine", "in", "von", "zu"],
    "es": ["el", "la", "los", "las", "de", "que", "y", "en", "un", "una"],
    "it": ["il", "la", "le", "di", "che", "un", "una", "in", "per", "con"],
    "pt": ["o", "a", "os", "as", "de", "que", "e", "um", "uma", "para"],
    "nl": ["de", "het", "een", "van", "is", "dat", "op", "in", "te", "en"],
    "ja": [],  # CJK handled differently
    "zh": [],
    "ko": [],
    "ar": [],
    "ru": ["и", "в", "не", "на", "я", "что", "это", "он", "с", "а"],
}

# Character ranges for non-Latin script detection
# (CJK and other scripts are handled by dedicated checks)
CJK_RANGES: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x3040, 0x309F),  # Hiragana (Japanese)
    (0x30A0, 0x30FF),  # Katakana (Japanese)
    (0xAC00, 0xD7AF),  # Hangul (Korean)
]


def detect_document_language(text: str) -> str:
    """Detect the document language from text content.

    Uses a statistical approach: counts occurrences of common words in each
    supported language and returns the language with the most matches.

    Args:
        text: Document text content (e.g., first page or full text).

    Returns:
        BCP-47 language code (e.g., "en-US", "fr-FR", "de-DE").
    """
    if not text or not text.strip():
        return "en-US"

    text_lower = text.lower()
    words = text_lower.split()

    # Check for Arabic script FIRST (before CJK check, since Arabic range
    # is in the same "non-Latin" detection block)
    arabic_count = sum(1 for ch in text if 0x0600 <= ord(ch) <= 0x06FF)
    if arabic_count > len(text) * 0.1:
        return "ar-SA"

    # Check for Cyrillic
    cyrillic_count = sum(1 for ch in text if 0x0400 <= ord(ch) <= 0x04FF)
    if cyrillic_count > len(text) * 0.1:
        return "ru-RU"

    # Check for CJK characters
    cjk_count = sum(
        1 for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES)
    )
    if cjk_count > len(text) * 0.1:
        # Determine specific CJK language
        hiragana = sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x309F)
        katakana = sum(1 for ch in text if 0x30A0 <= ord(ch) <= 0x30FF)
        hangul = sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7AF)

        if hiragana > 0 or katakana > 0:
            return "ja-JP"
        elif hangul > 0:
            return "ko-KR"
        else:
            return "zh-CN"

    # Word-frequency matching for Latin-script languages
    scores: dict[str, int] = {}
    for lang, markers in LANGUAGE_MARKERS.items():
        if not markers:
            continue
        score = sum(1 for m in markers if m in words)
        if score > 0:
            scores[lang] = score

    if not scores:
        return "en-US"

    best_lang = max(scores, key=lambda k: scores[k])

    # Map to BCP-47 regional variants
    lang_map: dict[str, str] = {
        "en": "en-US",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
        "it": "it-IT",
        "pt": "pt-BR",
        "nl": "nl-NL",
        "ru": "ru-RU",
    }

    return lang_map.get(best_lang, f"{best_lang}-{best_lang.upper()}")


def detect_per_passage_language(
    text: str,
    document_language: str,
) -> Optional[str]:
    """Detect if a passage is in a different language than the document.

    Args:
        text: Passage text.
        document_language: Detected document language.

    Returns:
        BCP-47 language code if different from document, else None.
    """
    if not text or len(text.strip()) < 20:
        return None

    passage_lang = detect_document_language(text)
    doc_lang_base = document_language.split("-")[0] if "-" in document_language else document_language
    passage_lang_base = passage_lang.split("-")[0] if "-" in passage_lang else passage_lang

    if passage_lang_base != doc_lang_base:
        return passage_lang

    return None


# ── Title Extraction ─────────────────────────────────────────────────────────


def extract_title(
    document: ExtractionDocument,
    filename: str = "",
) -> str:
    """Extract the document title using multiple strategies.

    Priority order:
        1. Source PDF metadata (if available)
        2. Largest heading on first page (font-size heuristic)
        3. First H1-like text block
        4. Filename (sanitized)

    Args:
        document: Phase 1 extraction document.
        filename: Original PDF filename (for fallback).

    Returns:
        Detected document title string.
    """
    # Strategy 1: Source PDF metadata
    if document.metadata:
        meta_title = document.metadata.get("title") or document.metadata.get("/Title")
        if meta_title and isinstance(meta_title, str) and len(meta_title.strip()) > 1:
            return meta_title.strip()

    # Strategy 2: Largest heading on first page
    if document.pages:
        first_page = document.pages[0]
        headers = [
            b for b in first_page.blocks
            if b.type in ("SectionHeader", "Title")
        ]
        if headers:
            # Pick the one with the largest font size
            best = max(
                headers,
                key=lambda b: (
                    b.font_info.size if b.font_info and b.font_info.size > 0 else 0
                ),
            )
            title = best.text.strip()
            if title:
                return title

    # Strategy 3: First non-empty text block on first page
    if document.pages:
        first_page = document.pages[0]
        for b in first_page.blocks:
            if b.text and b.text.strip() and len(b.text.strip()) > 3:
                return b.text.strip()[:200]

    # Strategy 4: Sanitized filename
    if filename:
        stem = Path(filename).stem
        # Clean up common patterns
        cleaned = re.sub(r"[_\-\s]+", " ", stem)
        cleaned = re.sub(r"\s*\(\d+\)$", "", cleaned)
        return cleaned.strip()

    return "Untitled Document"


# ── XMP Metadata Generation ──────────────────────────────────────────────────


def generate_xmp_metadata(
    tag_tree: TagTree,
    creator: str = "Reach PDF/UA-2 Remediation Pipeline",
) -> dict[str, Any]:
    """Generate XMP metadata dict for PDF/UA-2 export.

    PDF/UA-2 requires:
        - dc:title (with x-default language alternative)
        - dc:creator
        - dc:language
        - pdfuaid:part (2 for PDF/UA-2)
        - pdf:PDFVersion (2.0 for PDF 2.0)

    Args:
        tag_tree: Completed tag tree.
        creator: Tool identifier.

    Returns:
        Dict suitable for embedding as XMP metadata stream.
    """
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    metadata: dict[str, Any] = {
        "dc:title": {
            "x-default": tag_tree.title or "Untitled Document",
        },
        "dc:creator": [creator],
        "dc:language": tag_tree.language,
        "dc:format": "application/pdf",
        "xmp:CreateDate": now,
        "xmp:ModifyDate": now,
        "xmp:CreatorTool": creator,
        "pdf:PDFVersion": "2.0",
        "pdfuaid:part": 2,
        "pdf:Tagged": True,
        "pdf:DisplayDocTitle": tag_tree.display_title,
    }

    return metadata


def generate_xmp_string(metadata: dict[str, Any]) -> str:
    """Generate an XMP metadata packet string.

    Args:
        metadata: XMP metadata dict from generate_xmp_metadata().

    Returns:
        XMP packet as an XML string suitable for embedding in PDF.
    """
    xml_parts: list[str] = [
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '    <rdf:Description rdf:about=""',
        '      xmlns:dc="http://purl.org/dc/elements/1.1/"',
        '      xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
        '      xmlns:pdf="http://ns.adobe.com/pdf/1.3/"',
        '      xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/">',
    ]

    # Title
    if isinstance(metadata.get("dc:title"), dict):
        title = metadata["dc:title"].get("x-default", "")
        xml_parts.append(f"      <dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">{_escape_xml(title)}</rdf:li></rdf:Alt></dc:title>")
    else:
        title = str(metadata.get("dc:title", "Untitled Document"))
        xml_parts.append(f"      <dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">{_escape_xml(title)}</rdf:li></rdf:Alt></dc:title>")

    # Creator
    creators = metadata.get("dc:creator", [])
    if isinstance(creators, list):
        for c in creators:
            xml_parts.append(f"      <dc:creator><rdf:Seq><rdf:li>{_escape_xml(str(c))}</rdf:li></rdf:Seq></dc:creator>")
    else:
        xml_parts.append(f"      <dc:creator><rdf:Seq><rdf:li>{_escape_xml(str(creators))}</rdf:li></rdf:Seq></dc:creator>")

    # Language
    lang = metadata.get("dc:language", "en-US")
    xml_parts.append(f"      <dc:language>{_escape_xml(str(lang))}</dc:language>")

    # Format
    fmt = metadata.get("dc:format", "application/pdf")
    xml_parts.append(f"      <dc:format>{_escape_xml(str(fmt))}</dc:format>")

    # Dates and Tool
    xml_parts.append(f"      <xmp:CreateDate>{_escape_xml(str(metadata.get('xmp:CreateDate', '')))}</xmp:CreateDate>")
    xml_parts.append(f"      <xmp:ModifyDate>{_escape_xml(str(metadata.get('xmp:ModifyDate', '')))}</xmp:ModifyDate>")
    xml_parts.append(f"      <xmp:CreatorTool>{_escape_xml(str(metadata.get('xmp:CreatorTool', '')))}</xmp:CreatorTool>")

    # PDF-specific
    xml_parts.append(f"      <pdf:PDFVersion>{_escape_xml(str(metadata.get('pdf:PDFVersion', '2.0')))}</pdf:PDFVersion>")
    xml_parts.append(f"      <pdf:Tagged>true</pdf:Tagged>")
    xml_parts.append(f"      <pdf:DisplayDocTitle>{str(metadata.get('pdf:DisplayDocTitle', True)).lower()}</pdf:DisplayDocTitle>")

    # PDF/UA identifier
    xml_parts.append(f"      <pdfuaid:part>{metadata.get('pdfuaid:part', 2)}</pdfuaid:part>")

    xml_parts.extend([
        "    </rdf:Description>",
        "  </rdf:RDF>",
        "</x:xmpmeta>",
        '<?xpacket end="w"?>',
    ])

    return "\n".join(xml_parts)


def _escape_xml(text: str) -> str:
    """Escape special characters for XML content.

    Args:
        text: Raw text.

    Returns:
        XML-safe string.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ── Main Metadata Pipeline ───────────────────────────────────────────────────


def build_metadata(tag_tree: TagTree, filename: str = "") -> TagTree:
    """Populate tag_tree with metadata: title, language, XMP.

    Args:
        tag_tree: Partial tag tree from builder.
        filename: Original PDF filename.

    Returns:
        Updated TagTree with metadata populated.
    """
    doc = tag_tree.document

    # Extract title
    tag_tree.title = extract_title(doc, filename)

    # Detect document language from content
    all_text = _collect_document_text(doc)
    tag_tree.language = detect_document_language(all_text)

    # Detect per-passage language overrides
    tag_tree.language_spans = _detect_language_spans(doc, tag_tree.language)

    # Set display title flag
    tag_tree.display_title = True

    # Generate XMP metadata
    tag_tree.xmp_metadata = generate_xmp_metadata(tag_tree)

    return tag_tree


def _collect_document_text(document: ExtractionDocument) -> str:
    """Collect all text from a document for language detection.

    Args:
        document: Extraction document.

    Returns:
        Concatenated text from all blocks.
    """
    parts: list[str] = []
    for page in document.pages:
        for block in page.blocks:
            if block.text and block.text.strip():
                parts.append(block.text)
    return " ".join(parts)


def _detect_language_spans(
    document: ExtractionDocument,
    doc_language: str,
) -> list[LanguageSpan]:
    """Detect passages in a different language than the document.

    Args:
        document: Extraction document.
        doc_language: Detected document language.

    Returns:
        List of LanguageSpan objects for per-passage overrides.
    """
    spans: list[LanguageSpan] = []

    for page in document.pages:
        for block in page.blocks:
            if not block.text or len(block.text.strip()) < 20:
                continue

            override = detect_per_passage_language(block.text, doc_language)
            if override:
                spans.append(
                    LanguageSpan(
                        page_number=page.page_number,
                        bbox=block.bbox,
                        language=override,
                    )
                )

    return spans
