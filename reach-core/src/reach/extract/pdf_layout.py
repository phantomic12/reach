"""Lightweight layout detector using PDF text positions from pypdf."""
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def extract_text_blocks(pdf_path: str | Path) -> list[dict]:
    """Extract text blocks with positions from a PDF.

    Returns list of dicts with: text, x0, y0, x1, y1, font_size, label, page.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        all_blocks = []

        for page_num, page in enumerate(reader.pages):
            blocks = _extract_page_blocks(page, page_num)
            all_blocks.extend(blocks)

        log.debug("PDF text layout: %d blocks from %d pages", len(all_blocks), len(reader.pages))
        return [
            {
                "text": b["text"], "x0": b["x0"], "y0": b["y0"],
                "x1": b["x1"], "y1": b["y1"], "font_size": b["font_size"],
                "label": b["label"], "page": b["page"],
            }
            for b in all_blocks
        ]
    except Exception as e:
        log.debug("PDF text extraction failed: %s", e)
        return []


def _extract_page_blocks(page, page_num: int) -> list[dict]:
    """Extract positioned text blocks from a single page."""
    positions = []

    def visitor(text, cm, tm, font_dict, font_size):
        if text and text.strip():
            x, y = tm[4], tm[5]
            positions.append({
                "text": text.strip(),
                "x": x, "y": y,
                "font_size": font_size,
                "font_name": str(font_dict.get("/BaseFont", "")) if font_dict else "",
            })

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        return []

    # Merge nearby fragments and classify by font size
    return _merge_and_classify(positions, page_num)


def _merge_and_classify(positions: list[dict], page_num: int) -> list[dict]:
    """Merge adjacent text fragments and classify by font size."""
    if not positions:
        return []

    # Sort by y position (top to bottom), then x
    positions.sort(key=lambda p: (-p["y"], p["x"]))

    blocks = []
    current = None

    for p in positions:
        if current is None:
            current = dict(p)
            current["x0"] = p["x"]
            current["y0"] = p["y"]
            current["x1"] = p["x"] + p["font_size"] * len(p["text"]) * 0.5
            current["y1"] = p["y"] + p["font_size"]
            continue

        # Merge if same line (similar y) and close in x
        if (abs(p["y"] - current["y"]) < current["font_size"] * 0.5 and
                p["x"] - current["x1"] < current["font_size"] * 2):
            current["text"] += " " + p["text"]
            current["x1"] = max(current["x1"], p["x"] + p["font_size"] * len(p["text"]) * 0.5)
            current["font_size"] = max(current["font_size"], p["font_size"])
        else:
            # Classify and save
            current["label"] = _classify(current["font_size"])
            current["page"] = page_num
            blocks.append(current)
            current = dict(p)
            current["x0"] = p["x"]
            current["y0"] = p["y"]
            current["x1"] = p["x"] + p["font_size"] * len(p["text"]) * 0.5
            current["y1"] = p["y"] + p["font_size"]

    if current:
        current["label"] = _classify(current["font_size"])
        current["page"] = page_num
        blocks.append(current)

    return blocks


def _classify(font_size: float) -> str:
    """Classify block by font size."""
    if font_size >= 18:
        return "SectionHeader"
    elif font_size >= 14:
        return "Title"
    return "Text"
