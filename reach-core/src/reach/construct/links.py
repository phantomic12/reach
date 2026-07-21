"""
Link Preservation — preserve and tag internal/external link annotations.

PDF/UA-2 link requirements:
    - Every link annotation must be tagged with a Link structure element.
    - Link annotations must include /Contents (accessible description).
    - Internal links (TOC, cross-references) and external hyperlinks must both be preserved.
    - Link descriptions should be meaningful (not just URLs).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, String

log = logging.getLogger(__name__)


@dataclass
class LinkAnnotation:
    """Descriptor for a link annotation to be preserved/tagged."""

    page: int
    bbox: tuple[float, float, float, float] | None = None
    uri: str = ""  # External URL
    dest_page: int | None = None  # Internal link destination page
    dest_name: str = ""  # Named destination
    description: str = ""  # /Contents entry
    link_type: str = "URI"  # "URI", "GoTo", "GoToR", "Launch"
    tag_role: str = "Link"


class LinkPreserver:
    """Preserves and tags link annotations for PDF/UA-2 compliance."""

    def __init__(self) -> None:
        self._links: list[LinkAnnotation] = []

    def detect_links(self, pdf: Pdf) -> list[LinkAnnotation]:
        """Detect all link annotations in a PDF.

        Args:
            pdf: The open pikepdf.Pdf document.

        Returns:
            List of LinkAnnotation descriptors.
        """
        links: list[LinkAnnotation] = []

        for page_num, page in enumerate(pdf.pages):
            annots = page.get("/Annots", Array())
            for annot_ref in annots:
                annot = annot_ref
                if hasattr(annot, "resolve"):
                    annot = annot.resolve()

                if not isinstance(annot, Dictionary):
                    continue

                # Only process Link annotations
                subtype = str(annot.get("/Subtype", ""))
                if subtype != "/Link":
                    continue

                # Extract bounding box
                bbox: tuple[float, float, float, float] | None = None
                if "/Rect" in annot:
                    rect = annot["/Rect"]
                    if isinstance(rect, Array) and len(rect) == 4:
                        bbox = (
                            float(rect[0]),
                            float(rect[1]),
                            float(rect[2]),
                            float(rect[3]),
                        )

                # Determine link type and destination
                uri = ""
                dest_page: int | None = None
                dest_name = ""
                link_type = "URI"

                if "/A" in annot:
                    action = annot["/A"]
                    if hasattr(action, "resolve"):
                        action = action.resolve()
                    action_type = str(action.get("/S", ""))

                    if action_type == "/URI":
                        link_type = "URI"
                        uri_val = action.get("/URI", "")
                        if isinstance(uri_val, String):
                            uri = str(uri_val)
                        elif isinstance(uri_val, bytes):
                            uri = uri_val.decode("utf-8", errors="replace")
                    elif action_type == "/GoTo":
                        link_type = "GoTo"
                        if "/D" in action:
                            dest = action["/D"]
                            if isinstance(dest, Array) and len(dest) > 0:
                                # First element could be a page reference
                                dest_ref = dest[0]
                                if hasattr(dest_ref, "objgen"):
                                    for idx, pg in enumerate(pdf.pages):
                                        if pg.obj == dest_ref or pg.objgen == dest_ref.objgen:
                                            dest_page = idx
                                            break
                    elif action_type == "/GoToR":
                        link_type = "GoToR"
                    elif action_type == "/Launch":
                        link_type = "Launch"
                elif "/Dest" in annot:
                    link_type = "GoTo"
                    dest = annot["/Dest"]
                    if isinstance(dest, Name):
                        dest_name = str(dest)
                    elif isinstance(dest, String):
                        dest_name = str(dest)
                    elif isinstance(dest, Array) and len(dest) > 0:
                        dest_ref = dest[0]
                        if hasattr(dest_ref, "objgen"):
                            for idx, pg in enumerate(pdf.pages):
                                if pg.obj == dest_ref or pg.objgen == dest_ref.objgen:
                                    dest_page = idx
                                    break

                # Check for existing /Contents (accessible description)
                description = ""
                if "/Contents" in annot:
                    contents = annot["/Contents"]
                    if isinstance(contents, String):
                        description = str(contents)

                # Generate description if missing
                if not description:
                    description = self._generate_description(
                        link_type=link_type,
                        uri=uri,
                        dest_page=dest_page,
                        dest_name=dest_name,
                        page_num=page_num,
                    )

                link = LinkAnnotation(
                    page=page_num,
                    bbox=bbox,
                    uri=uri,
                    dest_page=dest_page,
                    dest_name=dest_name,
                    description=description,
                    link_type=link_type,
                )
                links.append(link)

                # Update the annotation with accessible description
                annot["/Contents"] = String(description)

        self._links = links
        log.info("Detected %d link annotations", len(links))
        return links

    def preserve(self, pdf: Pdf, links: list[LinkAnnotation] | None = None) -> None:
        """Preserve and tag all link annotations.

        Args:
            pdf: The open pikepdf.Pdf document.
            links: Optional pre-detected link list. If None, auto-detects.
        """
        if links is None:
            links = self.detect_links(pdf)
        else:
            self._links = list(links)

        for link in self._links:
            self._tag_link(pdf, link)

        log.info("Preserved %d links with accessible descriptions", len(self._links))

    def _tag_link(self, pdf: Pdf, link: "LinkAnnotation") -> None:
        """Tag a link annotation for the structure tree."""
        page_idx = link.page if hasattr(link, "page") else link.get("page", 0)
        bbox = link.bbox if hasattr(link, "bbox") else link.get("bbox", None)
        uri = link.uri if hasattr(link, "uri") else link.get("uri", None)

        if page_idx < len(pdf.pages):
            page = pdf.pages[page_idx]
            annots = page.get("/Annots", Array())

            for annot_ref in annots:
                annot = annot_ref
                if hasattr(annot, "resolve"):
                    annot = annot.resolve()
                if not isinstance(annot, Dictionary):
                    continue

                if str(annot.get("/Subtype", "")) != "/Link":
                    continue

                # Match by rectangle or URI
                matches = False
                if bbox and "/Rect" in annot:
                    rect = annot["/Rect"]
                    if (
                        isinstance(rect, Array)
                        and len(rect) == 4
                        and abs(float(rect[0]) - bbox[0]) < 0.5
                        and abs(float(rect[1]) - bbox[1]) < 0.5
                    ):
                        matches = True

                if not matches and uri and "/A" in annot:
                    action = annot["/A"]
                    if hasattr(action, "resolve"):
                        action = action.resolve()
                    action_uri = str(action.get("/URI", ""))
                    if action_uri == uri:
                        matches = True

                if matches:
                    desc = link.description if hasattr(link, "description") else link.get("description", "Link")
                    annot["/Contents"] = String(desc)
                    # Set /StructParent for tagging
                    # The actual structure element ID is assigned when building the tag tree
                    break

    @staticmethod
    def _generate_description(
        link_type: str = "URI",
        uri: str = "",
        dest_page: int | None = None,
        dest_name: str = "",
        page_num: int = 0,
    ) -> str:
        """Generate a human-readable link description.

        Args:
            link_type: The link action type.
            uri: External URL.
            dest_page: Internal destination page number.
            dest_name: Named destination.
            page_num: Current page number (for internal links).

        Returns:
            A human-readable description of the link.
        """
        if link_type == "URI" and uri:
            parsed = urlparse(uri)
            domain = parsed.netloc or uri[:50]
            desc = f"External link to {domain}"
            if len(desc) > 100:
                desc = desc[:97] + "..."
            return desc

        if link_type == "GoTo":
            if dest_page is not None:
                return f"Link to page {dest_page + 1}"
            if dest_name:
                return f"Link to {dest_name}"
            return "Internal document link"

        if link_type == "GoToR":
            return "Link to external document"

        if link_type == "Launch":
            return "Link to launch external application"

        return "Link"

    def to_tag_nodes(self) -> list[dict[str, Any]]:
        """Convert links to tag tree nodes for the structure tree.

        Returns:
            List of dicts suitable for creating Link structure elements in the tag tree.
        """
        nodes: list[dict[str, Any]] = []
        for link in self._links:
            nodes.append({
                "tag": "Link",
                "text": link.description,
                "attributes": {
                    "link_type": link.link_type,
                    "uri": link.uri,
                    "dest_page": link.dest_page,
                    "dest_name": link.dest_name,
                },
                "page": link.page,
                "bbox": link.bbox,
            })
        return nodes
