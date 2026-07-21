"""
Form Field Remediation — detect and tag AcroForm fields for PDF/UA-2.

PDF/UA-2 form requirements:
    - Every form field must be tagged with a Form (or sub-role) structure element.
    - Every field must have a tooltip (/TU entry) describing its purpose.
    - Tab order must match the logical reading order.
    - Radio buttons, checkboxes, and combo boxes need proper role mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pikepdf
from pikepdf import Array, Dictionary, Name, Pdf, String

log = logging.getLogger(__name__)

# Field type → PDF/UA-2 structure type mapping
FIELD_TYPE_ROLE_MAP: dict[str, str] = {
    "Btn": "Form",  # Push button or check box / radio button
    "Tx": "Form",  # Text field
    "Ch": "Form",  # Choice (combo box / list box)
    "Sig": "Form",  # Signature field
}

# Sub-type refinements
BTN_FLAG_ROLE_MAP: dict[str, str] = {
    "pushbutton": "Form",
    "checkbox": "Form",
    "radiobutton": "Form",
}


@dataclass
class FormField:
    """Descriptor for a form field to be remediated."""

    name: str  # Field name (T entry)
    field_type: str  # /FT: Btn, Tx, Ch, Sig
    tooltip: str = ""  # /TU entry
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    tab_order: int = 0
    role: str = "Form"
    attributes: dict[str, Any] = field(default_factory=dict)
    value: str = ""
    options: list[str] = field(default_factory=list)


class FormRemediator:
    """Detects, tags, and remediates AcroForm fields for PDF/UA-2 compliance."""

    def __init__(self) -> None:
        self._fields: list[FormField] = []

    def detect_fields(self, pdf: Pdf) -> list[FormField]:
        """Detect all AcroForm fields in a PDF.

        Args:
            pdf: The open pikepdf.Pdf document.

        Returns:
            List of FormField descriptors.
        """
        fields: list[FormField] = []
        catalog = pdf.Root

        if "/AcroForm" not in catalog:
            log.info("No AcroForm found in PDF")
            return fields

        acroform = catalog["/AcroForm"]
        field_array = acroform.get("/Fields", Array())

        for i, field_ref in enumerate(field_array):
            field = field_ref
            # Resolve indirect references
            if hasattr(field, "resolve"):
                field = field.resolve()

            if not isinstance(field, Dictionary):
                continue

            field_type = str(field.get("/FT", "")).strip("/")
            field_name = str(field.get("/T", f"field_{i}"))
            existing_tu = str(field.get("/TU", ""))

            # Determine page
            page_num = 0
            if "/P" in field:
                page_obj = field["/P"]
                if hasattr(page_obj, "resolve"):
                    page_obj = page_obj.resolve()
                # Find page number by iterating pages
                for idx, pg in enumerate(pdf.pages):
                    if pg.obj == page_obj:
                        page_num = idx
                        break

            # Determine bounding box
            bbox: tuple[float, float, float, float] | None = None
            if "/Rect" in field:
                rect = field["/Rect"]
                if isinstance(rect, Array) and len(rect) == 4:
                    bbox = (
                        float(rect[0]),
                        float(rect[1]),
                        float(rect[2]),
                        float(rect[3]),
                    )

            # Map role
            role = FIELD_TYPE_ROLE_MAP.get(field_type, "Form")

            # Collect options for choice fields
            options: list[str] = []
            if field_type == "Ch" and "/Opt" in field:
                opt = field["/Opt"]
                if isinstance(opt, Array):
                    for o in opt:
                        if isinstance(o, String):
                            options.append(str(o))
                        elif isinstance(o, Array) and len(o) >= 2:
                            options.append(str(o[1]))

            form_field = FormField(
                name=field_name,
                field_type=field_type,
                tooltip=existing_tu or self._generate_tooltip(field_type, field_name),
                page=page_num,
                bbox=bbox,
                tab_order=i,
                role=role,
                options=options,
                value=str(field.get("/V", "")),
            )
            fields.append(form_field)

        self._fields = fields
        log.info("Detected %d form fields", len(fields))
        return fields

    def remediate(self, pdf: Pdf, fields: list[FormField] | None = None) -> None:
        """Apply remediation to all form fields.

        Args:
            pdf: The open pikepdf.Pdf document.
            fields: Optional pre-detected field list. If None, auto-detects.
                     Can be a list of FormField or dict (converted automatically).
        """
        if fields is None:
            fields = self.detect_fields(pdf)
        else:
            self._fields = []
            for f in fields:
                if isinstance(f, FormField):
                    self._fields.append(f)
                elif isinstance(f, dict):
                    self._fields.append(FormField(
                        name=f.get("name", ""),
                        field_type=f.get("field_type", f.get("type", "")),
                        tooltip=f.get("tooltip", ""),
                        page=f.get("page", 0),
                        bbox=f.get("bbox"),
                        tab_order=f.get("tab_order", 0),
                        role=f.get("role", "Form"),
                    ))
                else:
                    log.warning("Skipping unrecognized form field: %s", f)

        for field in self._fields:
            self._remediate_field(pdf, field)

        # Set tab order in the document
        self._set_tab_order(pdf)

        log.info("Remediated %d form fields", len(self._fields))

    def _remediate_field(self, pdf: Pdf, field: FormField) -> None:
        """Remediate a single form field."""
        catalog = pdf.Root
        acroform = catalog.get("/AcroForm", Dictionary())
        field_array = acroform.get("/Fields", Array())

        # Find and update the field in the AcroForm
        for field_ref in field_array:
            f = field_ref
            if hasattr(f, "resolve"):
                f = f.resolve()
            if not isinstance(f, Dictionary):
                continue

            fname = str(f.get("/T", ""))
            if fname == field.name:
                # Add tooltip if missing
                if not f.get("/TU"):
                    f["/TU"] = String(field.tooltip)
                    log.debug("Added tooltip to field '%s': %s", field.name, field.tooltip)

                # Ensure the field type matches
                if field.role:
                    # Field role is set via the structure tree, not the field itself
                    pass

                break

    def _set_tab_order(self, pdf: Pdf) -> None:
        """Set the tab order on all pages to match reading order.

        PDF/UA-2 requires that tab order follows the logical document structure.
        Setting /Tabs to /S (structure order) on each page ensures this.
        """
        for page in pdf.pages:
            page["/Tabs"] = Name("/S")  # Structure order

    @staticmethod
    def _generate_tooltip(field_type: str, field_name: str) -> str:
        """Generate a default tooltip based on field type and name."""
        type_hints: dict[str, str] = {
            "Btn": "Button",
            "Tx": "Text field",
            "Ch": "Choice field",
            "Sig": "Signature field",
        }
        hint = type_hints.get(field_type, "Form field")
        return f"{hint}: {field_name}"

    def to_tag_nodes(self) -> list[dict[str, Any]]:
        """Convert form fields to tag tree nodes for the structure tree.

        Returns:
            List of dicts suitable for creating Form structure elements in the tag tree.
        """
        nodes: list[dict[str, Any]] = []
        for field in self._fields:
            nodes.append({
                "tag": "Form",
                "text": field.tooltip,
                "attributes": {
                    "field_name": field.name,
                    "field_type": field.field_type,
                    "role": field.role,
                },
                "page": field.page,
                "bbox": field.bbox,
                "tab_order": field.tab_order,
            })
        return nodes
