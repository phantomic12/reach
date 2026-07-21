"""Tests for VLM content enrichment."""

import pytest

from reach.enrich.enrich import (
    ALT_TEXT_SYSTEM_PROMPT,
    ALT_TEXT_USER_TEMPLATE,
    DECORATIVE_CLASSIFICATION_TEMPLATE,
    Enricher,
    MockEnrichmentModel,
    classify_decorative,
    classify_figure_type,
    filter_alt_text,
)
from reach.tagtree.tag_models import (
    AltTextResult,
    BBox,
    EnrichmentReport,
    ExtractionBlock,
    ExtractionDocument,
    ExtractionPage,
    TagNode,
    TagTreeConfig,
    TagType,
)


def make_figure_block(
    text: str = "chart",
    block_type: str = "Figure",
    page: int = 1,
) -> ExtractionBlock:
    return ExtractionBlock(
        type=block_type,
        bbox=BBox(x0=50, y0=10, x1=300, y1=200),
        text=text,
        page_number=page,
    )


class TestClassifyFigureType:
    def test_known_types(self):
        assert classify_figure_type("Figure") == "figure"
        assert classify_figure_type("Picture") == "picture"
        assert classify_figure_type("Table") == "table"
        assert classify_figure_type("Formula") == "mathematical formula"
        assert classify_figure_type("Chart") == "chart"
        assert classify_figure_type("Diagram") == "diagram"

    def test_unknown_type(self):
        assert classify_figure_type("UnknownType") == "figure"


class TestFilterAltText:
    config = TagTreeConfig()

    def test_valid_alt_text(self):
        text = "A bar chart showing quarterly sales data for 2023."
        filtered, rejected, reason = filter_alt_text(text, self.config)
        assert not rejected
        assert filtered == text
        assert reason == ""

    def test_empty_alt_text(self):
        filtered, rejected, reason = filter_alt_text("", self.config)
        assert rejected
        assert reason == "empty"

    def test_too_short(self):
        filtered, rejected, reason = filter_alt_text("short", self.config)
        assert rejected
        assert "too short" in reason

    def test_too_long(self):
        long_text = "X" * 600
        filtered, rejected, reason = filter_alt_text(long_text, self.config)
        assert rejected
        assert "too long" in reason

    def test_generic_phrase(self):
        filtered, rejected, reason = filter_alt_text("a chart", self.config)
        assert rejected
        assert "generic" in reason

    def test_generic_diagram(self):
        filtered, rejected, reason = filter_alt_text("a diagram", self.config)
        assert rejected
        assert "generic" in reason

    def test_hallucination_indicator(self):
        filtered, rejected, reason = filter_alt_text(
            "I'm not sure what this image shows but it might be a chart.",
            self.config,
        )
        assert rejected
        assert "hallucination" in reason

    def test_possibly_detected(self):
        filtered, rejected, reason = filter_alt_text(
            "This possibly shows a graph of data.",
            self.config,
        )
        assert rejected
        assert "hallucination" in reason

    def test_custom_config(self):
        """Custom config with different threshold values."""
        config = TagTreeConfig(min_alt_text_length=50)
        filtered, rejected, _ = filter_alt_text("A graph.", config)
        assert rejected


class TestClassifyDecorative:
    def test_explicit_decorative(self):
        assert classify_decorative("DECORATIVE") is True

    def test_decorative_prefix(self):
        assert classify_decorative("DECORATIVE border pattern") is True

    def test_logo(self):
        assert classify_decorative("company logo in header") is True

    def test_informational(self):
        assert classify_decorative("A bar chart showing 2023 revenue by quarter.") is False

    def test_separator(self):
        assert classify_decorative("decorative line separator") is True


class TestMockEnrichmentModel:
    def test_is_available(self):
        model = MockEnrichmentModel()
        assert model.is_available()

    def test_chart_prompt(self):
        model = MockEnrichmentModel()
        result = model.generate("Describe this chart...")
        assert "bar chart" in result.lower() or "chart" in result.lower()

    def test_diagram_prompt(self):
        model = MockEnrichmentModel()
        result = model.generate("Describe this diagram...")
        assert "flowchart" in result.lower() or "diagram" in result.lower()

    def test_photo_prompt(self):
        model = MockEnrichmentModel()
        result = model.generate("Describe this photo...")
        assert "photograph" in result.lower() or "picture" in result.lower()

    def test_decorative_prompt(self):
        model = MockEnrichmentModel()
        result = model.generate("Is this image decorative?")
        assert result == "DECORATIVE"

    def test_call_tracking(self):
        model = MockEnrichmentModel()
        model.generate("prompt one")
        model.generate("prompt two")
        assert model.call_count == 2
        assert model.last_prompt == "prompt two"

    def test_custom_responses(self):
        model = MockEnrichmentModel(
            responses={"custom_key": "Custom response text"}
        )
        result = model.generate("This contains custom_key somewhere")
        assert result == "Custom response text"


class TestEnricher:
    def test_collect_figures(self):
        doc = ExtractionDocument(
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        make_figure_block(block_type="Figure"),
                        ExtractionBlock(
                            type="Text",
                            bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                            text="Just text",
                        ),
                        make_figure_block(block_type="Chart"),
                        make_figure_block(block_type="Diagram"),
                    ],
                ),
            ],
        )
        model = MockEnrichmentModel()
        enricher = Enricher(model)
        figures = enricher._collect_figures(doc)
        assert len(figures) == 3

    def test_process_figure(self):
        block = make_figure_block(block_type="Chart")
        model = MockEnrichmentModel()
        enricher = Enricher(model)

        result = enricher._process_figure(block)
        assert isinstance(result, AltTextResult)
        assert not result.rejected
        assert len(result.alt_text) > 0
        assert result.model_used == "qwen3.5-9b"

    def test_enrichment_disabled(self):
        doc = ExtractionDocument(pages=[
            ExtractionPage(
                page_number=1,
                width=612,
                height=792,
                blocks=[make_figure_block()],
            )
        ])
        config = TagTreeConfig(enable_enrichment=False)
        model = MockEnrichmentModel()
        enricher = Enricher(model, config=config)
        report = enricher.enrich(doc)

        assert report.figures_processed == 0
        assert report.alt_texts_generated == 0

    def test_full_enrichment_pipeline(self):
        doc = ExtractionDocument(pages=[
            ExtractionPage(
                page_number=1,
                width=612,
                height=792,
                blocks=[
                    make_figure_block(block_type="Chart", page=1),
                    make_figure_block(block_type="Diagram", page=1),
                ],
            ),
        ])
        model = MockEnrichmentModel()
        enricher = Enricher(model)
        report = enricher.enrich(doc)

        assert report.figures_processed == 2
        assert report.alt_texts_generated == 2
        assert report.alt_texts_rejected == 0
        assert len(report.results) == 2

        for result in report.results:
            assert result.node_id.startswith("fig_")
            assert len(result.alt_text) > 0
            assert not result.rejected

    def test_apply_enrichment_to_tag_tree(self):
        root = TagNode(
            tag=TagType.DOCUMENT,
            children=[
                TagNode(
                    tag=TagType.FIGURE,
                    page_number=1,
                    bbox=BBox(x0=50, y0=10, x1=300, y1=200),
                ),
                TagNode(
                    tag=TagType.P,
                    text="Some text",
                ),
            ],
        )

        report = EnrichmentReport(
            results=[
                AltTextResult(
                    node_id="fig_1_50_10",
                    alt_text="A bar chart showing data.",
                    is_decorative=False,
                    confidence=1.0,
                ),
            ],
        )

        model = MockEnrichmentModel()
        enricher = Enricher(model)
        enricher.apply_enrichment(root, report)

        # Figure node should now have alt text
        assert root.children[0].alt_text == "A bar chart showing data."
        assert not root.children[0].is_decorative

        # Paragraph should be unchanged
        assert root.children[1].alt_text == ""


class TestPromptTemplates:
    def test_alt_text_user_template(self):
        prompt = ALT_TEXT_USER_TEMPLATE.format(figure_type="chart")
        assert "chart" in prompt
        assert "blind reader" in prompt
        assert "concise" in prompt

    def test_decorative_template(self):
        assert "YES" in DECORATIVE_CLASSIFICATION_TEMPLATE
        assert "NO" in DECORATIVE_CLASSIFICATION_TEMPLATE
