"""
Reach CLI — entry point for all commands.
"""
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="reach",
    help="Automated PDF/UA-2 remediation pipeline",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print Reach version and model status."""
    from reach import __version__

    typer.echo(f"Reach v{__version__}")
    typer.echo(f"Python {sys.version}")
    typer.echo(f"Config: {Path.home() / '.reach' / 'config.yaml'}")
    typer.echo(f"Models: {Path.home() / '.reach' / 'models'}")


@app.command()
def download_models(force: bool = typer.Option(False, "--force", "-f", help="Re-download even if cached")) -> None:
    """Download all required models for offline use."""
    from reach.models.download import download_all

    download_all(force=force)


@app.command()
def extract(
    input_path: str = typer.Argument(..., help="Path to the source PDF"),
    output: str = typer.Option("output.json", "--output", "-o", help="Output JSON path"),
    dpi: int = typer.Option(200, "--dpi", help="Page rendering DPI"),
    no_models: bool = typer.Option(False, "--no-models", help="Skip loading ML models (layout/OCR will be empty)"),
) -> None:
    """Extract layout and text from a PDF. Produces structured JSON."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from reach.extract import extract_document

    typer.echo(f"Extracting: {input_path}")
    try:
        doc = extract_document(
            input_path,
            output_path=output,
            dpi=dpi,
            load_models=not no_models,
        )
        typer.echo(f"  Pages: {len(doc.pages)}")
        typer.echo(f"  Blocks: {len(doc.all_blocks)}")
        typer.echo(f"  Output: {output}")
    except FileNotFoundError:
        typer.echo(f"Error: file not found: {input_path}", err=True)
        raise typer.Exit(code=1)


@app.command()
def build_tags(
    input_path: str,
    output: str = typer.Option("tags.json", "--output", "-o"),
    enrichment: bool = typer.Option(True, "--enrichment/--no-enrichment", help="Enable/disable VLM enrichment"),
    language: str = typer.Option("", "--language", "-l", help="Force document language (BCP-47, auto-detect if empty)"),
) -> None:
    """Build PDF/UA-2 tag tree from extraction JSON."""
    import json
    import logging
    from pathlib import Path

    from reach.enrich.enrich import Enricher, LlamaCppEnrichmentModel, MockEnrichmentModel
    from reach.tagtree.builder import build_tag_tree_from_file
    from reach.tagtree.metadata import build_metadata
    from reach.tagtree.reading_order import compute_reading_order
    from reach.tagtree.tag_models import TagTreeConfig

    log = logging.getLogger(__name__)

    # --- Load extraction data ---
    input_file = Path(input_path)
    if not input_file.exists():
        typer.echo(f"Error: input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Building tag tree from: {input_path}")

    # --- Configuration ---
    config = TagTreeConfig(enable_enrichment=enrichment)
    if language:
        config.default_language = language

    # --- Step 1: Build tag tree ---
    tag_tree = build_tag_tree_from_file(input_path, config=config)
    typer.echo(f"  ✓ Tag tree built: {len(tag_tree.root.children)} top-level nodes")

    # --- Step 2: Reading order ---
    for page in tag_tree.document.pages:
        compute_reading_order(page)
    typer.echo(f"  ✓ Reading order computed for {len(tag_tree.document.pages)} pages")

    # --- Step 3: Metadata ---
    filename = tag_tree.document.filename or Path(input_path).stem
    tag_tree = build_metadata(tag_tree, filename=filename)
    typer.echo(f"  ✓ Title: {tag_tree.title}")
    typer.echo(f"  ✓ Language: {tag_tree.language}")

    # --- Step 4: Enrichment ---
    if enrichment:
        # Use real LlamaCpp model when GGUF is available, fall back to mock
        real_model = LlamaCppEnrichmentModel(config)
        if real_model.is_available():
            model = real_model
            model._load_model()
            typer.echo("  Using Qwen 3.5 9B (llama.cpp)")
        else:
            model = MockEnrichmentModel()
            typer.echo("  Qwen model not found — using mock enrichment")
        enricher = Enricher(model=model, config=config)
        report = enricher.enrich(tag_tree.document)
        enricher.apply_enrichment(tag_tree.root, report)

        typer.echo(f"  ✓ Enrichment: {report.figures_processed} figures processed")
        typer.echo(f"    - Alt texts generated: {report.alt_texts_generated}")
        if report.alt_texts_rejected:
            typer.echo(f"    - Alt texts rejected: {report.alt_texts_rejected}")
        if report.decorative_classified:
            typer.echo(f"    - Decorative: {report.decorative_classified}")

    # --- Write output ---
    output_path = Path(output)
    tag_tree.to_json(output_path)
    typer.echo(f"\n✓ Tag tree written to: {output_path}")


@app.command()
def remediate(
    input_path: str = typer.Argument(..., help="Path to the source PDF"),
    output: str = typer.Option("remediated.pdf", "--output", "-o", help="Output PDF path"),
    report: str = typer.Option("report.json", "--report", "-r", help="Report JSON path"),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip veraPDF validation"),
    no_auto_fix: bool = typer.Option(False, "--no-auto-fix", help="Skip auto-fix loop"),
    title: str = typer.Option("", "--title", help="Override document title"),
    language: str = typer.Option("", "--lang", help="Override document language (e.g., en-US)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Full pipeline: PDF → remediated PDF/UA-2.

    Runs extraction → tag tree construction → PDF construction → validation → report.
    """
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    from reach.construct.pipeline import RemediationPipeline

    pipeline = RemediationPipeline()

    metadata_overrides: dict[str, str] = {}
    if title:
        metadata_overrides["title"] = title
    if language:
        metadata_overrides["language"] = language

    typer.echo(f"=== Reach PDF/UA-2 Remediation ===")
    typer.echo(f"Input:  {input_path}")
    typer.echo(f"Output: {output}")
    typer.echo(f"Report: {report}")
    typer.echo("")

    try:
        result = pipeline.remediate(
            input_path=input_path,
            output=output,
            report_path=report,
            validate=not no_validate,
            auto_fix=not no_auto_fix,
            metadata=metadata_overrides if metadata_overrides else None,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo(f"Status:  {result.overall_status}")
    typer.echo(f"Time:    {result.processing_time_seconds:.1f}s")
    typer.echo(f"Fixes:   {result.total_fixes}")
    typer.echo(f"Confidence: {result.overall_confidence:.1%}")

    if result.verapdf_compliant:
        typer.echo("veraPDF: PASSED ✓")
    elif result.validation_failures > 0:
        typer.echo(f"veraPDF: {result.validation_failures} failure(s) ✗")
        for cat, count in result.validation_by_category.items():
            typer.echo(f"  {cat}: {count}")

    if result.review_queue:
        typer.echo(f"Human review needed: {len(result.review_queue)} item(s)")
        for item in result.review_queue[:5]:
            typer.echo(f"  [{item.priority}] {item.category}: {item.description[:80]}")

    typer.echo(f"\nDone. Output: {output}\nReport: {report}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8501) -> None:
    """Start the Reach web server."""
    import uvicorn

    typer.echo(f"Reach server starting on http://{host}:{port}")
    typer.echo(f"Health check: http://{host}:{port}/health")
    typer.echo(f"Web UI:      http://{host}:{port}/")
    uvicorn.run("reach-server.main:app", host=host, port=port, log_level="info")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
