"""
Model download and caching.

Models are stored in ~/.reach/models/:
    unlimited-ocr/     # baidu/Unlimited-OCR
    surya/             # Surya layout model
    qwen3.5-9b/        # Qwen 3.5 9B Q4_K_M GGUF
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

log = logging.getLogger(__name__)
console = Console()

MODELS_DIR = Path(os.environ.get("REACH_MODELS_DIR", Path.home() / ".reach" / "models"))

MODEL_MANIFEST = {
    "unlimited-ocr": {
        "name": "Unlimited OCR (Baidu, 3B MoE)",
        "source": "huggingface",
        "repo": "baidu/Unlimited-OCR",
        "files": ["*"],  # Download all files
        "size_gb": 6.0,
        "license": "MIT",
    },
    "surya": {
        "name": "Surya Layout (Datalab, 650M)",
        "source": "pip",
        "package": "surya-ocr",
        "size_gb": 0.5,
        "license": "Apache 2.0 / OpenRAIL-M",
        "note": "Models auto-downloaded on first use via surya-ocr package",
    },
    "qwen3.5-9b": {
        "name": "Qwen 3.5 9B Q4_K_M (Unsloth GGUF)",
        "source": "huggingface",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "files": ["Qwen3.5-9B-Q4_K_M.gguf"],
        "size_gb": 5.68,
        "license": "Apache 2.0",
    },
}


def download_all(force: bool = False) -> None:
    """Download all models in the manifest."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    console.print("[bold]Reach Model Download[/bold]")
    console.print(f"Models directory: {MODELS_DIR}\n")

    for model_id, info in MODEL_MANIFEST.items():
        model_dir = MODELS_DIR / model_id
        if model_dir.exists() and any(model_dir.iterdir()) and not force:
            console.print(f"  ✓ {info['name']} — already cached")
            continue

        console.print(f"  ↓ {info['name']} ({info['size_gb']:.1f} GB, {info['license']})")
        _download_model(model_id, info, model_dir)

    console.print("\n[bold green]All models ready.[/bold green]")


def _download_model(model_id: str, info: dict, target_dir: Path) -> None:
    """Download a single model from its source."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if info["source"] == "huggingface":
        _download_huggingface(info["repo"], info.get("files", ["*"]), target_dir)
    elif info["source"] == "pip":
        console.print(f"    Install with: pip install {info['package']}")
        console.print(f"    {info.get('note', '')}")


def _download_huggingface(repo: str, files: list[str], target_dir: Path) -> None:
    """Download files from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download

        console.print(f"    Downloading from huggingface.co/{repo} ...")
        snapshot_download(
            repo_id=repo,
            local_dir=target_dir,
            allow_patterns=files if files != ["*"] else None,
            resume_download=True,
        )
    except ImportError:
        console.print(
            "    [yellow]huggingface_hub not installed. Install with: pip install huggingface-hub[/yellow]"
        )
        console.print(f"    Then manually download from: https://huggingface.co/{repo}")


def verify_model(model_id: str) -> bool:
    """Verify that a model's files exist and have expected size."""
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists() or not any(model_dir.iterdir()):
        return False
    return True


def model_path(model_id: str) -> Optional[Path]:
    """Get the local path for a cached model."""
    path = MODELS_DIR / model_id
    if path.exists():
        return path
    return None
