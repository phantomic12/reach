"""
Reach: Automated PDF/UA-2 remediation pipeline.

Usage:
    reach download-models          # Download all required models
    reach extract input.pdf        # Extract layout + text → JSON
    reach build-tags output.json   # Build tag tree from extraction
    reach remediate input.pdf      # Full pipeline: PDF → remediated PDF
    reach serve                    # Start web server
    reach version                  # Print version info
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("reach-core")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
