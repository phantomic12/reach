# ADR-0006: Shared Python Core Architecture

**Status**: Accepted  
**Date**: 2026-07-21

## Context

Reach targets two deployment paths: a web-hosted server-side version and a Windows/Linux desktop application. The question is how much code they share versus implementing separately.

## Decision

**Shared Python core package (`reach-core`) with thin frontend wrappers.** The web version runs the core via FastAPI. The desktop version bundles the same core and runs it locally with a browser-based UI.

## Rationale

1. The remediation pipeline logic is the hard part — layout analysis, OCR coordination, tag-tree construction, PDF export, validation. This is pure Python and should exist once.
2. FastAPI provides both the web server backend and can run locally for the desktop app (localhost server + browser window).
3. Desktop bundling via PyInstaller or similar packages the Python core, llama.cpp binaries, and model files into a single installable application.
4. Testing is unified — the same test suite validates the pipeline regardless of deployment target.
5. GPU acceleration (CUDA/ROCm/Vulkan/Metal) is handled at the llama.cpp and PyTorch level, transparent to the Python core.
6. Alternative approaches rejected: separate implementations (code duplication), platform-native GUIs (unnecessary complexity for an academic tool).

## Consequences

- Single Python package: `reach-core`. Published to PyPI or installable from source.
- `reach-server`: FastAPI application importing reach-core. Docker-deployable.
- `reach-desktop`: PyInstaller bundle of reach-core + reach-server running on localhost, opened in browser.
- Auto-installer scripts handle CUDA/ROCm/Vulkan toolkit detection and llama.cpp backend compilation for desktop users.
- All three targets share the same test suite.
