# Reach — Install Guide

End-to-end install for Reach, the PDF/UA-2 remediation pipeline.

## 1. Python package

Reach requires Python 3.12+ and ships as `reach-core` on PyPI.

```bash
pip install reach-core
```

For OCR (Unlimited OCR + transformers backend):

```bash
pip install "reach-core[ocr]"
```

For all extras (server, desktop build, dev tools, OCR, surya):

```bash
pip install "reach-core[all]"
```

## 2. Download models

Reach uses three local models. They are cached at `~/.reach/models/`.

```bash
reach download-models
```

This fetches:

| Model                  | Size   | Purpose                          |
| ---------------------- | ------ | -------------------------------- |
| Unlimited OCR (3B MoE) | ~6 GB  | Text + tables + formulas         |
| Surya v2 (layout)      | ~700 MB | Layout detection + reading order |
| Qwen 3.5 9B Q4_K_M     | ~5.7 GB | Alt-text enrichment             |

Total disk: ~12 GB.

To pin a custom cache directory, set `REACH_MODELS_DIR=/path/to/cache` in your environment or `~/.reach/config.yaml`.

## 3. (Optional) veraPDF for PDF/UA-2 validation

Reach defaults to skipping veraPDF when the binary isn't found. To enable full
PDF/UA-2 validation of the output:

### macOS

```bash
brew install verapdf
```

### Linux (Debian / Ubuntu)

```bash
# Pre-built installer from https://software.verapdf.org/releases/
wget https://software.verapdf.org/rel/verapdf-installer.zip
unzip verapdf-installer.zip
cd verapdf-installer-* && ./verapdf-install
export PATH="$HOME/verapdf/verapdf-greenfield:$PATH"
```

### Linux (Arch)

```bash
yay -S verapdf
```

### Windows

Download the installer from <https://software.verapdf.org/releases/> and add
`verapdf.bat` to your PATH.

### Verify

```bash
verapdf --version
```

If the binary lives somewhere unusual, point Reach at it:

```bash
export VERAPDF_CLI=/path/to/verapdf
reach remediate input.pdf
```

Or in `~/.reach/config.yaml`:

```yaml
verapdf_cli: /path/to/verapdf
```

## 4. Try it

```bash
reach remediate path/to/input.pdf -o output.pdf -r report.json
```

The report JSON lists the remediation stages, confidence scores, and any
PDF/UA-2 validation failures if veraPDF is installed.

## 5. GPU acceleration (optional but recommended)

Reach auto-detects CUDA / ROCm / Vulkan / CPU. To force a specific backend:

```bash
export REACH_BACKEND=cuda    # or rocm, vulkan, cpu
```

For Qwen enrichment with llama.cpp, build llama-cpp-python against your GPU:

```bash
# CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python

# ROCm (Linux + AMD GPU)
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install --force-reinstall llama-cpp-python

# Vulkan (any vendor)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install --force-reinstall llama-cpp-python
```

CPU-only works without any of the above; enrichment just runs slower.

## Troubleshooting

- **"transformers not installed"** when running OCR: install with
  `pip install "reach-core[ocr]"`.
- **"veraPDF CLI not found"** when running with `--validate`: see section 3.
- **Models fail to download**: check that `~/.reach/models/` is writable, or
  set `REACH_MODELS_DIR` to a writable path.
- **GPU not detected**: install the matching CUDA / ROCm / Vulkan drivers and
  rebuild llama-cpp-python against them (section 5).
- **Out of memory**: set `REACH_NO_ML=1` to skip model loading and run the
  layout-only path, which uses pikepdf-only extraction (no GPU/Docker needed).
