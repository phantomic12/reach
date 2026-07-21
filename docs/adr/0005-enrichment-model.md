# ADR-0005: Content Enrichment with Qwen 3.5 9B

**Status**: Accepted  
**Date**: 2026-07-21

## Context

PDF/UA-2 requires alt text on every non-decorative image. Academic PDFs contain charts, diagrams, photos, and equations rendered as images. A vision-language model must examine each figure and produce a meaningful description. The model must fit within a 16GB RAM budget alongside the extraction models.

## Decision

**Qwen 3.5 9B at Q4_K_M quantization (~5.68 GB, ~7 GB VRAM at 8K context) via llama.cpp.**

## Rationale

1. Qwen 3.5 is natively multimodal — vision is baked into the base model through early fusion, not bolt-on adapters. No separate mmproj file needed.
2. Apache 2.0 license. No restrictions on use.
3. At Q4_K_M, peak VRAM with 8K context is ~7 GB. With sequential model loading, total peak on 16GB system RAM is ~12 GB — comfortable headroom.
4. Outperforms dedicated VLMs at comparable size: Qwen3-VL-4B has stronger pure-vision benchmarks, but Qwen 3.5 9B has 2x the capacity and stronger reasoning — critical for describing what a chart *means*, not just what it looks like.
5. 262K context window, 201-language support.
6. llama.cpp backend supports CUDA, ROCm, Vulkan, Metal — all target GPU vendors.
7. Alternatives considered: Qwen3-VL-4B (weaker reasoning), Bonsai 27B 1-bit (slower, lower vision quality at 1-bit precision), Gemma 3 4B (worse benchmarks, restrictive license).

## Consequences

- Model is ~5.68 GB on disk. Must be downloaded as part of Reach Desktop setup.
- Inference speed on CPU is moderate; GPU strongly recommended for production use.
- The same Qwen 3.5 9B instance can serve double duty for agent orchestration if needed.
- Qwen 3.5 auto-detects GPU backend via llama.cpp — no per-vendor code path needed.
