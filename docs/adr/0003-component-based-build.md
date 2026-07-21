# ADR-0003: Component-Based Build (Not Extending OpenDataLoader)

**Status**: Accepted  
**Date**: 2026-07-21

## Context

Two paths existed for building the core pipeline: fork/extend OpenDataLoader PDF (Apache 2.0, #1 in benchmarks, built-in auto-tagging), or build from individual components.

## Decision

**Build from individual components.** Use OpenDataLoader PDF as a reference for architecture and benchmarks, but do not fork or depend on it.

## Rationale

1. OpenDataLoader's core is 82.8% Java. The Python package is a thin subprocess wrapper. Modifying the pipeline internals requires working in Java — a language mismatch for a Python-first project.
2. The auto-tagging feature (Apache 2.0) produces Tagged PDFs but NOT PDF/UA-2 compliant ones. The PDF/UA-2 export layer is enterprise closed-source. We would still need to build PDF/UA-2 export from scratch.
3. The subprocess JVM-per-invocation architecture adds latency and JRE dependency — problematic for both web server (latency) and desktop app (install surface).
4. Vendor risk: OpenDataLoader is backed by Hancom (enterprise partner). The open-source boundary could tighten.
5. No visible form-field remediation support — a major gap for interactive academic PDFs.

## Consequences

- We rebuild layout analysis, OCR integration, tag-tree construction, and reading-order logic ourselves.
- Full control over the pipeline architecture and data model.
- No JVM dependency. Pure Python stack.
- OpenDataLoader remains a reference for benchmark comparisons and architecture patterns.
