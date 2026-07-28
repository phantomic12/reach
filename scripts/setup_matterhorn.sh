#!/usr/bin/env bash
# Set up the veraPDF Matterhorn corpus for the Reach benchmark.
#
# The corpus is ~246 MB / 434 PDFs across PDF_UA-1 and PDF_UA-2.
# It's git-ignored so this script handles the clone on demand.
#
# Usage:
#   scripts/setup_matterhorn.sh           # clone the full corpus
#   scripts/setup_matterhorn.sh --shallow # shallow clone (default)
#
# After cloning, run the benchmark with:
#   scripts/benchmark_matterhorn.py --flavour ua2 --limit 20 --markdown
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${PROJECT_ROOT}/tests/matterhorn"
REPO="https://github.com/veraPDF/veraPDF-corpus.git"

if [ -d "$TARGET/.git" ]; then
    echo "Matterhorn corpus already present at $TARGET"
    exit 0
fi

mkdir -p "$(dirname "$TARGET")"

if [ "${1:-}" = "--full" ]; then
    git clone "$REPO" "$TARGET"
else
    git clone --depth 1 "$REPO" "$TARGET"
fi

echo "Cloned corpus to $TARGET"
echo "PDF counts:"
find "$TARGET" -name "*.pdf" | awk -F/ '{print $(NF-1)}' | sort | uniq -c
