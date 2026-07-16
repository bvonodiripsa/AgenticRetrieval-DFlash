#!/bin/bash
# Build the food Graph Index using question-driven mode (relevant docs only)
# This builds GI from documents relevant to the 10 benchmark questions first,
# for a quick demo. For full build, remove --question-driven flags.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "=== Food GI Build (question-driven, all 10 questions) ==="
echo "Using: Qwen2.5-32B via vLLM (localhost:8000) + in-process Qwen3-Embedding-0.6B"
echo ""

# Quick build: only docs relevant to the 10 benchmark questions
# ~300 docs, takes ~5-10 minutes
python -u gi_builder.py \
    --config my.yaml \
    --question-driven \
    --question-index all \
    --question-k 30 \
    --extraction-rounds 1 \
    --concurrency 20

echo ""
echo "=== Done! Check out_gi/ for results ==="
