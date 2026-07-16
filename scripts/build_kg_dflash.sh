#!/bin/bash
# End-to-end pipeline for food-dflash:
#   1. Set up database & copy question-relevant docs from food → food-dflash
#   2. Build KG (question-driven, all 10 questions)
#   3. Run KG-RAG benchmark
#   4. Run validation test
#
# Prerequisites:
#   - az login (RBAC auth to Cosmos DB)
#   - vLLM server running on localhost:8000

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

CONFIG="my.yaml"

echo "================================================================"
echo "  Food-DFlash: Full Pipeline"
echo "================================================================"
echo ""

# ── Step 1: Setup database & copy documents ──────────────────────────
echo "── STEP 1: Setup database & copy documents ──"
python3 -u scripts/setup_food_dflash.py \
    --config "$CONFIG" \
    --questions data/food.json \
    --k-per-question 30

echo ""

# ── Step 2: Build Knowledge Graph ────────────────────────────────────
echo "── STEP 2: Build Knowledge Graph ──"
python3 -u kg_builder.py \
    --config "$CONFIG" \
    --question-driven \
    --question-index all \
    --question-k 30 \
    --extraction-rounds 1 \
    --concurrency 20

echo ""

# ── Step 3: Run KG-RAG benchmark ─────────────────────────────────────
echo "── STEP 3: KG-RAG Benchmark (10 questions) ──"
python3 -u kg_query.py \
    --config "$CONFIG" \
    --questions data/food.json

echo ""

# ── Step 4: Validation test ──────────────────────────────────────────
echo "── STEP 4: Validation test ──"
python3 -u test_food_dflash.py --config "$CONFIG"

echo ""
echo "================================================================"
echo "  Pipeline complete! Check out_kg_dflash/ for results."
echo "================================================================"
