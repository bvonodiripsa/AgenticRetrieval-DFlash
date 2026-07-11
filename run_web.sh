#!/bin/bash
# Start the Food KG-RAG web UI on port 8080
# Prerequisites: an LLM backend reachable from the chosen config
#   - config_kg_dflash.yaml -> local vLLM (Qwen3.5-27B + DFlash) on port 8000
#   - config_kg_glm.yaml     ->  API key set in the config
#
# Usage:
#   ./run_web.sh                                  # default port 8080, config_kg_dflash.yaml
#   PORT=9090 ./run_web.sh                         # custom port
#   CONFIG_PATH=config_kg_glm.yaml ./run_web.sh    # serve GLM-5.2

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
CONFIG_PATH="${CONFIG_PATH:-config_kg_dflash.yaml}"

echo "Starting Food KG-RAG Web UI on port $PORT"
echo "Config: $CONFIG_PATH"
echo "LLM endpoint: $(grep -A1 'endpoint:' "$CONFIG_PATH" | head -2 | tail -1 | xargs)"
echo ""

exec python api.py \
    --config "$CONFIG_PATH" \
    --host 0.0.0.0 \
    --port "$PORT"
