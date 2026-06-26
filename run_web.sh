#!/bin/bash
# Start the Food KG-RAG web UI on port 8080
# Prerequisites: vLLM server running on port 8000 with Qwen3.5-27B + DFlash
#
# Usage:
#   ./run_web.sh                  # default port 8080
#   PORT=9090 ./run_web.sh        # custom port

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
CONFIG_PATH="${CONFIG_PATH:-config_kg_dflash.yaml}"

echo "Starting Food KG-RAG Web UI on port $PORT"
echo "Config: $CONFIG_PATH"
echo "LLM endpoint: $(grep -A1 'endpoint:' "$CONFIG_PATH" | head -2 | tail -1 | xargs)"
echo ""

export CONFIG_PATH
exec python -m uvicorn api:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
