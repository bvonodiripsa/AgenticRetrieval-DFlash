#!/bin/bash
# Start the Food GI-RAG web UI on port 8080
# Prerequisites: an LLM backend reachable from the config (local vLLM or a
# hosted OpenAI-compatible gateway) + Azure CLI logged in for Cosmos DB RBAC.
#
# Usage:
#   ./run_web.sh                                # default port 8080, my.yaml
#   PORT=9090 ./run_web.sh                       # custom port
#   CONFIG_PATH=other.yaml ./run_web.sh          # custom config

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
CONFIG_PATH="${CONFIG_PATH:-my.yaml}"

echo "Starting Food GI-RAG Web UI on port $PORT"
echo "Config: $CONFIG_PATH"
echo "LLM endpoint: $(grep -A1 'endpoint:' "$CONFIG_PATH" | head -2 | tail -1 | xargs)"
echo ""

exec python api.py \
    --config "$CONFIG_PATH" \
    --host localhost \
    --port "$PORT"
