#!/usr/bin/env bash
# =============================================================================
# Sync the upstream AgenticRetrieval repo into external/agenticretrieval.
#
# This project uses the upstream source AS-IS (the folder is git-ignored), so it
# can be re-synced at any time without merge conflicts against local forks.
#
# Usage:
#   ./scripts/sync_upstream.sh              # clone or fast-forward pull
#   ./scripts/sync_upstream.sh v1.2.3       # check out a specific ref/tag
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/agenticretrieval"
URL="https://github.com/AzureCosmosDB/AgenticRetrieval.git"
REF="${1:-}"

if [ -d "$DEST/.git" ]; then
  echo "Updating upstream in $DEST ..."
  git -C "$DEST" fetch --all --prune
  if [ -n "$REF" ]; then git -C "$DEST" checkout "$REF"; else git -C "$DEST" pull --ff-only; fi
else
  echo "Cloning $URL into $DEST ..."
  mkdir -p "$(dirname "$DEST")"
  git clone "$URL" "$DEST"
  if [ -n "$REF" ]; then git -C "$DEST" checkout "$REF"; fi
fi

echo "Upstream synced at: $DEST"
