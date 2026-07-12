"""Bootstrap access to the vendored upstream AgenticRetrieval clone.

The upstream source lives (git-ignored) in ``external/agenticretrieval`` and is
used AS-IS so it can be re-synced at any time (``scripts/sync_upstream.*``).

Import this module and call :func:`ensure_on_path` before importing any upstream
module (``dynamic_retriever``, ``cosmos_db_upload``, ``utils.cosmos_retriever``,
``greedy_log_det``, ``prompts``, ``timing_summary``) so they resolve to the
pristine vendored code.
"""
from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent / "external" / "agenticretrieval"


def ensure_on_path() -> Path:
    """Insert the vendored upstream clone at the front of ``sys.path``."""
    if not VENDOR_ROOT.exists():
        raise RuntimeError(
            f"Upstream clone not found at {VENDOR_ROOT}. "
            "Run scripts/sync_upstream.ps1 (or scripts/sync_upstream.sh) first."
        )
    p = str(VENDOR_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return VENDOR_ROOT
