"""Shared test fixtures – loads config before any test module imports."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import dynamic_retriever` works
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load config into dynamic_retriever.CONFIG before cosmos_retriever is imported
import dynamic_retriever
dynamic_retriever.load_config(Path(__file__).resolve().parent / "config.test.yaml.example")
