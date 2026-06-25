#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_PATH="$PROJECT_ROOT/.venv"
if [[ "${OSTYPE:-}" == darwin* ]]; then
  VENV_PYTHON="$VENV_PATH/bin/python"
  ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
else
  VENV_PYTHON="$VENV_PATH/Scripts/python.exe"
  ACTIVATE_SCRIPT="$VENV_PATH/Scripts/activate"
fi

if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
  echo "requirements.txt not found at $PROJECT_ROOT/requirements.txt" >&2
  return 1 2>/dev/null || exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating virtual environment at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
fi

echo "Installing dependencies from requirements.txt ..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Activating virtual environment in current shell ..."
  # shellcheck disable=SC1090
  source "$ACTIVATE_SCRIPT"
  echo "Virtual environment ready and activated."
else
  echo "Setup complete. To activate in your current shell, run:"
  echo "  source ./run.sh"
fi
