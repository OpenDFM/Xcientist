#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
    exec uv run xcientist-pipeline "$@"
fi

exec python -m src pipeline "$@"
