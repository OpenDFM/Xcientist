#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--prepare_only" ]]; then
        ARGS+=("--prepare-only")
    else
        ARGS+=("$arg")
    fi
done

if command -v uv >/dev/null 2>&1; then
    exec uv run xcientist-experiment "${ARGS[@]}"
fi

exec python -m src experiment "${ARGS[@]}"
