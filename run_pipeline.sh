#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

SYNC_PYTHON="python"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    SYNC_PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi
"$SYNC_PYTHON" "$SCRIPT_DIR/scripts/sync_claude_anthropic_env.py" \
    --env-file "$SCRIPT_DIR/.env" \
    --settings "${CLAUDE_CODE_GLOBAL_SETTINGS:-$HOME/.claude/settings.json}"

if command -v xcientist >/dev/null 2>&1; then
    exec xcientist pipeline "$@"
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/xcientist" ]]; then
    exec "$SCRIPT_DIR/.venv/bin/xcientist" pipeline "$@"
fi

exec python -m src pipeline "$@"
