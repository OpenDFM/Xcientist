#!/bin/bash

TOPIC=""

# Treat the first argument as a topic unless it looks like a flag.
if [ "$#" -ge 1 ] && [[ "$1" != "-"* ]]; then
    TOPIC="$1"
    shift
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ENV_FILE="$SCRIPT_DIR/src/agents/idea_agent/env/env.sh"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    # Load only exported variables to avoid executing commands in env.sh.
    # shellcheck disable=SC1090
    source <(grep -E '^export ' "$ENV_FILE")
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

HAS_RAG_CONFIG=0
HAS_PARALLELISM=0
for arg in "$@"; do
    if [ "$arg" = "--rag-config" ] || [[ "$arg" == --rag-config=* ]]; then
        HAS_RAG_CONFIG=1
    elif [ "$arg" = "--parallelism" ] || [[ "$arg" == --parallelism=* ]]; then
        HAS_PARALLELISM=1
    fi
done

CMD=(python "$SCRIPT_DIR/src/agents/idea_agent/scripts/run.py")
if [ -n "$TOPIC" ]; then
    CMD+=(--topics "$TOPIC")
fi
if [ "$HAS_RAG_CONFIG" -eq 0 ] && [ -n "$IDEA_AGENT_RAG_CONFIG" ]; then
    CMD+=(--rag-config "$IDEA_AGENT_RAG_CONFIG")
fi
if [ "$HAS_PARALLELISM" -eq 0 ] && [ -n "$IDEA_AGENT_PARALLELISM" ]; then
    CMD+=(--parallelism "$IDEA_AGENT_PARALLELISM")
fi
CMD+=("$@")

"${CMD[@]}"
