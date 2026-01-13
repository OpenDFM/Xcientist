#!/bin/bash

TOPIC=""

# Check if arguments are provided
if [ "$#" -ge 1 ]; then
    TOPIC="$1"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ENV_FILE="$SCRIPT_DIR/src/agents/idea_agent/env/env.sh"
if [ -f "$ENV_FILE" ]; then
    echo "Sourcing environment variables from $ENV_FILE"
    source "$ENV_FILE"
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

python "$SCRIPT_DIR/src/agents/idea_agent/scripts/run.py" --topics "$TOPIC"
