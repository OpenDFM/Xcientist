#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CMD=(python "$SCRIPT_DIR/src/agents/idea_agent/run.py")
CMD+=("$@")

"${CMD[@]}"