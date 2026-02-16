#!/bin/bash

# proxy setting on D12 for huggingface browser access.
# export HTTP_PROXY="http://127.0.0.1:10809"
# export HTTPS_PROXY="http://127.0.0.1:10809"
# export NO_PROXY="localhost,127.0.0.1"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CMD=(python "$SCRIPT_DIR/src/agents/idea_agent/run.py")
CMD+=("$@")

"${CMD[@]}"