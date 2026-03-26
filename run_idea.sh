#!/bin/bash

# proxy setting on D12 for huggingface browser access.
# export HTTP_PROXY="http://127.0.0.1:10809"
# export HTTPS_PROXY="http://127.0.0.1:10809"
# export NO_PROXY="localhost,127.0.0.1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com
export IDEA_AGENT_CONFIG="$SCRIPT_DIR/src/config/default.yaml"

CMD=(python "$SCRIPT_DIR/src/agents/idea_agent/run.py")
CMD+=("$@")

"${CMD[@]}"