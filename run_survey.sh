#!/bin/bash

# Survey Agent runner script
# Usage: ./run_survey.sh
# Topic is configured in src/config/default.yaml

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Disable proxy for internal API calls (API server 58.210.177.113 cannot be reached via proxy)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="58.210.177.113,localhost,127.0.0.1"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export MINERU_MODEL_SOURCE=modelscope

CONFIG="src/config/default.yaml"
CONFIG_DIR="src/config"
CONFIG_NAME="default"

# Resolve config path relative to project root
CONFIG_PATH="$SCRIPT_DIR/$CONFIG"
CONFIG_DIR_PATH="$SCRIPT_DIR/$CONFIG_DIR"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Running Survey Agent..."
echo "  Config: $CONFIG_PATH"

cd "$SCRIPT_DIR" && python src/agents/survey_agent/scripts/run_deep_survey.py \
    --config-path "$CONFIG_DIR_PATH" \
    --config-name "$CONFIG_NAME"
