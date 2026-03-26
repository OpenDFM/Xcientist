#!/bin/bash

# Survey Agent runner script
# Usage: ./run_survey.sh
# Topic is configured in src/config/default.yaml

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export MINERU_MODEL_SOURCE=modelscope

CONFIG="src/config/default.yaml"

# Resolve config path relative to project root
CONFIG_PATH="$SCRIPT_DIR/$CONFIG"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Running Survey Agent..."
echo "  Config: $CONFIG_PATH"

cd "$SCRIPT_DIR" && python src/agents/survey_agent/scripts/run_deep_survey.py
