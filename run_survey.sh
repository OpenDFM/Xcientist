#!/bin/bash

# Survey Agent runner script
# Usage: ./run_survey.sh [--topic "Your Topic"] [--config config/deep_survey.yaml]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Default values
TOPIC=""
CONFIG="src/agents/survey_agent/config/deep_survey.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --topic)
            TOPIC="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--topic \"Your Topic\"] [--config config/deep_survey.yaml]"
            exit 1
            ;;
    esac
done

# Check if topic is provided
if [ -z "$TOPIC" ]; then
    echo "Error: --topic is required"
    echo "Usage: $0 [--topic \"Your Topic\"] [--config config/deep_survey.yaml]"
    exit 1
fi

# Resolve config path relative to project root
CONFIG_PATH="$SCRIPT_DIR/$CONFIG"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Running Survey Agent..."
echo "  Topic: $TOPIC"
echo "  Config: $CONFIG_PATH"

cd "$SCRIPT_DIR" && python src/agents/survey_agent/scripts/run_deep_survey.py topic="$TOPIC" --config-path "$SCRIPT_DIR/src/agents/survey_agent/config" --config-name "$(basename "$CONFIG")"
