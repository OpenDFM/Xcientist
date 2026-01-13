#!/bin/bash
export SHOW_LLM_REASONING=1
export EXPERIMENT_AGENT_MEMORY_TOOL_LOGS=0
export EXPERIMENT_AGENT_MEMORY_ENABLED=0
export EXPERIMENT_AGENT_MEMORY_WRITEBACK=0
export AGENT_BASH_TIMEOUT_SECONDS=600000

DO_PREPARE=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$SCRIPT_DIR/src/agents/experiment_agent/workspaces"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT="$2"; shift ;;
        --idea-json) IDEA_JSON="$2"; shift ;;
        --prepare) DO_PREPARE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$EXPERIMENT" ]; then
    echo "Error: --experiment is required"
    exit 1
fi

if [ -z "$IDEA_JSON" ]; then
    echo "Error: --idea-json is required"
    exit 1
fi

if [ ! -f "$IDEA_JSON" ]; then
    echo "Error: --idea-json file not found: $IDEA_JSON"
    exit 1
fi

EXPERIMENT_DIR="$WORKSPACE_DIR/$EXPERIMENT"

if [ "$DO_PREPARE" = "1" ]; then
    mkdir -p "$EXPERIMENT_DIR"
    cp "$IDEA_JSON" "$EXPERIMENT_DIR/idea.json"
    echo "Copied $IDEA_JSON to $EXPERIMENT_DIR/idea.json"
    python -m src.agents.experiment_agent.prepare --experiment "$EXPERIMENT" --force --clone-depth 1 --verbose
fi
python -m src.agents.experiment_agent.main --experiment "$EXPERIMENT" --verbose --resume