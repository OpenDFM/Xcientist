#!/bin/bash
set -euo pipefail

export SHOW_LLM_REASONING=1
export EXPERIMENT_AGENT_MEMORY_TOOL_LOGS=0
export EXPERIMENT_AGENT_MEMORY_ENABLED=0
export EXPERIMENT_AGENT_MEMORY_WRITEBACK=0
export AGENT_BASH_TIMEOUT_SECONDS=600000
export EXPERIMENT_AGENT_PREWARM_ON_BOOT="${EXPERIMENT_AGENT_PREWARM_ON_BOOT:-0}"
export EXPERIMENT_AGENT_MCP_USE_WRAPPERS="${EXPERIMENT_AGENT_MCP_USE_WRAPPERS:-1}"
export EXPERIMENT_AGENT_MCP_WRAPPER_DIR="${EXPERIMENT_AGENT_MCP_WRAPPER_DIR:-$HOME/.cache/researchagent_mcp/bin}"
export EXPERIMENT_AGENT_INSTALL_MCP_WRAPPERS_ON_BOOT="${EXPERIMENT_AGENT_INSTALL_MCP_WRAPPERS_ON_BOOT:-0}"

DO_PREPARE=0
EXPERIMENT=""
IDEA_JSON=""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
CONFIG_PATH="$SCRIPT_DIR/src/config/default.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT="$2"; shift ;;
        --idea-json) IDEA_JSON="$2"; shift ;;
        --prepare) DO_PREPARE=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Error: --experiment is required"
    exit 1
fi

if [[ -z "$IDEA_JSON" ]]; then
    echo "Error: --idea-json is required"
    exit 1
fi

if [[ ! -f "$IDEA_JSON" ]]; then
    echo "Error: --idea-json file not found: $IDEA_JSON"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config file not found: $CONFIG_PATH"
    exit 1
fi

get_python_cmd() {
    local conda_python="$HOME/anaconda3/envs/openhands/bin/python"
    if [[ -f "$conda_python" ]]; then
        echo "$conda_python"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        echo "python"
    fi
}

PYTHON_CMD="$(get_python_cmd)"

WORKSPACE_ROOT="$("$PYTHON_CMD" -c 'from src.config import load_config; print(load_config().experiment.workspace.root)' 2>/dev/null)"
if [[ -z "$WORKSPACE_ROOT" ]]; then
    echo "Error: failed to resolve experiment.workspace.root from $CONFIG_PATH"
    exit 1
fi

EXPERIMENT_DIR="$WORKSPACE_ROOT/$EXPERIMENT"

prewarm_mcp() {
    if [[ "${EXPERIMENT_AGENT_PREWARM_ON_BOOT}" == "0" ]]; then
        echo "[mcp] prewarm skipped (EXPERIMENT_AGENT_PREWARM_ON_BOOT=0)"
        return 0
    fi
    if [[ "${EXPERIMENT_AGENT_MCP_USE_WRAPPERS}" != "0" ]]; then
        echo "[mcp] prewarm skipped (using local MCP wrappers)"
        return 0
    fi
    echo "[mcp] prewarm start"
    npx -y -p @modelcontextprotocol/server-github node -e "process.exit(0)" || true
    npx -y -p @modelcontextprotocol/server-filesystem node -e "process.exit(0)" || true
    npx -y -p @kazuph/mcp-fetch node -e "process.exit(0)" || true
    uvx --from minimax-coding-plan-mcp python -c "import sys; sys.exit(0)" || true
    echo "[mcp] prewarm done"
}

install_mcp_wrappers_if_needed() {
    if [[ "${EXPERIMENT_AGENT_INSTALL_MCP_WRAPPERS_ON_BOOT}" == "0" ]]; then
        return 0
    fi
    echo "[mcp] install wrappers start"
    bash "$SCRIPT_DIR/scripts/install_mcp_wrappers.sh"
    echo "[mcp] install wrappers done"
}

install_mcp_wrappers_if_needed
prewarm_mcp

mkdir -p "$EXPERIMENT_DIR"

TARGET_IDEA_JSON="$EXPERIMENT_DIR/idea.json"
TARGET_IDEA_RESULT_JSON="$EXPERIMENT_DIR/idea_result.json"
SOURCE_REALPATH="$(realpath "$IDEA_JSON")"
TARGET_REALPATH="$(realpath -m "$TARGET_IDEA_JSON")"

if [[ "$SOURCE_REALPATH" != "$TARGET_REALPATH" ]]; then
    cp "$IDEA_JSON" "$TARGET_IDEA_JSON"
    cp "$IDEA_JSON" "$TARGET_IDEA_RESULT_JSON"
    echo "[prepare] copied idea json to $TARGET_IDEA_JSON"
else
    cp "$IDEA_JSON" "$TARGET_IDEA_RESULT_JSON"
    echo "[prepare] idea json already in target location: $TARGET_IDEA_JSON"
fi

echo "[main] start: experiment=$EXPERIMENT workspace=$EXPERIMENT_DIR"
MAIN_CMD=(
    "$PYTHON_CMD" -m src.agents.experiment_agent.main
    --experiment "$EXPERIMENT"
    --verbose
)

if [[ "$DO_PREPARE" == "1" ]]; then
    MAIN_CMD+=(--force --clone-depth 1)
else
    MAIN_CMD+=(--skip-prepare)
fi

EXPERIMENT_AGENT_WORKSPACE_DIR="$EXPERIMENT_DIR" "${MAIN_CMD[@]}"
