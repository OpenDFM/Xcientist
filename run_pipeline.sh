#!/bin/bash
set -euo pipefail

# X-Scientist Pipeline Runner
# All settings are in src/config/default.yaml

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT=https://hf-mirror.com

# Use conda environment python if available
CONDA_PYTHON="$HOME/anaconda3/envs/openhands/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
    PYTHON_CMD="$CONDA_PYTHON"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

CONFIG_PATH="$SCRIPT_DIR/src/config/default.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "=========================================="
echo "X-Scientist Pipeline"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo ""

# Run pipeline
cd "$SCRIPT_DIR"
$PYTHON_CMD -m src.pipeline.run_loop --config "$CONFIG_PATH" "$@"
