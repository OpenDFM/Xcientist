#!/bin/bash
# Run blog agent workflow for a specific experiment/project

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
EXPERIMENT=""
RESUME=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT="$2"; shift ;;
        --resume) RESUME=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$EXPERIMENT" ]; then
    echo "Error: --experiment is required"
    echo "Usage: $0 --experiment <project_name> [--resume]"
    exit 1
fi

echo "Running blog workflow for experiment: $EXPERIMENT"
if [ "$RESUME" = true ]; then
    echo "Resuming from last completed step..."
fi

cd "$SCRIPT_DIR"

# Build command
CMD="python -m src.agents.blog_agent.scripts.run --experiment $EXPERIMENT"
if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

eval $CMD