#!/bin/bash

# ============================================================
# Deep Survey Batch Runner with Custom Parameters
# 
# Usage:
#   ./run_deep_survey_custom.sh [OPTIONS]
#
# Options:
#   --topic "Your Topic"           Topic to search for (required)
#   --output-dir PATH              Output directory (default: ./outputs/custom-run)
#   --config-yaml PATH              Base YAML config file (default: config/deep_survey_batch_others_huoshan.yaml)
#   --api-key KEY                   API key (overrides yaml)
#   --api-base-url URL              API base URL (overrides yaml)
#   --model MODEL                   LLM model name (overrides yaml)
#   --debug [true|false]            Debug mode (default: true)
#   --max-seed-paper NUM            Max seed papers (default: 15)
#   --graph-depth DEPTH             Reference graph depth (default: 1)
#   --db-path PATH                  Paper graph database path
#
# If an option is not specified, the value from --config-yaml will be used.
# ============================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
TOPIC=""
OUTPUT_DIR="${PROJECT_ROOT}/outputs/custom-run"
CONFIG_YAML="config/deep_survey_batch_others_huoshan.yaml"
API_KEY=""
API_BASE_URL=""
MODEL_NAME=""
DEBUG="true"
MAX_SEED_PAPER="15"
GRAPH_DEPTH="1"
DB_PATH=""
EVAL_SAVE_PATH=""
SAVE_PATH=""
SAVE_JSON_PATH=""

# Topic path (can be overridden)
TOPIC_PATH=""

# Show help
show_help() {
    cat << EOF
Deep Survey Custom Batch Runner

Usage:
    $0 [OPTIONS]

Options:
    --topic TOPIC              Topic to search for (REQUIRED)
    --output-dir DIR           Output directory (default: ./outputs/custom-run)
    --config-yaml PATH          Base YAML config file (default: config/deep_survey_batch_others_huoshan.yaml)
    
    # API Configuration (overrides yaml if provided)
    --api-key KEY               API key
    --api-base-url URL          API base URL
    --model MODEL               LLM model name
    
    # Processing Configuration (overrides yaml if provided)
    --debug [true|false]        Debug mode (default: true)
    --max-seed-paper NUM        Max seed papers (default: 15)
    --graph-depth DEPTH         Reference graph depth (default: 1)
    --db-path PATH              Paper graph database path
    --topic-path PATH           Topic file path (overrides yaml)
    
    # Output Configuration
    --eval-save-path PATH       Evaluation save path
    --save-path-prefix PREFIX  Prefix for save_path (topic will be appended)
    --save-json-path-prefix PREFIX  Prefix for save_json_path (topic will be appended)
    
    --help                     Show this help message

Examples:
    # Run with a specific topic
    $0 --topic "Large Language Models"
    
    # Run with custom API and output
    $0 --topic "Vision Transformers" --api-key "your-key" --output-dir ./my-output
    
    # Run with custom processing config
    $0 --topic "Graph Neural Networks" --max-seed-paper 20 --graph-depth 2
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --topic)
            TOPIC="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config-yaml)
            CONFIG_YAML="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --api-base-url)
            API_BASE_URL="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --debug)
            DEBUG="$2"
            shift 2
            ;;
        --max-seed-paper)
            MAX_SEED_PAPER="$2"
            shift 2
            ;;
        --graph-depth)
            GRAPH_DEPTH="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --topic-path)
            TOPIC_PATH="$2"
            shift 2
            ;;
        --eval-save-path)
            EVAL_SAVE_PATH="$2"
            shift 2
            ;;
        --save-path-prefix)
            SAVE_PATH="$2"
            shift 2
            ;;
        --save-json-path-prefix)
            SAVE_JSON_PATH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$TOPIC" ]; then
    echo "Error: --topic is required"
    show_help
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Set default evaluation save path if not provided
if [ -z "$EVAL_SAVE_PATH" ]; then
    EVAL_SAVE_PATH="${OUTPUT_DIR}/evaluation.txt"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Hydra overrides
OVERRIDES=""

# BasicInfo overrides
OVERRIDES="${OVERRIDES} BasicInfo.topic=\"${TOPIC}\""
OVERRIDES="${OVERRIDES} BasicInfo.output_base_dir=\"${OUTPUT_DIR}\""
OVERRIDES="${OVERRIDES} BasicInfo.base_dir=\"${PROJECT_ROOT}\""
OVERRIDES="${OVERRIDES} BasicInfo.cache_path=\"./database\""
OVERRIDES="${OVERRIDES} BasicInfo.debug=${DEBUG}"
OVERRIDES="${OVERRIDES} BasicInfo.evaluation_save_path=\"${EVAL_SAVE_PATH}\""

# Add topic path if provided
if [ -n "$TOPIC_PATH" ]; then
    OVERRIDES="${OVERRIDES} BasicInfo.topic_path=\"${TOPIC_PATH}\""
fi

# Add save path if provided (will be overridden per topic in the loop)
# The Python script handles topic-specific save paths, but we can set base
# For single topic run, we can set save_path directly
if [ -n "$SAVE_PATH" ]; then
    # Convert topic to safe filename
    SAFE_TOPIC=$(echo "$TOPIC" | sed 's/ /_/g' | sed 's/[^a-zA-Z0-9_-]//g')
    FULL_SAVE_PATH="${SAVE_PATH}/${SAFE_TOPIC}.md"
    OVERRIDES="${OVERRIDES} BasicInfo.save_path=\"${FULL_SAVE_PATH}\""
fi

if [ -n "$SAVE_JSON_PATH" ]; then
    SAFE_TOPIC=$(echo "$TOPIC" | sed 's/ /_/g' | sed 's/[^a-zA-Z0-9_-]//g')
    FULL_SAVE_JSON_PATH="${SAVE_JSON_PATH}/${SAFE_TOPIC}.json"
    OVERRIDES="${OVERRIDES} BasicInfo.save_json_path=\"${FULL_SAVE_JSON_PATH}\""
fi

# APIInfo overrides (only if provided)
if [ -n "$API_KEY" ]; then
    OVERRIDES="${OVERRIDES} APIInfo.llm_api_key=\"${API_KEY}\""
fi

if [ -n "$API_BASE_URL" ]; then
    OVERRIDES="${OVERRIDES} APIInfo.llm_api_base_url=\"${API_BASE_URL}\""
fi

if [ -n "$MODEL_NAME" ]; then
    OVERRIDES="${OVERRIDES} APIInfo.llm_model_name=\"${MODEL_NAME}\""
fi

# ModuleInfo overrides (only if provided)
if [ -n "$MAX_SEED_PAPER" ]; then
    OVERRIDES="${OVERRIDES} ModuleInfo.WorkCollector.max_seed_paper_num=${MAX_SEED_PAPER}"
fi

if [ -n "$GRAPH_DEPTH" ]; then
    OVERRIDES="${OVERRIDES} ModuleInfo.WorkCollector.reference_graph_depth=${GRAPH_DEPTH}"
fi

if [ -n "$DB_PATH" ]; then
    OVERRIDES="${OVERRIDES} ModuleInfo.PaperGraphRetriever.db_path=\"${DB_PATH}\""
fi

# Extract config name from yaml path
CONFIG_NAME=$(basename "$CONFIG_YAML" .yaml)

echo "=================================================="
echo "Deep Survey Custom Batch Runner"
echo "=================================================="
echo "Topic:            ${TOPIC}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config File:      ${CONFIG_YAML}"
echo "Evaluation Save:  ${EVAL_SAVE_PATH}"
echo ""
echo "Custom Parameters:"
[ -n "$API_KEY" ] && echo "  API Key:         [PROVIDED]"
[ -n "$API_BASE_URL" ] && echo "  API Base URL:    ${API_BASE_URL}"
[ -n "$MODEL_NAME" ] && echo "  Model:           ${MODEL_NAME}"
echo "  Debug:            ${DEBUG}"
echo "  Max Seed Paper:   ${MAX_SEED_PAPER}"
echo "  Graph Depth:      ${GRAPH_DEPTH}"
[ -n "$DB_PATH" ] && echo "  DB Path:         ${DB_PATH}"
echo ""
echo "Hydra Overrides:"
echo "${OVERRIDES}" | fold -w 80 -s
echo "=================================================="

# Run the Python script with Hydra overrides
# Note: Hydra will merge the overrides with the base config YAML
python scripts/run_deep_survey_batch.py \
    --config-path "../config" \
    --config-name "${CONFIG_NAME}" \
    ${OVERRIDES}

echo ""
echo "=================================================="
echo "Pipeline completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=================================================="
