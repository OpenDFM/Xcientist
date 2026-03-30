#!/bin/bash

# ============================================================
# Deep Survey CLI - Bash Wrapper
# 
# This script provides a convenient way to run Deep Survey
# with parameters specified from the command line, overriding
# the YAML configuration.
#
# Usage:
#   ./run_deep_survey_cli.sh --topic "Your Topic" --output_path "./outputs"
#   ./run_deep_survey_cli.sh --topic "LLM Agent" --api_key "your-key" ...
#
# ============================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default config
CONFIG="deep_survey_fast"

# Show help
show_help() {
    cat << EOF
Deep Survey CLI - Bash Wrapper

Usage:
    $0 [OPTIONS]

Required Options:
    --topic TOPIC              Topic to search for (REQUIRED)
    --output_path PATH         Output directory for results (optional, will use default if not set)

API Configuration Options:
    --api_key KEY              LLM API key (overrides yaml)
    --api_base_url URL         LLM API base URL (overrides yaml)
    --model MODEL              LLM model name (overrides yaml)
    --semantic_scholar_api_key KEY   Semantic Scholar API key

Basic Info Options:
    --base_dir PATH            Base directory for the project
    --cache_path PATH          Cache path for database
    --save_path PATH           Full save path for survey markdown
    --save_json_path PATH      Full save path for survey JSON
    --evaluation_save_path PATH  Path to save evaluation results
    --debug [true|false]       Debug mode (default: from yaml)
    --error_conservatism_mode [true|false]  Error conservatism mode

WorkCollector Options:
    --max_seed_paper_num NUM   Maximum number of seed papers
    --reference_graph_depth DEPTH  Reference graph depth for expansion
    --related_work_top_k NUM   Top K related works to retrieve
    --use_seed_filter_llm [true|false]  Use LLM to filter seed papers
    --llm_seed_threshold NUM   LLM seed filter threshold (1-5)

Database Options:
    --default_top_k NUM        Default top K for database retrieval

WorkAnalyzer Options:
    --abstract_only_mode [true|false]  Use abstract only mode
    --clustering_temperature TEMP  Clustering temperature

SurveyGenerator Options:
    --include_initial_analysis [true|false]  Include initial analysis
    --include_relation_graph [true|false]  Include relation graph
    --include_relation_table [true|false]  Include relation table

Judge Options:
    --judge_api_key KEY        Judge LLM API key
    --judge_api_base_url URL   Judge LLM API base URL
    --judge_model MODEL        Judge model name

Config Options:
    --config CONFIG            Base config file name (default: deep_survey_fast)

Other Options:
    --help                     Show this help message

Examples:
    # Basic usage with topic and output path
    $0 --topic "LLM Agent Memory System" --output_path "./outputs/my-survey"
    
    # With API configuration
    $0 --topic "Vision Transformers" \\
        --api_key "your-key" \\
        --api_base_url "https://api.example.com" \\
        --output_path "./outputs/vit"
    
    # With processing parameters
    $0 --topic "Graph Neural Networks" \\
        --max_seed_paper_num 20 \\
        --reference_graph_depth 2 \\
        --output_path "./outputs/gnn"
    
    # Debug mode
    $0 --topic "My Topic" --debug true --output_path "./outputs/debug"
    
    # Full custom configuration
    $0 --topic "Deep Learning" \\
        --api_key "key" \\
        --api_base_url "url" \\
        --model "minimax-m2.5" \\
        --max_seed_paper_num 10 \\
        --reference_graph_depth 1 \\
        --include_relation_graph true \\
        --include_relation_table true \\
        --output_path "./outputs/dl"

Environment Variables:
    The following environment variables can also be used (they will be used if --api_* options are not provided):
    - LLM_API_KEY
    - LLM_API_BASE_URL
    - LLM_MODEL_NAME
EOF
}

# Check if help is requested
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        show_help
        exit 0
    fi
done

# Check required arguments
TOPIC=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --topic)
            TOPIC="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --api_base_url)
            API_BASE_URL="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --semantic_scholar_api_key)
            SEMANTIC_SCHOLAR_API_KEY="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --cache_path)
            CACHE_PATH="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --save_json_path)
            SAVE_JSON_PATH="$2"
            shift 2
            ;;
        --evaluation_save_path)
            EVALUATION_SAVE_PATH="$2"
            shift 2
            ;;
        --debug)
            DEBUG="$2"
            shift 2
            ;;
        --error_conservatism_mode)
            ERROR_CONSERVATISM_MODE="$2"
            shift 2
            ;;
        --max_seed_paper_num)
            MAX_SEED_PAPER_NUM="$2"
            shift 2
            ;;
        --reference_graph_depth)
            REFERENCE_GRAPH_DEPTH="$2"
            shift 2
            ;;
        --related_work_top_k)
            RELATED_WORK_TOP_K="$2"
            shift 2
            ;;
        --use_seed_filter_llm)
            USE_SEED_FILTER_LLM="$2"
            shift 2
            ;;
        --llm_seed_threshold)
            LLM_SEED_THRESHOLD="$2"
            shift 2
            ;;
        --default_top_k)
            DEFAULT_TOP_K="$2"
            shift 2
            ;;
        --abstract_only_mode)
            ABSTRACT_ONLY_MODE="$2"
            shift 2
            ;;
        --clustering_temperature)
            CLUSTERING_TEMPERATURE="$2"
            shift 2
            ;;
        --include_initial_analysis)
            INCLUDE_INITIAL_ANALYSIS="$2"
            shift 2
            ;;
        --include_relation_graph)
            INCLUDE_RELATION_GRAPH="$2"
            shift 2
            ;;
        --include_relation_table)
            INCLUDE_RELATION_TABLE="$2"
            shift 2
            ;;
        --judge_api_key)
            JUDGE_API_KEY="$2"
            shift 2
            ;;
        --judge_api_base_url)
            JUDGE_API_BASE_URL="$2"
            shift 2
            ;;
        --judge_model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
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

# Use environment variables as fallback
: ${API_KEY:=$LLM_API_KEY}
: ${API_BASE_URL:=$LLM_API_BASE_URL}
: ${MODEL:=$LLM_MODEL_NAME}

# Change to project root
cd "$PROJECT_ROOT"

# Build command
CMD="python ${SCRIPT_DIR}/run_deep_survey_cli.py --topic \"${TOPIC}\" --config ${CONFIG}"

# Add optional parameters
[ -n "$OUTPUT_PATH" ] && CMD="$CMD --output_path \"${OUTPUT_PATH}\""
[ -n "$API_KEY" ] && CMD="$CMD --api_key \"${API_KEY}\""
[ -n "$API_BASE_URL" ] && CMD="$CMD --api_base_url \"${API_BASE_URL}\""
[ -n "$MODEL" ] && CMD="$CMD --model \"${MODEL}\""
[ -n "$SEMANTIC_SCHOLAR_API_KEY" ] && CMD="$CMD --semantic_scholar_api_key \"${SEMANTIC_SCHOLAR_API_KEY}\""
[ -n "$BASE_DIR" ] && CMD="$CMD --base_dir \"${BASE_DIR}\""
[ -n "$CACHE_PATH" ] && CMD="$CMD --cache_path \"${CACHE_PATH}\""
[ -n "$SAVE_PATH" ] && CMD="$CMD --save_path \"${SAVE_PATH}\""
[ -n "$SAVE_JSON_PATH" ] && CMD="$CMD --save_json_path \"${SAVE_JSON_PATH}\""
[ -n "$EVALUATION_SAVE_PATH" ] && CMD="$CMD --evaluation_save_path \"${EVALUATION_SAVE_PATH}\""
[ -n "$DEBUG" ] && CMD="$CMD --debug ${DEBUG}"
[ -n "$ERROR_CONSERVATISM_MODE" ] && CMD="$CMD --error_conservatism_mode ${ERROR_CONSERVATISM_MODE}"
[ -n "$MAX_SEED_PAPER_NUM" ] && CMD="$CMD --max_seed_paper_num ${MAX_SEED_PAPER_NUM}"
[ -n "$REFERENCE_GRAPH_DEPTH" ] && CMD="$CMD --reference_graph_depth ${REFERENCE_GRAPH_DEPTH}"
[ -n "$RELATED_WORK_TOP_K" ] && CMD="$CMD --related_work_top_k ${RELATED_WORK_TOP_K}"
[ -n "$USE_SEED_FILTER_LLM" ] && CMD="$CMD --use_seed_filter_llm ${USE_SEED_FILTER_LLM}"
[ -n "$LLM_SEED_THRESHOLD" ] && CMD="$CMD --llm_seed_threshold ${LLM_SEED_THRESHOLD}"
[ -n "$DEFAULT_TOP_K" ] && CMD="$CMD --default_top_k ${DEFAULT_TOP_K}"
[ -n "$ABSTRACT_ONLY_MODE" ] && CMD="$CMD --abstract_only_mode ${ABSTRACT_ONLY_MODE}"
[ -n "$CLUSTERING_TEMPERATURE" ] && CMD="$CMD --clustering_temperature ${CLUSTERING_TEMPERATURE}"
[ -n "$INCLUDE_INITIAL_ANALYSIS" ] && CMD="$CMD --include_initial_analysis ${INCLUDE_INITIAL_ANALYSIS}"
[ -n "$INCLUDE_RELATION_GRAPH" ] && CMD="$CMD --include_relation_graph ${INCLUDE_RELATION_GRAPH}"
[ -n "$INCLUDE_RELATION_TABLE" ] && CMD="$CMD --include_relation_table ${INCLUDE_RELATION_TABLE}"
[ -n "$JUDGE_API_KEY" ] && CMD="$CMD --judge_api_key \"${JUDGE_API_KEY}\""
[ -n "$JUDGE_API_BASE_URL" ] && CMD="$CMD --judge_api_base_url \"${JUDGE_API_BASE_URL}\""
[ -n "$JUDGE_MODEL" ] && CMD="$CMD --judge_model \"${JUDGE_MODEL}\""

echo "=================================================="
echo "Deep Survey CLI - Bash Wrapper"
echo "=================================================="
echo "Topic:            ${TOPIC}"
echo "Config:           ${CONFIG}.yaml"
echo "Output Path:      ${OUTPUT_PATH:-<default>}"
echo ""
echo "Running command:"
echo "$CMD"
echo "=================================================="
echo ""

# Execute the command
eval "$CMD"

echo ""
echo "=================================================="
echo "Pipeline completed!"
echo "=================================================="
