#!/bin/bash

set -euo pipefail

# ----- Direct configuration (edit here as needed) -----
TOPICS_FILE="topics_demo.txt"
BASE_PATH="./output/survey_bench_ablation"
GPU="0"
SECTION_NUM="7"
SUBSECTION_LEN="700"
RAG_NUM="60"
OUTLINE_REFERENCE_NUM="1500"
DB_PATH="./database/database"
EMBEDDING_MODEL="../SurveyForge/gte-large-en-v1.5"
MAX_RETRY="2"

# Multiple models: space-separated model names. One shared API key/base below.
MODELS="gpt-4o-mini gemini-3-flash gemini-3-pro-thinking grok-4.1-fast llama-3.3-70b-instruct claude-sonnet-4.5 gemini-2.5-flash deepseek-v3.2 qwen3-vl-32b-instruct qwen3-235b-a22b"
API_KEY=""
API_BASE=""

log() { printf "%s %s\n" "$(date '+%F %T')" "$*"; }

run_one() {
  local topic="$1" model="$2" exp_num="$3"
  local save_path="${BASE_PATH}/${topic}/${model}/exp_${exp_num}"

#   if [[ -d "$save_path" ]]; then
#     log "Skip: ${topic} ${model} exp_${exp_num} already exists"
#     return 0
#   fi
  mkdir -p "$save_path"

  local attempt
  for (( attempt=1; attempt<=MAX_RETRY; attempt++ )); do
    log "Run topic=${topic} model=${model} exp=${exp_num} attempt=${attempt}"
    if python3 main.py \
      --topic "$topic" \
      --gpu "$GPU" \
      --saving_path "$save_path" \
      --model "$model" \
      --section_num "$SECTION_NUM" \
      --subsection_len "$SUBSECTION_LEN" \
      --rag_num "$RAG_NUM" \
      --outline_reference_num "$OUTLINE_REFERENCE_NUM" \
      --db_path "$DB_PATH" \
      --embedding_model "$EMBEDDING_MODEL" \
      --api_key "$API_KEY" \
      --api_url "$API_BASE"; then
      log "Success: ${topic} ${model} exp_${exp_num}"
      return 0
    fi
    log "Failed attempt ${attempt} for ${topic} ${model} exp_${exp_num}"
    sleep 5
  done
  log "Max retries reached: ${topic} ${model} exp_${exp_num}"
  return 1
}

main() {
  if [[ ! -f "$TOPICS_FILE" ]]; then
    log "topics file not found: $TOPICS_FILE"; exit 1
  fi

  mkdir -p "$BASE_PATH"

  while IFS= read -r topic; do
    [[ -z "$topic" ]] && continue
    log "=== Topic: $topic ==="
    local m
    for m in $MODELS; do
      run_one "$topic" "$m" 1
    done
  done < "$TOPICS_FILE"

  log "All ablations finished. Output in $BASE_PATH"
}

main "$@"
