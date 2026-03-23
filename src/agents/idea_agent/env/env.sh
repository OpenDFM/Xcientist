export OPENAI_API_KEY="sk-ccvbi9mee9jjkgu6sunujtrcnk8lu78hxqya9en30o4gr56z"
export OPENAI_BASE_URL="https://api.xiaomimimo.com/v1"
export S2_API_KEY="1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ"
export S2_API_TIMEOUT="60"
export MINERU_MODEL_SOURCE=modelscope

export IDEA_AGENT_PARALLELISM="1"
export IDEA_AGENT_RAG_CONFIG="src/agents/survey_agent/config/outcomeRAG.yaml"

# Example:
# python src/agents/idea_agent/scripts/run.py --topics "Multimodal Large Language Models" \
#   --parallelism "$IDEA_AGENT_PARALLELISM" \
#   --rag-config "$IDEA_AGENT_RAG_CONFIG"
