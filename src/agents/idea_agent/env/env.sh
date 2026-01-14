export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8.3.2"
export OPENAI_API_KEY="sk-ccvbi9mee9jjkgu6sunujtrcnk8lu78hxqya9en30o4gr56z"
export OPENAI_BASE_URL="https://api.xiaomimimo.com/v1"
export S2_API_KEY="1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ"
export S2_API_TIMEOUT="60"

python src/agents/idea_agent/scripts/run.py --topics "Multimodal Large Language Models" --parallelism 1 --rag-config "/home/lococo/project/ResearchAgent/src/agents/survey_agent/config/outcomeRAG.yaml"