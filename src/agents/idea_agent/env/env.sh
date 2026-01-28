export OPENAI_API_KEY="sk-BWZ0Kqbk3PvdF0zRFf69B63901B84e85A5B4D8B1AfE27e2e"
export OPENAI_BASE_URL="https://api.xi-ai.cn/v1"
export S2_API_KEY="1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ"
export S2_API_TIMEOUT="60"
export SERPER_API_KEY="7c7ca61c4c665f666902d03f3bca49c4b4b5bed4"
#export MINERU_MODEL_SOURCE=modelscope

export IDEA_AGENT_PARALLELISM="1"
export IDEA_AGENT_RAG_CONFIG="src/agents/survey_agent/config/outcomeRAG.yaml"


python src/agents/idea_agent/scripts/run.py
