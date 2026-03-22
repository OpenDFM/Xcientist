export CONFIG_NAME="deep_survey_settings"

export https_proxy=http://127.0.0.1:1087

export CUDA_VISIBLE_DEVICES=2

python3 scripts/run_deep_survey_batch_settings.py \
  BasicInfo.topic_max_retry=2 \
  BasicInfo.debug=false \
  APIInfo.llm_api_key= \
  APIInfo.llm_api_base_url=http://122.193.22.114:8889/v1/chat/completions \
  APIInfo.llm_model_name=gpt-4o-mini \
  APIInfo.llm_max_context_length=128000 \
  APIInfo.llm_max_context_overhead_length=50000 \
  APIInfo.batch_chat_agent_worker=4 \
  APIInfo.semantic_scholar_api_key=1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ \
  APIInfo.arxiv_api_max_retry=3 \
  ModuleInfo.WorkAnalyzer.abstract_only_mode=false \
  ModuleInfo.WorkAnalyzer.cache_enabled=true \
  ModuleInfo.WorkAnalyzer.paper_reading_max_retry=5 \
  ModuleInfo.WorkAnalyzer.clustering_batch_size=16 \
  ModuleInfo.WorkAnalyzer.clustering_temperature=1.0 \
  ModuleInfo.WorkCollector.sentence_transformer_model=sentence-transformers/all-MiniLM-L6-v2 \
  ModuleInfo.WorkCollector.max_seed_paper_num=10 \
  ModuleInfo.WorkCollector.reference_graph_depth=2 \
  ModuleInfo.WorkCollector.sentence_transformer_batch_size=32 \
  ModuleInfo.WorkCollector.related_work_top_k=20 \
  ModuleInfo.WorkCollector.related_work_threshold=0.5 \
  ModuleInfo.WorkCollector.related_work_threshold_for_llm=3 \
  ModuleInfo.SurveyGenerator.use_full_text_in_survey_generation=True \
  ModuleInfo.Judge.rubrics_eval_4_dimensions=true \
  ModuleInfo.Judge.citation_eval=true \
  ModuleInfo.Judge.remove_failed_citation_in_eval=true \
  ModuleInfo.Judge.nli_temperature=0.1 \
  ModuleInfo.Judge.citation_quality_threshold=0.5 \
  ModuleInfo.Judge.model=gemini-3-flash-preview
  > ./logs/bash_run.log 2>&1

if [ $? -eq 0 ]; then
    echo "Experiment completed successfully."
else
    echo "Experiment failed. Check logs."
fi