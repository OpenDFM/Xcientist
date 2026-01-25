DATASET_QUERY_GENERATION_PROMPT = (
    "You are generating web search queries to find datasets. "
    "Output JSON: {{\"queries\": [..]}}. "
    "Rules: 4-6 queries, each <= 6 words; all queries targeting Google Dataset Search; "
    "mix broad and specific; avoid overly long keyword chains; no explanations.\n"
    "Topic: {topic}\nTask: {task}\nIdeaTitle: {idea_title}\nIdeaContext: {idea_context}\n"
    "Domain: {domain}\nDataType: {data_type}\nModalities: {modalities}\nEvaluationAxes: {evaluation_axes}\n"
    "ReferenceTitles: {reference_titles}\n"
)
