BASELINE_QUERY_GENERATION_PROMPT = (
    "You are generating web search queries to find strong baselines for a research idea. "
    "Given the inputs, output JSON: {{\"queries\": [..]}}. "
    "Rules: 4-6 queries, each <= 5 words; mix broad and specific; include at most 2 queries with site: and at most 2 with github; "
    "avoid overly long keyword chains; do not include explanations.\n"
    "Topic: {topic}\nTask: {task}\nIdeaTitle: {idea_title}\nIdeaContext: {idea_context}\n"
    "Benchmarks: {benchmarks}\nMethodFamily: {method_family}\nKeyComponents: {key_components}\nEvaluationAxes: {evaluation_axes}\n"
    "ReferenceTitles: {reference_titles}\n"
)
