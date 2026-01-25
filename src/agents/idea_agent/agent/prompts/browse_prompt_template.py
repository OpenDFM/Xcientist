BROWSE_PROMPT_TEMPLATE = (
    "Please read the source content and answer a following question.\n"
    "If there is no relevant information, return an empty JSON array.\n"
    "{schema}\n"
    "---begin of source content---\n"
    "{source_text}\n"
    "---end of source content---\n\n"
    "Question: {browse_query}\n"
    "Answer with JSON only."
)
