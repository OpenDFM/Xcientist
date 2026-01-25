EXTRACT_CANDIDATE_NAMES_PROMPT = (
    "You are extracting names from web search snippets. "
    "Return JSON array of up to {max_names} {kind} names only. "
    "Prefer canonical names, no commentary."
    "\nTask: {task}\nSearchResults:\n{results}\n"
)
