PREPROCESS_CANDIDATE_NAMES_PROMPT = (
    "You are filtering a list of candidate {kind} names. "
    "Remove irrelevant items, generic terms, or model-only names. "
    "Keep canonical {kind} names. Deduplicate. Return JSON array of names only."
    "\nTopic: {topic}\nTask: {task}\nCandidates: {candidates}\n"
)
