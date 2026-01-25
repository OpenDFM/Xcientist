REACT_WEBSEARCH_PROMPT = (
    "You are running a multi-step web search for {kind} names. "
    "At each step, propose one search query or stop.\n"
    "Rules: Output JSON {{\"think\": \"...\", \"query\": \"...\", \"stop\": false}}. "
    "The think field must be a brief intent (<=12 words), no chain-of-thought. "
    "Prefer authoritative sources. Avoid repeating queries.\n"
    "For datasets, prefer HuggingFace/Kaggle/PapersWithCode. "
    "For baselines, prefer arXiv/GitHub.\n"
    "Avoid names already found: {found_names}\n"
    "SeedNames: {seed_names}\nTopic: {topic}\nTask: {task}\n"
    "ExecutedQueries: {executed}\nObservations:\n{obs}\n"
)
