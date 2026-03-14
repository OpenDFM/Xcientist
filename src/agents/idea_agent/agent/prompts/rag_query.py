SURVEY_SUBSECTION_RAG_PROMPT = """
You are an expert agent navigating inside a comprehensive survey paper. Your task is to query the survey's database to retrieve a specific methodological subsection.

== Topic ==
{topic}

== Mature idea (optional) ==
{mature_idea}

== The agent has ALREADY read these papers/sections ==
{papers}

== Task ==
Formulate ONE extremely brief search query string to retrieve the most relevant SUBSECTION from the survey that can inspire the next mechanism design. 

Constraints & Directives:
- TARGET SURVEY HEADINGS: Surveys are written with concise, high-level structural headings (e.g., "Hierarchical Planning", "Episodic Memory", "Reward Shaping", "State Abstraction"). Your query should mimic these theoretical, component-level phrases.
- EXTREME BREVITY: The query MUST be 5 to 8 words maximum. It must act as a strict keyword combination.
- NO META-WORDS: Do NOT use words like "survey", "review", "paper", "section", "subsection", or "chapter". The database already consists entirely of survey chunks.
- NO EVALUATION/QUESTIONS: Do NOT write conversational questions ("how to use MCTS") and do NOT use evaluation terms ("benchmark", "dataset", "SOTA").
- NOVELTY OVER REPETITION: Look at the `{papers}` already read. Aim the query at an adjacent or missing methodological cluster that has not been explored yet.

Return STRICT JSON ONLY matching this schema:
{{
  "query_design_scratchpad": {{
    "missing_mechanism": "What high-level mechanism or theoretical cluster is missing from the existing knowledge?",
    "heading_prediction": "If this missing mechanism were a section header in a highly-cited survey paper, what exact 5-8 words would it use?"
  }},
  "query": "The final 5-8 word string from your heading_prediction."
}}
"""

# Backward-compatible export used by the prompt registry.
RAG_QUERY_PROMPT = SURVEY_SUBSECTION_RAG_PROMPT
