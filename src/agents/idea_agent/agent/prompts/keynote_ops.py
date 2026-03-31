KEYNOTE_SCORING_PROMPT = """
You are scoring one survey-cited paper keynote for LigAgent.

== Topic ==
{topic}

== Retrieval Query ==
{rag_query}

== Paper Title ==
{title}

== Keynote ==
{keynote}

Return STRICT JSON ONLY:
{{
  "score": 0
}}

Scoring rule:
- integer only, 0 to 100
- higher means more relevant and more important for grounding idea generation under the given topic/query
- score based only on this keynote
"""


KEYNOTE_SINGLE_COMPRESSION_PROMPT = """
You are compressing one survey-cited paper keynote for LigAgent.

== Topic ==
{topic}

== Retrieval Query ==
{rag_query}

== Paper Title ==
{title}

== Keynote ==
{keynote}

Return STRICT JSON ONLY:
{{
  "summary": "",
  "insight": ""
}}

Requirements:
- `summary`: 1-2 factual sentences
- `insight`: 1-3 concise sentence capturing the most decision-useful mechanism, limitation, or evaluation takeaway
- use only the keynote
"""


KEYNOTE_GROUP_SUMMARY_PROMPT = """
You are summarizing the lower-priority survey-cited paper keynotes for LigAgent.

== Topic ==
{topic}

== Retrieval Query ==
{rag_query}

== Papers ==
{papers}

Return STRICT JSON ONLY:
{{
  "summary": ""
}}

Requirements:
- produce one concise synthesis summary for the whole set
- emphasize recurring mechanisms, limitations, and evaluation signals
- do not enumerate every paper
- use only the provided keynotes
"""
