THEORY_TRANSFER_QUERY_PROMPT = """
You are preparing cross-domain retrieval for the skill `theory-transfer-injection`.

Write one retrieval query for the CURRENT idea. The query must clearly state:
1. what missing mechanism/content the current idea still needs, and
2. what role that missing content should play inside the current idea.

== Constraints ==
- The idea must stay in its fixed root domain(s): {root_domains}
- The query is for retrieval, so it should describe the needed mechanism/function, not ask a vague question.
- Focus on one missing mechanism only.
- Keep the query concise but specific.

== Topic == 
{topic}

== Current idea ==
{idea}

== Compiled edit plan ==
{edit_plan}

Return STRICT JSON:
{{
  "query": "concise retrieval query",
  "needed_content": "what is missing in the current idea",
  "expected_role": "how the missing content should function in the current idea"
}}
"""
