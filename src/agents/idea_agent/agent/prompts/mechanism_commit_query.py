MECHANISM_COMMIT_QUERY_PROMPT = """
You are preparing literature grounding for the skill `mechanism-commit-innovation`.

Write one retrieval query for the CURRENT idea. The query must clearly state:
1. what concrete mechanism is still missing or underspecified in the current idea, and
2. what role that mechanism should play on the primary execution path.

== Constraints ==
- The idea must stay in its fixed root domain(s): {root_domains}
- The query is for retrieval, so it should describe the needed mechanism/component, not ask a vague question.
- Focus on one mechanism only.
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
  "mechanism_gap": "what concrete mechanism is missing or unclear",
  "expected_role": "how the mechanism should function on the primary execution path"
}}
"""
