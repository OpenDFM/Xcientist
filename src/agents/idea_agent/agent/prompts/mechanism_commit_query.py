from __future__ import annotations

from typing import Optional

from src.agents.idea_agent.agent.prompts.prompt_modes import (
    is_conceptual_surprise_mode,
)


MECHANISM_COMMIT_QUERY_PROMPT = """
You are preparing literature grounding for the skill `mechanism-commit-innovation`.

Write one retrieval query for the CURRENT idea as a single short paragraph. The query must clearly state:
1. what concrete component or mechanism the current idea needs, and
2. what role that component should play on the primary execution path.

== Constraints ==
- The idea must stay in its fixed root domain(s): {root_domains}
- The query is for retrieval, so it should describe the needed mechanism/component, not ask a vague question.
- Focus on one mechanism only.
- Keep the query to 1-2 sentences.
- Center the query on the needed component and its role, not on a long diagnosis of the current idea's weaknesses.
- If the compiled edit plan is a component replacement, anchor the query to that existing component role and describe the stronger internal mechanism it should use.
- If `refinement_scope` or the current idea narrows the edit to an existing subsystem, keep the query at that same subsystem granularity.
- Prefer refining or replacing an existing component over introducing a broader architecture-level coordination policy unless the current idea already centers such a policy.
- Do not list candidate implementations, paper families, examples, or long sub-mechanism enumerations.
- If the current idea appears training-free, prefer retrieval queries for training-free mechanisms or inference-time rules. Do not steer retrieval toward new training modules unless a training shift appears indispensable.
- Do NOT use threshold/gating/suppression/quota language in `query`, `mechanism_gap`, or `expected_role`.
- If the current idea already uses that language, restate the missing mechanism in neutral functional terms instead of repeating those tokens in the retrieval query.

== Topic ==
{topic}

== Refinement scope (optional; if empty, ignore) ==
{refinement_scope}

== Current idea ==
{idea}

== Compiled edit plan ==
{edit_plan}

Return STRICT JSON:
{{
  "query": "one short paragraph describing the needed component and its role",
  "mechanism_gap": "short noun phrase or short sentence naming the needed component/mechanism in positive functional terms",
  "expected_role": "one short sentence stating the role this component should play on the primary execution path"
}}
"""


CONCEPTUAL_SURPRISE_MECHANISM_COMMIT_QUERY_PROMPT = """
You are preparing literature grounding for the skill `mechanism-commit-innovation`.

Write one retrieval query for the CURRENT idea as a single short paragraph. The query must clearly state:
1. which local assumption, framing, or principle in the current idea is still weak or underspecified,
2. what concrete mechanism is needed to realize a stronger commitment on that same method axis, and
3. what role that mechanism should play on the primary execution path.

== Constraints ==
- The idea must stay in its fixed root domain(s): {root_domains}
- Treat this as a local thesis repair for the current idea, not a replacement direction.
- The query is for retrieval, so it should describe the needed grounding pattern, not ask a vague question.
- Focus on one conceptual repair and one mechanism only.
- Keep the query to 1-2 sentences.
- Center the query on the needed component and its role, not on a long diagnosis of the current idea's weaknesses.
- If the compiled edit plan is a component replacement, anchor the query to that existing component role and the stronger internal mechanism that would replace it.
- If `refinement_scope` or the current idea narrows the edit to an existing subsystem, keep the query at that same subsystem granularity.
- Prefer a local repair of an existing component over introducing a broader architecture-level coordination policy unless the current idea already centers such a policy.
- Do not list candidate implementations, paper families, examples, or long sub-mechanism enumerations.
- If the current idea appears training-free, prefer conceptual repairs that can be realized without adding new training machinery. Only seek training-based grounding if the repair seems impossible otherwise.
- Do NOT use threshold/gating/suppression/quota language in `query`, `mechanism_gap`, or `expected_role`.
- If the current idea already uses that language, restate the local repair in neutral functional terms instead of repeating those tokens in the retrieval query.

== Topic ==
{topic}

== Refinement scope (optional; if empty, ignore) ==
{refinement_scope}

== Current idea ==
{idea}

== Compiled edit plan ==
{edit_plan}

Return STRICT JSON:
{{
  "query": "one short paragraph describing the needed component and its role",
  "mechanism_gap": "short noun phrase or short sentence naming the mechanism needed to realize the local conceptual repair",
  "expected_role": "one short sentence stating the role this component should play on the primary execution path"
}}
"""


def get_mechanism_commit_query_prompt(mode: Optional[str] = None) -> str:
    if is_conceptual_surprise_mode(mode):
        return CONCEPTUAL_SURPRISE_MECHANISM_COMMIT_QUERY_PROMPT
    return MECHANISM_COMMIT_QUERY_PROMPT
