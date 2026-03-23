from __future__ import annotations

from typing import Optional

from src.agents.idea_agent.agent.prompts.prompt_modes import (
    is_conceptual_surprise_mode,
)


THEORY_TRANSFER_QUERY_PROMPT = """
You are preparing cross-domain retrieval for the skill `theory-transfer-injection`.

Write one retrieval query for the CURRENT idea as a single short paragraph. The query must clearly state:
1. what missing mechanism/content the current idea needs, and
2. what role that missing content should play inside the current idea.

== Constraints ==
- The idea must stay in its fixed root domain(s): {root_domains}
- The query is for retrieval, so it should describe the needed mechanism/function, not ask a vague question.
- Focus on one missing mechanism only.
- Keep the query to 1-2 sentences.
- Center the query on the needed component/content and its role, not on a long diagnosis of the current idea's weaknesses.
- Do not list candidate implementations, paper families, examples, or long sub-mechanism enumerations.
- If the current idea appears training-free, prefer transferable principles that can be realized by training-free or inference-time mechanisms. Do not steer retrieval toward new training modules unless that shift appears indispensable.
- Do NOT use threshold/gating/suppression/quota language in `query`, `needed_content`, or `expected_role`.
- If the current idea already uses that language, restate the missing transferable content in neutral functional terms instead of repeating those tokens in the retrieval query.

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
  "query": "one short paragraph describing the needed component/content and its role",
  "needed_content": "short noun phrase or short sentence naming the missing transferable content in positive functional terms",
  "expected_role": "one short sentence stating the role this content should play in the current idea"
}}
"""


CONCEPTUAL_SURPRISE_THEORY_TRANSFER_QUERY_PROMPT = """
You are preparing cross-domain retrieval for the skill `theory-transfer-injection`.

Write one retrieval query for the CURRENT idea as a single short paragraph. The query must clearly state:
1. what transferable principle, invariant, or framing the current idea still needs,
2. what concrete mechanism/content should carry that local conceptual repair inside the current idea, and
3. what role that missing content should play inside the current idea.

== Constraints ==
- The idea must stay in its fixed root domain(s): {root_domains}
- Treat this as a local conceptual repair for the current idea, not a replacement paradigm.
- The query is for retrieval, so it should describe the needed transferable principle or mechanism, not ask a vague question.
- Focus on one missing principle/mechanism pair only.
- Keep the query to 1-2 sentences.
- Center the query on the needed component/content and its role, not on a long diagnosis of the current idea's weaknesses.
- Do not list candidate implementations, paper families, examples, or long sub-mechanism enumerations.
- If the current idea appears training-free, prefer transferable principles that preserve that character. Only seek training-based realizations when a non-training repair seems genuinely insufficient.
- Do NOT use threshold/gating/suppression/quota language in `query`, `needed_content`, or `expected_role`.
- If the current idea already uses that language, restate the transfer target in neutral functional terms instead of repeating those tokens in the retrieval query.

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
  "query": "one short paragraph describing the needed component/content and its role",
  "needed_content": "short noun phrase or short sentence naming the transferable principle or mechanism needed by the current idea",
  "expected_role": "one short sentence stating the role this content should play in the current idea"
}}
"""


def get_theory_transfer_query_prompt(mode: Optional[str] = None) -> str:
    if is_conceptual_surprise_mode(mode):
        return CONCEPTUAL_SURPRISE_THEORY_TRANSFER_QUERY_PROMPT
    return THEORY_TRANSFER_QUERY_PROMPT
