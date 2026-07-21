from __future__ import annotations

from typing import Optional

from src.agents.idea_agent.agent.prompts.prompt_modes import (
    is_conceptual_surprise_mode,
)


MCTS_IDEA_GENERATION_PROMPT = """
You control the expansion step of a memory-guided MCTS that iteratively rewrites research ideas.
Your mission is to surface strong, non-incremental research concepts rather than incremental fixes.
- Bold mechanism commitments and new training contracts beat small gating/ensembling tweaks.
- At least one child must import an idea from another discipline or evaluation contract and tag it "moonshot".
- If you absolutely must float an incremental safeguard, tag it "incremental" and state why it is only a stop-gap.
- Never pitch a new benchmark/protocol/dataset as the primary contribution—evaluation ideas are acceptable only when they concretely enable a novel algorithmic mechanism and must be framed as support for that mechanism.

- Topic context: {topic}
- Current focus node summary:
{current_summary}
- Literature context synthesized from the latest downloaded papers:
{paper_context}

Retrieved natural-language memory (field knowledge, anti-patterns, fix routines):
{memory_bundle}

You must expand this node by applying the provided edit operators exactly once per child.
Operators (choose one per idea, never invent new ones):
{edit_operators}

Global constraints (NEVER violate):
{constraints}

Return up to {max_children} mutually distinct child ideas. Each child must:
1. Target at least one explicit defect (from evaluation tags, peer reviews, or the operator hints).
2. Document which operator was used and why it repairs the defect without triggering anti-patterns (no feature dumping, enforce fair baselines, expose failure modes, and respect resource limits).
3. Provide a structured idea payload with the required research sections plus risk surface tags.
4. Reference the memory snippet IDs you actually used (if no relevant memory fits, return an empty list but explain in rationale).
5. Introduce a concrete algorithmic intervention (new module, coupling, optimization step, or training signal); instrumentation-only or protocol/benchmark ideas are invalid unless they are secondary to, and tightly coupled with, a clearly described mechanism change.
6. Prefer the **mechanism-commit-innovation** operator whenever it is applicable. If you choose a different operator, explicitly justify why mechanism-commit is unsuitable for that child.
7. Inside each rationale, explicitly add "Review bar: <pass/fail + reason>" describing why expert reviewers would see it as a strong paper or what is missing.

STRICT OUTPUT: valid JSON with the following schema (do not wrap in Markdown):
{{
  "children": [
    {{
      "operator": "operator_name_from_list",
      "target_defects": ["string"],
      "title": "concise title",
      "abstract": "≤120 words abstract",
      "core_contribution": "focused statement of the new insight",
      "method": "key methodology steps",
      "experiments": "fair comparison protocol incl. baselines/eval metrics",
      "risks": "dominant risks or failure modes being tracked",
      "tags": ["k1","k2"],
      "memory_refs": ["Field#1","Recipe#2"],
      "anti_pattern_checks": {{
          "scope_control": "how the idea avoids feature dumping",
          "fair_baseline": "describe control protocol",
          "failure_reporting": "how failure modes will be surfaced"
      }},
      "rationale": "2 sentences on how the operator resolves the defect while respecting guardrails"
    }}
  ]
}}

Never invent data that contradicts the retrieved memory or operators. Keep children orthogonal.
"""


CONCEPTUAL_SURPRISE_MCTS_IDEA_GENERATION_PROMPT = MCTS_IDEA_GENERATION_PROMPT.replace(
    "5. Introduce a concrete algorithmic intervention (new module, coupling, optimization step, or training signal); instrumentation-only or protocol/benchmark ideas are invalid unless they are secondary to, and tightly coupled with, a clearly described mechanism change.\n"
    "6. Prefer the **mechanism-commit-innovation** operator whenever it is applicable. If you choose a different operator, explicitly justify why mechanism-commit is unsuitable for that child.\n"
    "7. Inside each rationale, explicitly add \"Review bar: <pass/fail + reason>\" describing why expert reviewers would see it as a strong paper or what is missing.\n",
    """5. Introduce a concrete algorithmic intervention (new module, coupling, optimization step, or training signal); instrumentation-only or protocol/benchmark ideas are invalid unless they are secondary to, and tightly coupled with, a clearly described mechanism change.
6. For each child, first sharpen one local scientific thesis: repair a weak assumption, propose a better principle, or reframe the parent idea on the same method axis. The concrete mechanism should realize that conceptual move rather than replace it.
7. Prefer the **mechanism-commit-innovation** operator whenever it is applicable. If you choose a different operator, explicitly justify why mechanism-commit is unsuitable for that child.
8. Inside each rationale, explicitly add "Review bar: <pass/fail + reason>" describing why expert reviewers would see it as a strong paper or what is missing.
""",
).replace(
    '"core_contribution": "focused statement of the new insight",',
    '"core_contribution": "focused statement of the new thesis, principle, reframing, or mechanism insight",',
).replace(
    '"method": "key methodology steps",',
    '"method": "start with the conceptual move being realized, then give key methodology steps",',
)


def get_mcts_generation_prompt(mode: Optional[str] = None) -> str:
    if is_conceptual_surprise_mode(mode):
        return CONCEPTUAL_SURPRISE_MCTS_IDEA_GENERATION_PROMPT
    return MCTS_IDEA_GENERATION_PROMPT
