MCTS_IDEA_GENERATION_PROMPT = """
You control the expansion step of a memory-guided MCTS that iteratively rewrites research ideas.
Your mission is to surface ICML/NeurIPS-ready concepts rather than incremental fixes.
- Bold mechanism commitments and new training contracts beat small gating/ensembling tweaks.
- At least one child must import an idea from another discipline or evaluation contract and tag it "moonshot".
- If you absolutely must float an incremental safeguard, tag it "incremental" and state why it is only a stop-gap.

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
5. Introduce a concrete algorithmic intervention (new module, coupling, optimization step, or training signal); instrumentation-only fixes are insufficient unless paired with a clearly described mechanism change.
6. Prefer the **mechanism-commit-innovation** operator whenever it is applicable. If you choose a different operator, explicitly justify why mechanism-commit is unsuitable for that child.
7. Inside each rationale, explicitly add "ICML bar: <pass/fail + reason>" describing why reviewers would see it as top-tier or what is missing.

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
          "resource_budget": "how the idea avoids feature dumping",
          "fair_baseline": "describe control protocol",
          "failure_reporting": "how failure modes will be surfaced"
      }},
      "rationale": "2 sentences on how the operator resolves the defect while respecting guardrails"
    }}
  ]
}}

Never invent data that contradicts the retrieved memory or operators. Keep children orthogonal.
"""
