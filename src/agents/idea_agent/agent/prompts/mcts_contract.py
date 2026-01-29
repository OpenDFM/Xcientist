MCTS_IDEA_CONTRACT_PROMPT = """
You are extracting a frozen IdeaContract from a mature research idea. Do not invent new claims.

Mature idea:
{mature_idea}

Return STRICT JSON with:
{{
  "scope_statement": "...",
  "thesis": "...",
  "core_claims": ["...","..."],
  "mechanism_invariants": ["..."],
  "evaluation_invariants": ["..."],
  "non_goals": ["..."],
  "budget_ceiling": {{"latency": "...", "compute": "...", "memory": "...", "params": "..."}} or {{}},
  "allowed_axes": ["..."]
}}

Constraints:
- core_claims: 2-4 items.
- mechanism_invariants: immutable mechanism elements (2-5 items).
- evaluation_invariants: required tasks/metrics/protocols (1-4 items).
- allowed_axes: ONLY dimensions that children may vary.
- non_goals: explicitly list forbidden directions.
- budget_ceiling: include only if stated or implied; otherwise {{}}.
"""
