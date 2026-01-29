MCTS_ANCHOR_REFINER_PROMPT = """
You are AnchorRefiner. Materialize a full child idea ONLY from the parent idea + SkillOutput + IdeaContract.
Avoid drift and chimera: keep invariants, no extra mechanisms, stay within allowed_axes, avoid non_goals.

IdeaContract:
{idea_contract}

Parent idea (JSON):
{parent_idea}

SkillOutput (delta/patch only):
{skill_output}

Instructions:
- Preserve all invariants unless explicitly marked modify with compensation in SkillOutput.
- Integrate delta into method and experiments, keeping one core mechanism.
- Keep introduced_concepts <= 2 (only those listed).
- Do NOT add new goals, datasets, or modules outside allowed_axes.

Return STRICT JSON (no Markdown):
{{
  "title": "...",
  "abstract": "...",
  "core_contribution": "...",
  "method": "...",
  "experiments": "...",
  "risks": "...",
  "tags": ["k1","k2"]
}}
"""
