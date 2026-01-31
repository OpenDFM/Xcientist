MCTS_IDEA_SKILL_REPAIR_PROMPT = """
You are repairing a SkillOutput to satisfy IdeaContract constraints.
Do NOT rewrite full idea. Only fix the SkillOutput fields.

IdeaContract:
{idea_contract}

Parent idea (JSON):
{parent_idea}

Original SkillOutput (may be invalid):
{skill_output}

Validation errors to fix:
{errors}

Return STRICT JSON with the corrected SkillOutput only:
{{
  "delta": {{"intervention": "...", "mechanism": "...", "implementation_notes": "..."}},
  "experiment_patch": {{"regression_tests": ["..."], "ablation_tests": ["..."], "stress_tests": ["..."]}},
  "risk_patch": ["..."],
  "introduced_concepts": ["..."],
  "anchor_mapping": {{"M1": "..."}},
  "invariant_impact": {{"M1": {{"status": "preserve|modify", "reason": "...", "compensation": "..."}}}},
  "memory_refs": ["Field#1"]
}}

Additional repair rules:
- mechanism must be a single, atomic clause. Do NOT use conjunctions ("and", "+", "&", "plus", ";") in the mechanism field.
- If multiple steps are needed, keep mechanism atomic and move the extra steps to implementation_notes.
"""
