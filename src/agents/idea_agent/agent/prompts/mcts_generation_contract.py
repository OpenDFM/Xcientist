MCTS_IDEA_SKILL_GENERATION_PROMPT = """
You control the expansion step for a mature research idea under a frozen IdeaContract.
Do NOT rewrite a full paper. Output only minimal, checkable SkillOutput deltas.

Topic: {topic}

Mature idea context (center the children around this, but do NOT self-censor if you deviate):
{mature_idea}

IdeaContract (frozen, cannot be violated):
{idea_contract}

Current node summary (parent state):
{current_summary}

Retrieved memory snippets:
{memory_bundle}

Allowed operators (choose ONE per child, never invent new ones):
{edit_operators}

Global constraints (NEVER violate):
{constraints}

Guidelines for each child (do your best, but NEVER return an empty children list):
- Output SkillOutput only; no full-paper rewrite.
- Prefer a single core mechanism; if you need extra steps, place them in implementation_notes.
- Include experiment_patch with regression/ablation/stress tests whenever possible.
- Try to reference invariants via anchor_mapping/invariant_impact, but if unsure, still output a child.
- If budget_ceiling exists, mention budget impact in implementation_notes.

Return up to {max_children} mutually distinct children centered on the mature idea.
If a child drifts away from the mature idea, still output it (do NOT self-censor); the evaluator will penalize misalignment later.

STRICT OUTPUT: valid JSON with schema:
{{
  "children": [
    {{
      "operator": "operator_name_from_list",
      "target_defects": ["string"],
      "skill_output": {{
        "delta": {{
          "intervention": "what changes",
          "mechanism": "single core mechanism",
          "implementation_notes": "how it is implemented + budget impact if any"
        }},
        "experiment_patch": {{
          "regression_tests": ["reproduce parent key metrics"],
          "ablation_tests": ["isolate delta vs parent"],
          "stress_tests": ["pressure test aligned to parent failure modes"]
        }},
        "risk_patch": ["dominant new risks"],
        "introduced_concepts": ["new_term_1", "new_term_2"],
        "anchor_mapping": {{
          "M1": "map invariant -> delta/experiment sentence/step",
          "E1": "map invariant -> delta/experiment sentence/step"
        }},
        "invariant_impact": {{
          "M1": {{"status": "preserve|modify|violate", "reason": "...", "compensation": "if modify, add compensating experiment"}},
          "E1": {{"status": "preserve|modify|violate", "reason": "...", "compensation": "..."}}
        }},
        "memory_refs": ["Field#1", "Recipe#2"]
      }}
    }}
  ]
}}

Do not include prose outside JSON.
"""
