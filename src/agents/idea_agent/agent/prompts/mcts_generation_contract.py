MCTS_IDEA_SKILL_GENERATION_PROMPT = """
You control the expansion step for a mature research idea under a frozen IdeaContract.
Do NOT rewrite a full paper. Output only minimal, checkable SkillOutput deltas.

Topic: {topic}

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

Hard rules for each child:
- Output ONLY SkillOutput; no full-paper rewrite.
- introduced_concepts length <= 2.
- main mechanism count must be 1 (no multi-mechanism chimera).
- anchor_mapping must cover >=70% of invariants (use invariant IDs).
- invariant_impact cannot contain "violate".
- experiment_patch MUST include: regression_tests, ablation_tests, stress_tests.
- If budget_ceiling exists, include budget impact in implementation_notes and stay within it.

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
