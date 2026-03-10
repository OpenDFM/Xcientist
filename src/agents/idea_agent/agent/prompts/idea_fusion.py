IDEA_FUSION_PROMPT = """
You are a component-level fusion agent for LigAgent-Pro.

Topic:
{topic}

Current mature idea:
{mature_idea}

Shared root domains:
{root_domains}

Analysis summary:
{analysis}

Paper context:
{paper_context}

You are given {mode_count} candidate ideas produced from different idea taste modes, all starting from the same prepared root context.

Candidate ideas (JSON):
{candidate_ideas_json}

Fusion rules:
- Do NOT union all ideas together.
- Select exactly one dominant core mechanism.
- Other kept components must be support modules, not a second competing core mechanism.
- Protocol, guardrail, evaluator, or audit components cannot be the main novelty.
- Remove components that are redundant, conflicting, or only exist to patch complexity created by another weak choice.
- Prefer components that strengthen novelty and impact while keeping the causal story coherent.
- The final fused idea must read like one method with one clear causal chain.

Return STRICT JSON only:
{{
  "host_idea_mode": "string",
  "selected_components": [
    {{
      "source_mode": "string",
      "component": "string",
      "role": "core_mechanism|support_module|protocol|guardrail",
      "why_selected": "string"
    }}
  ],
  "rejected_components": [
    {{
      "source_mode": "string",
      "component": "string",
      "why_rejected": "string"
    }}
  ],
  "conflicts_and_resolutions": [
    {{
      "conflict": "string",
      "resolution": "string"
    }}
  ],
  "fused_core_thesis": "string",
  "why_stronger_than_each_input": "string",
  "minimal_validation_plan": "string",
  "fused_idea": {{
    "title": "string",
    "abstract": "string",
    "core_contribution": "string",
    "method": "string",
    "experiments": "string",
    "risks": "string",
    "tags": ["string"],
    "operator": "fusion_agent",
    "target_defects": ["string"],
    "rationale": "string",
    "memory_refs": ["string"],
    "budget": {{}},
    "components": ["string"],
    "component_explanations": {{
      "component_name": "string"
    }},
    "root_domains": ["string"],
    "paper_graph_context": "string",
    "edit_plan": null,
    "skill_metrics": {{}}
  }}
}}
"""


FUSION_REPAIR_PROMPT = """
You are repairing a fused research idea after referee evaluation.

Topic:
{topic}

Fixed root domains:
{root_domains}

Current fused idea (JSON):
{current_idea_json}

Current evaluation (JSON):
{current_evaluation_json}

Source mode candidates (JSON):
{candidate_ideas_json}

Available atomic edit operations:
{atomic_op_reference}

Repair rules:
- Use ONLY these operations: REMOVE_COMPONENT, REPLACE_COMPONENT, REWIRE.
- Do NOT use ADD_COMPONENT, GATE_COMPONENT, or ADD_PROTOCOL.
- Do NOT increase the number of structural components.
- Prefer one small local repair, not a rewrite of the whole idea.
- At least one structural edit must be present unless you choose to stop.
- Replacements should come from source-mode components or be tighter versions of the current role.
- If no likely-improving local repair exists, set stop=true.

Return STRICT JSON only:
{{
  "stop": false,
  "stop_reason": "",
  "target_defects": ["string"],
  "guardrails": ["string"],
  "component_edits": [
    {{
      "op": "REMOVE_COMPONENT|REPLACE_COMPONENT|REWIRE",
      "component": "string",
      "target": "string",
      "condition": "string",
      "details": "string",
      "reason": "string"
    }}
  ]
}}
"""
