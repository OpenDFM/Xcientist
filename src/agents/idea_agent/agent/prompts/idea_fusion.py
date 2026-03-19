IDEA_FUSION_PROMPT = """
You are a component-level fusion agent for LigAgent-Pro.

== Topic == 
{topic}

== Current mature idea ==
{mature_idea}

== Shared root domains ==
{root_domains}

== Analysis summary ==
{analysis}

== Paper context ==
{paper_context}

You are given {mode_count} candidate ideas produced from different idea taste modes, all starting from the same prepared root context.

== Candidate ideas (JSON) == 
{candidate_ideas_json}

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

== Fusion rules ==
- Do NOT union all ideas together.
- Select exactly one dominant core mechanism.
- Other kept components must be support modules, not a second competing core mechanism.
- Protocol, guardrail, evaluator, or audit components cannot be the main novelty.
- Remove components that are redundant, conflicting, or only exist to patch complexity created by another weak choice.
- Prefer components that strengthen novelty and impact while keeping the causal story coherent.
- The final fused idea must read like one method with one clear causal chain.
"""


FUSION_REPAIR_PROMPT = """
You are repairing a fused research idea after referee evaluation.

== Topic == 
{topic}

== Fixed root domains == 
{root_domains}

== Current fused idea (JSON) == 
{current_idea_json}

== Current evaluation (JSON) == 
{current_evaluation_json}

== Source mode candidates (JSON) ==
{candidate_ideas_json}

== Available atomic edit operations == 
{atomic_op_reference}

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

== Repair rules == 
- Use ONLY these operations: REMOVE_COMPONENT, REPLACE_COMPONENT, REWIRE.
- Do NOT use ADD_COMPONENT, GATE_COMPONENT, or ADD_PROTOCOL.
- Do NOT increase the number of structural components.
- Prefer one small local repair, not a rewrite of the whole idea.
- At least one structural edit must be present unless you choose to stop.
- Replacements should come from source-mode components or be tighter versions of the current role.
- If no likely-improving local repair exists, set stop=true.
- Treat component names as STRICT symbols, not paraphrases.
- When you mention an existing component from the current fused idea, copy its name exactly, character-for-character.
- When you mention a component from a source-mode candidate, copy its name exactly, character-for-character.
- Do NOT rename, summarize, translate, normalize, or slightly rewrite component names.
- If you are not confident that an edit can be expressed with exact component names, set stop=true.

== Field semantics ==
- REMOVE_COMPONENT:
  - "component" = the existing component to delete from the current fused idea.
  - "target" should be "".
- REPLACE_COMPONENT:
  - "target" = the existing component in the current fused idea that will be replaced.
  - "component" = the replacement component name.
  - The replacement component may be copied from a source-mode candidate or be a tighter variant, but must still be written as one concrete component name.
- REWIRE:
  - "component" = the source or upstream component whose connection is being changed.
  - "target" = the destination or downstream component / interface it should connect to.
  - REWIRE changes topology only; it does not add a new component.

== Valid examples ==
- Valid REPLACE_COMPONENT:
  {{
    "op": "REPLACE_COMPONENT",
    "component": "retrieval-grounded verifier",
    "target": "static verifier",
    "condition": "",
    "details": "Replace static verifier with retrieval-grounded verifier",
    "reason": "Improves mechanism grounding without expanding the architecture"
  }}
- Valid REMOVE_COMPONENT:
  {{
    "op": "REMOVE_COMPONENT",
    "component": "redundant audit head",
    "target": "",
    "condition": "",
    "details": "Remove redundant audit head",
    "reason": "Reduces complexity caused by overlapping functionality"
  }}
- Valid REWIRE:
  {{
    "op": "REWIRE",
    "component": "uncertainty estimator",
    "target": "controller",
    "condition": "",
    "details": "Rewire uncertainty estimator -> controller",
    "reason": "Lets the controller react directly to uncertainty"
  }}

== Invalid patterns ==
- Invalid: using ADD_COMPONENT, GATE_COMPONENT, or ADD_PROTOCOL.
- Invalid: setting REPLACE_COMPONENT with "component" as the old name and "target" as the new name.
- Invalid: inventing a near-match such as "dynamic uncertainty-aware controller" when the listed component is "uncertainty-aware controller".
- Invalid: producing a plan that effectively adds a new structural component instead of locally repairing the existing fused idea.
"""
