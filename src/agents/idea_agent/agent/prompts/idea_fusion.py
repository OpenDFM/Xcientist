IDEA_FUSION_PROMPT = """
You are a component-level fusion agent for LigAgent-Pro.

== Topic == 
{topic}

== Current mature idea ==
{mature_idea}

== Refinement scope (optional) ==
{refinement_scope}

== Shared root domains ==
{root_domains}

== Analysis summary ==
{analysis}

You are given {mode_count} candidate ideas produced from different idea taste modes, all starting from the same prepared root context.

== Candidate ideas (compact JSON) == 
{candidate_ideas_json}

Return STRICT JSON only:
{{
  "fused_idea": {{
    "title": "string",
    "abstract": "string",
    "core_contribution": "string",
    "method": "string",
    "risks": "string",
    "tags": ["string"],
    "operator": "fusion_agent",
    "target_defects": ["string"],
    "rationale": "string",
    "components": ["string"],
    "component_explanations": {{
      "component_name": "string"
    }},
    "root_domains": ["string"],
    "paper_graph_context": "string",
    "edit_plan": null,
    "skill_metrics": {{}}
  }},
  "fusion_metadata": {{
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
    "minimal_validation_plan": "string"
  }}
}}

== Fusion rules ==
- The top-level JSON value MUST be an object with exactly two keys: `fused_idea` and `fusion_metadata`.
- `fused_idea` MUST appear first.
- Build `fused_idea` first, then fill `fusion_metadata`.
- Do NOT return a raw array.
- Do NOT return a single selected component object as the top-level JSON.
- Do NOT union all ideas together.
- Select exactly one dominant core mechanism.
- Other kept components must be support modules, not a second competing core mechanism.
- When you reuse a component, copy its name exactly from `components`.
- Protocol, guardrail, evaluator, or audit components cannot be the main novelty.
- Gate/router/controller/threshold-style components should almost never be the dominant core mechanism. Prefer the underlying task-solving mechanism, representation change, update rule, or transferred principle instead.
- If the current mature idea is not centered on gating or routing, preserve that character in fusion. Do not elevate a gate/router wrapper into the fused thesis just because it is easy to combine.
- If a gate/router/controller/threshold component mainly patches complexity created by another weak choice, reject it instead of carrying that patch into the fused idea.
- If refinement_scope is provided, keep the fused novelty inside that scope. Reject candidates that look strong only because they move the contribution to a different subsystem.
- Remove components that are redundant, conflicting, or only exist to patch complexity created by another weak choice.
- Prefer components that strengthen novelty and impact while keeping the causal story coherent.
- The final fused idea must read like one method with one clear causal chain.
"""


FUSION_REPAIR_INSTANTIATION_PROMPT = """
You are materializing a local repair plan into a revised fused research idea.

== Context ==
Topic: {topic}
Fixed root domains: {root_domains}
Refinement scope: {refinement_scope}
Current fused idea: {parent_summary}
Current fused components: {parent_components}
Literature context: {paper_context}
Memory bundle: {memory_bundle}
{additional_retrieval_context}

== Local Repair Plan ==
Objective: {plan_objective}
Target defects: {target_defects}
Component edits:
{component_edits}
Validation protocols:
{validation_protocols}
Guardrails: {guardrails}

== Your Task ==
1. Keep the same overall thesis as the current fused idea.
2. Realize only the local repair encoded by the component edits above.
3. Treat component names in the repair plan as exact symbols. Do not rename or paraphrase them.
4. Do not introduce extra structural components beyond what is already in the fused idea plus the explicit repair edits.
5. Write a concrete title, abstract, core contribution, method, risks, and rationale for the repaired idea.
6. Avoid centering the repaired idea on a gate/router/controller/threshold component unless the current fused idea already depends on that mechanism as its core thesis.
7. If `refinement_scope` is provided and not equal to `None`, keep the repaired novelty inside that scope.
8. Provide a `component_mapping` only for names that literally appear in the repair plan. Since these names are already concrete, use identity mappings such as `"memory_updater": "memory_updater"`.
9. Provide `component_role_explanations` only for concrete names that appear in `component_mapping`.

Return STRICT JSON (no Markdown wrapping):
{{
  "title": "concise repaired idea title",
  "abstract": "≤150 words abstract describing the repaired fused idea",
  "core_contribution": "one focused statement of the repaired mechanism",
  "method": "concrete methodology for the repaired fused idea using the exact component names from the repair plan",
  "risks": "concrete failure modes and mitigation strategies",
  "rationale": "2-3 sentences on how the local repair improves the fused idea",
  "component_mapping": {{
      "exact_component_name_1": "exact_component_name_1",
      "exact_component_name_2": "exact_component_name_2"
    }},
  "component_role_explanations": {{
      "exact_component_name_1": "Specific computational role in the repaired idea...",
      "exact_component_name_2": "Specific computational role in the repaired idea..."
    }},
  "edit_reasons": [
      "Reason 1: Why this local edit improves the fused idea...",
      "Reason 2: Why this local edit resolves the target defect..."
    ]
}}
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
- Do NOT use ADD_COMPONENT or ADD_PROTOCOL.
- Do NOT increase the number of structural components.
- Prefer one small local repair, not a rewrite of the whole idea.
- At least one structural edit must be present unless you choose to stop.
- Replacements should come from source-mode components or be tighter versions of the current role.
- Prefer removing or replacing gate/router/controller/threshold-style components when they are auxiliary scaffolding rather than the true core mechanism.
- If a gate/router/controller/threshold component is only patching complexity created elsewhere, treat that as evidence to simplify the architecture rather than preserve the patch.
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
    "target": "memory_updater",
    "condition": "",
    "details": "Rewire uncertainty estimator -> memory_updater",
    "reason": "Lets the update rule react directly to uncertainty"
  }}

== Invalid patterns ==
- Invalid: using ADD_COMPONENT or ADD_PROTOCOL.
- Invalid: setting REPLACE_COMPONENT with "component" as the old name and "target" as the new name.
- Invalid: inventing a near-match such as "dynamic uncertainty-aware updater" when the listed component is "uncertainty-aware updater".
- Invalid: producing a plan that effectively adds a new structural component instead of locally repairing the existing fused idea.
- Invalid: preserving a gate/router/controller/threshold component as the repaired core mechanism when it only acts as auxiliary scaffolding.
"""
