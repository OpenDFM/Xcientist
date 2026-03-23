from __future__ import annotations

from typing import Optional

from src.agents.idea_agent.agent.prompts.prompt_modes import (
    is_conceptual_surprise_mode,
)


SKILL_INSTANTIATION_PROMPT = """
You are an expert research scientist instantiating a structured skill-based edit plan into a concrete research idea.

Given a compiled edit plan (skill + atomic component edits + validation protocols), your job is to fill in the **concrete, topic-specific content** for each field.

== Context ==
Topic: {topic}
Fixed root domains for this MCTS run: {root_domains}
Refinement scope: {refinement_scope}
Taste guidance (soft preference only): {taste_guidance}
Mature idea (ANCHOR): {mature_idea}
Parent idea: {parent_summary}
Parent components (actual modules in the current idea): {parent_components}
Literature context: {paper_context}
Memory bundle: {memory_bundle}
Skill-specific mechanism references: {skill_references}
{additional_retrieval_context}

== Compiled Edit Plan **IMPORTANT** ==
Skill: {skill_name}
Objective: {plan_objective}
Target defects: {target_defects}
Component edits (atomic blueprint, already compiled — component names are GENERIC TEMPLATES that you must map to concrete names):
{component_edits}
Validation protocols:
{validation_protocols}
Guardrails: {guardrails}

== Your Task ==
Instantiate the above plan into a concrete research idea. You must:
1. Replace every generic placeholder (e.g. "core_mechanism_module", "backbone_model", "constraint_penalty_module") with a **specific, topic-relevant** module/method name.
2. Write a concrete title, abstract, core contribution, methodology, risks, and rationale — as if drafting a real paper.
3. Keep consistency with the edit plan structure (same number of component edits, same operator types).
4. Reference the defects you are fixing and explain how the instantiated mechanism addresses them.
5. Ensure the methodology is specific enough that someone could implement it (name architectures, loss terms, optimization steps, etc.).
6. If a mature idea (anchor) is provided, the instantiated idea MUST build upon, refine, or extend that mature idea — do NOT ignore it or propose an unrelated direction.
7. Provide a "component_mapping" that maps EVERY generic template component name in the edit plan to a concrete, topic-specific name. For REWIRE/REPLACE targets that refer to existing parent components, map them to the actual parent component name. For new components (ADD_COMPONENT), give them a specific name reflecting their role in this idea.
8. Keep "component_mapping" MINIMAL and bounded. It may contain ONLY generic names that literally appear in the compiled edit plan above as a component or target. Do NOT add auxiliary modules, losses, datasets, stores, encoders, optimizers, helper blocks, or other extra names unless they literally appear in that compiled edit plan.
9. Provide "edit_reasons": a JSON list of short reason strings (one per component edit in the same order as the component edits above). Each reason should explain **why** this specific atomic operation is needed to address the target defects — e.g., what gap it fills, what failure mode it prevents, or what capability it adds.
10. Provide "component_role_explanations": a JSON object that explains the role of each concrete component name that appears in "component_mapping". The explanation should describe what the component does inside the idea, not just repeat the name.
11. The instantiated idea MUST stay in the fixed root domain(s) above.
12. If additional retrieved core references are provided, use them to ground the mechanism choices. If any of them are cross-domain, extract only the transferable mechanism or invariant that helps the current idea. Do not copy paper-specific content verbatim.
13. Treat the taste guidance above as a soft preference only. Reflect it when possible, but it MUST NOT override the compiled edit plan, target defects, validation protocols, or guardrails.
14. If skill-specific mechanism references are provided, use them as compact mechanism patterns and failure-mode checks. Reuse the pattern, not the literal names, unless the names already fit the compiled edit plan.
15. If the mature idea or parent idea is training-free or inference-time only, preserve that character when possible. Do NOT introduce a new training stage, trainable controller, auxiliary loss, fine-tuning loop, or learned module unless it is indispensable to the compiled edit and clearly more important than keeping the method training-free.
16. If you do introduce new training into an originally training-free idea, explicitly justify why a training-free realization is insufficient and why the training shift is central rather than optional.
17. If `skill_name` is `mechanism-commit-innovation` and the parent idea is not already centered on threshold/control logic, do NOT realize the edit as thresholding, gating, suppression, or quota adjustment. Prefer a direct mechanism, representation change, memory update rule, or objective-local repair intrinsic to the parent idea.
18. If `refinement_scope` is provided and not equal to `None`, treat it as a hard boundary on where the novelty may appear. Keep the instantiated contribution inside that scope and do not relocate the main change to another subsystem.

Return STRICT JSON (no Markdown wrapping):
{{
  "title": "concise, specific paper title using the concrete component names",
  "abstract": "≤150 words abstract describing the concrete contribution",
  "core_contribution": "one focused statement of the new insight/mechanism",
  "method": "concrete methodology steps using the concrete names defined for the compiled edit-plan placeholders in your component_mapping. You may mention standard losses, optimizers, datasets, encoders, or helper routines in prose without adding them to component_mapping unless they literally appear in the compiled edit plan.",
  "risks": "concrete failure modes and mitigation strategies",
  "rationale": "2-3 sentences on how this skill application resolves the target defects. If you introduced new training into a training-free parent idea, explicitly justify why that shift is necessary.",
  "component_mapping": {{
      "generic_template_name_1": "concrete_topic_specific_name_1",
      "generic_template_name_2": "concrete_topic_specific_name_2"
    }},
  "component_role_explanations": {{
      "concrete_topic_specific_name_1": "Specific computational role in the new method...",
      "concrete_topic_specific_name_2": "Specific computational role in the new method..."
    }},
  "edit_reasons": [
      "Reason 1: Why generic_template_1 fixes Target Defect X...",
      "Reason 2: Why generic_template_2 is needed for the validation..."
    ]
}}
"""


CONCEPTUAL_SURPRISE_SKILL_INSTANTIATION_PROMPT = SKILL_INSTANTIATION_PROMPT.replace(
    "Return STRICT JSON (no Markdown wrapping):",
    """19. Treat the main contribution as a local conceptual repair of the parent idea, not merely a new module insertion. First identify the weak assumption, framing, or principle in the parent idea; then use the compiled edits as the implementation vehicle for that repair.
20. Keep the improvement small-version and thesis-preserving. Do NOT replace the parent idea with a different research paradigm.
21. In `abstract`, lead with the scientific thesis or conceptual repair first, then explain the concrete mechanism that realizes it.
22. In `core_contribution`, express the new insight as a principle, invariant, assumption repair, or reframing. A module name by itself is not a sufficient contribution.
23. In `method`, explicitly separate the conceptual move from the mechanism realization, while still giving concrete implementation details.

Return STRICT JSON (no Markdown wrapping):""",
).replace(
    '"abstract": "≤150 words abstract describing the concrete contribution",',
    '"abstract": "≤150 words abstract describing the concrete contribution. The opening sentence should state the scientific thesis or conceptual repair; later sentences can explain the mechanism vehicle.",',
).replace(
    '"core_contribution": "one focused statement of the new insight/mechanism",',
    '"core_contribution": "one focused statement of the thesis, principle, invariant, or conceptual repair being introduced; not just a module name",',
).replace(
    '"method": "concrete methodology steps using the concrete names defined for the compiled edit-plan placeholders in your component_mapping. You may mention standard losses, optimizers, datasets, encoders, or helper routines in prose without adding them to component_mapping unless they literally appear in the compiled edit plan.",',
    '"method": "start by naming the conceptual move being realized, then give concrete methodology steps using the concrete names defined for the compiled edit-plan placeholders in your component_mapping. You may mention standard losses, optimizers, datasets, encoders, or helper routines in prose without adding them to component_mapping unless they literally appear in the compiled edit plan.",',
).replace(
    '"rationale": "2-3 sentences on how this skill application resolves the target defects",',
    '"rationale": "2-3 sentences on how this skill application resolves the target defects and sharpens the parent idea\'s thesis",',
)


def get_skill_instantiation_prompt(mode: Optional[str] = None) -> str:
    if is_conceptual_surprise_mode(mode):
        return CONCEPTUAL_SURPRISE_SKILL_INSTANTIATION_PROMPT
    return SKILL_INSTANTIATION_PROMPT
