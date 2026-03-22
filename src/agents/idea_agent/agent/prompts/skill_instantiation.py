SKILL_INSTANTIATION_PROMPT = """
You are an expert research scientist instantiating a structured skill-based edit plan into a concrete research idea.

Given a compiled edit plan (skill + atomic component edits + validation protocols), your job is to fill in the **concrete, topic-specific content** for each field.

== Context ==
Topic: {topic}
Fixed root domains for this MCTS run: {root_domains}
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

Return STRICT JSON (no Markdown wrapping):
{{
  "title": "concise, specific paper title using the concrete component names",
  "abstract": "≤150 words abstract describing the concrete contribution",
  "core_contribution": "one focused statement of the new insight/mechanism",
  "method": "concrete methodology steps using the concrete names defined for the compiled edit-plan placeholders in your component_mapping. You may mention standard losses, optimizers, datasets, encoders, or helper routines in prose without adding them to component_mapping unless they literally appear in the compiled edit plan.",
  "risks": "concrete failure modes and mitigation strategies",
  "rationale": "2-3 sentences on how this skill application resolves the target defects",
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
