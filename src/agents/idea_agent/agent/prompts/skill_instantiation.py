from __future__ import annotations

from typing import Optional

from src.agents.idea_agent.agent.prompts.prompt_modes import (
    is_conceptual_surprise_mode,
)


_BASE_HARD_TASKS = (
    'Replace every generic placeholder in the compiled edit plan with a specific, topic-relevant module or method name.',
    'Write a concrete title, abstract, core contribution, method, risks, and rationale as if drafting a real paper.',
    'Keep the instantiated idea consistent with the compiled edit plan structure: preserve the same component-edit count and the same operator types.',
    'Make the method concrete enough that someone could implement it, and make the causal link from the instantiated mechanism to the target defects explicit.',
    'If a mature idea is provided, treat it as the anchor: refine it directly, keep the title/abstract/method framed as a refinement of that mature idea, and do not drift into an unrelated direction or temporary internal search alias.',
    'Provide a "component_mapping" for every generic template name that literally appears in the compiled edit plan. For REWIRE/REPLACE targets that refer to existing parent components, map them to the actual parent component name. For ADD_COMPONENT, give a concrete name that reflects its role in this idea.',
    'Keep "component_mapping" minimal and bounded to names that literally appear in the compiled edit plan. Also provide aligned "edit_reasons" and "component_role_explanations" for those mapped components.',
    'Stay inside the fixed root domain(s) and refinement scope above. If `skill_name` is `mechanism-commit-innovation` and the parent idea is not already centered on threshold/control logic, do not realize the edit as thresholding, gating, suppression, or quota adjustment.',
)

_BASE_HEURISTICS = (
    'Use additional retrieved references and skill-specific mechanism references only to ground mechanism choices or failure-mode checks. If a reference is cross-domain, transfer only the useful mechanism or invariant, not the paper-specific packaging or names.',
    'Treat the taste guidance above as a soft preference only. It must not override the compiled edit plan, target defects, validation protocols, or guardrails.',
    'If the mature idea or parent idea is training-free or inference-time only, preserve that character when possible. If you introduce new training, explicitly justify why the training shift is necessary and central.',
    'Prefer direct mechanism, representation, objective-local, or update-rule repairs over audit, guardrail, controller, or wrapper-heavy realizations. Validation and support modules should stay secondary to the task-solving path.',
)

_CONCEPTUAL_SURPRISE_HEURISTICS = (
    'Use additional retrieved references and skill-specific mechanism references only to ground mechanism choices or failure-mode checks. If a reference is cross-domain, transfer only the useful mechanism or invariant, not the paper-specific packaging or names.',
    'Treat the taste guidance above as a soft preference only. It must not override the compiled edit plan, target defects, validation protocols, or guardrails.',
    'If the mature idea or parent idea is training-free or inference-time only, preserve that character when possible. If you introduce new training, explicitly justify why the training shift is necessary and central.',
    'Treat the contribution as a local conceptual repair of the parent idea, not merely a new module insertion. Keep it thesis-preserving; in `abstract`, lead with the repaired thesis; in `core_contribution`, state the principle or invariant; in `method`, separate the conceptual move from the concrete mechanism realization.',
)


def _numbered_lines(lines: tuple[str, ...]) -> str:
    return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))


def _build_output_schema(*, conceptual_surprise: bool) -> str:
    abstract_desc = "≤150 words abstract describing the concrete contribution"
    core_contribution_desc = "one focused statement of the new insight/mechanism"
    method_desc = (
        "concrete methodology steps using the concrete names defined for the compiled edit-plan placeholders "
        "in your component_mapping. You may mention standard losses, optimizers, datasets, encoders, or helper "
        "routines in prose without adding them to component_mapping unless they literally appear in the compiled edit plan."
    )
    rationale_desc = (
        "2-3 sentences on how this skill application resolves the target defects. "
        "If you introduced new training into a training-free parent idea, explicitly justify why that shift is necessary."
    )

    if conceptual_surprise:
        abstract_desc += ". The opening sentence should state the scientific thesis or conceptual repair; later sentences can explain the mechanism vehicle."
        core_contribution_desc = (
            "one focused statement of the thesis, principle, invariant, or conceptual repair being introduced; "
            "not just a module name"
        )
        method_desc = (
            "start by naming the conceptual move being realized, then give concrete methodology steps using the "
            "concrete names defined for the compiled edit-plan placeholders in your component_mapping. You may "
            "mention standard losses, optimizers, datasets, encoders, or helper routines in prose without adding "
            "them to component_mapping unless they literally appear in the compiled edit plan."
        )
        rationale_desc = (
            "2-3 sentences on how this skill application resolves the target defects and sharpens the parent idea's thesis. "
            "If you introduced new training into a training-free parent idea, explicitly justify why that shift is necessary."
        )

    return "\n".join(
        [
            "Return STRICT JSON (no Markdown wrapping):",
            "{{",
            '  "title": "concise, specific paper title using the concrete component names",',
            f'  "abstract": "{abstract_desc}",',
            f'  "core_contribution": "{core_contribution_desc}",',
            f'  "method": "{method_desc}",',
            '  "risks": "concrete failure modes and mitigation strategies",',
            f'  "rationale": "{rationale_desc}",',
            '  "component_mapping": {{',
            '      "generic_template_name_1": "concrete_topic_specific_name_1",',
            '      "generic_template_name_2": "concrete_topic_specific_name_2"',
            "    }},",
            '  "component_role_explanations": {{',
            '      "concrete_topic_specific_name_1": "Specific computational role in the new method...",',
            '      "concrete_topic_specific_name_2": "Specific computational role in the new method..."',
            "    }},",
            '  "edit_reasons": [',
            '      "Reason 1: Why generic_template_1 fixes Target Defect X...",',
            '      "Reason 2: Why generic_template_2 is needed for the validation..."',
            "    ]",
            "}}",
        ]
    )


def _build_skill_instantiation_prompt(*, conceptual_surprise: bool) -> str:
    heuristics = (
        _CONCEPTUAL_SURPRISE_HEURISTICS
        if conceptual_surprise
        else _BASE_HEURISTICS
    )
    return "\n".join(
        [
            "You are an expert research scientist instantiating a structured skill-based edit plan into a concrete research idea.",
            "",
            "Given a compiled edit plan (skill + atomic component edits + validation protocols), your job is to fill in the concrete, topic-specific content for each field.",
            "",
            "== Context ==",
            "Topic: {topic}",
            "Fixed root domains for this MCTS run: {root_domains}",
            "Refinement scope: {refinement_scope}",
            "Taste guidance (soft preference only): {taste_guidance}",
            "Mature idea (ANCHOR): {mature_idea}",
            "Parent idea: {parent_summary}",
            "Parent components (actual modules in the current idea): {parent_components}",
            "Literature context: {paper_context}",
            "Memory bundle: {memory_bundle}",
            "Skill-specific mechanism references: {skill_references}",
            "{additional_retrieval_context}",
            "",
            "== Compiled Edit Plan ==",
            "Skill: {skill_name}",
            "Objective: {plan_objective}",
            "Target defects: {target_defects}",
            "Component edits (atomic blueprint, already compiled; component names are generic templates that you must map to concrete names):",
            "{component_edits}",
            "Validation protocols:",
            "{validation_protocols}",
            "Guardrails: {guardrails}",
            "",
            "== Hard Constraints ==",
            "Satisfy all of the following:",
            _numbered_lines(_BASE_HARD_TASKS),
            "",
            "== Heuristics ==",
            "When multiple outputs satisfy the hard constraints, prefer the following:",
            _numbered_lines(heuristics),
            "",
            _build_output_schema(conceptual_surprise=conceptual_surprise),
            "",
        ]
    )


SKILL_INSTANTIATION_PROMPT = _build_skill_instantiation_prompt(
    conceptual_surprise=False
)

CONCEPTUAL_SURPRISE_SKILL_INSTANTIATION_PROMPT = _build_skill_instantiation_prompt(
    conceptual_surprise=True
)


def get_skill_instantiation_prompt(mode: Optional[str] = None) -> str:
    if is_conceptual_surprise_mode(mode):
        return CONCEPTUAL_SURPRISE_SKILL_INSTANTIATION_PROMPT
    return SKILL_INSTANTIATION_PROMPT
