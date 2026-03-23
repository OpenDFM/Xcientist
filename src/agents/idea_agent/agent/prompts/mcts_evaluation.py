from __future__ import annotations

from typing import Optional

from src.agents.idea_agent.agent.prompts.prompt_modes import (
    is_conceptual_surprise_mode,
)


MCTS_IDEA_EVALUATION_PROMPT = """
You score research ideas encountered during a memory-guided MCTS search.

== Topic == 
{topic}

== Fixed root domains for this MCTS run == 
{root_domains}

== Mature idea (optional alignment anchor) == 
{mature_idea}

== Refinement scope (optional alignment boundary) ==
{refinement_scope}

== Compiled component-level edit plan for this node ==:
{edit_plan}

== Candidate idea ==:
{idea}

== Defect registry ==:
{defect_registry}

== Component-level symbolic memory insights (from component ablations) ==:
{symbolic_memory_hints}

Scoring policy:
- Prefer concrete mechanism-level edits over vague incremental changes.
- Treat evaluator, contract, audit, or auxiliary control-wrapper additions (for example gates, routers, controllers, or threshold policies) as scaffolding unless they clearly enable or verify a distinct core mechanism change in the generator, objective, representation, planner, or data path.
- Reward validation only when it is the lightest protocol set needed to falsify the core mechanism. Do not reward protocol bulk by itself.
- Penalize evaluator-first proposals whose main contribution is auditing, auxiliary control wrapping, or benchmarking rather than changing the task-solving mechanism.
- Penalize plans that add an auxiliary control wrapper merely to manage extra machinery they just introduced. Such a wrapper can improve feasibility, but it should not by itself raise novelty or impact.
- Penalize feature dumping and unsupported complexity jumps.
- If the idea drifts from topic constraints or the fixed root domains above, reduce alignment_score.
- If refinement_scope is provided and the candidate moves the novelty outside that allowed edit surface, reduce alignment_score and raise complexity_penalty.
- If the mature idea above is training-free or inference-time only, penalize candidates that add new training stages, learned controllers, auxiliary losses, or fine-tuning loops without a strong mechanism-level justification. Such drift should usually reduce alignment_score and increase complexity_penalty.
- If the proposal mostly improves diagnosis, measurement, or guardrails without changing the task-solving path, clarity may improve, but novelty and impact should stay limited.
- In `feedback`, `failure_modes`, and `defect_fix_summary`, describe concerns in neutral mechanism terms. Do not default to recommending threshold/gate/budget-style fixes unless the candidate itself already centers that logic.
- Do not speculate about compute/resource budgeting unless the candidate explicitly makes resource management part of its core mechanism.
- When symbolic memory hints are available, interpret them as component-ablation evidence:
  * Positive result means removing that component helped; treat that component family as risky, redundant, or harmful in similar designs.
  * Negative result means removing that component hurt; treat that component family as beneficial or structurally important.
  * Inconclusive results are weak evidence and should not dominate scoring.
  * If the candidate repeats a component family that ablation evidence says is harmful, note it explicitly in feedback.
- In "detected_defects", list ALL defect tags from the registry above that still apply to this idea AFTER the proposed edit. Choose only from the canonical tags.

Scoring rubric (use this exact rubric for consistency):
- Use the full scale. Avoid defaulting to 3 unless the evidence is genuinely mixed.
- All 0-5 metric scores must be integers only. Do not return decimals such as 3.5 or 4.2.
- Positive metrics where HIGHER is better: novelty, surprise, feasibility, clarity, impact, conciseness, alignment_score, protocol_score.
- Penalty metrics where HIGHER is worse: risk, complexity_penalty.
- Score each metric independently. Do not let a strong score in one metric automatically raise another.
- Distinguish novelty vs surprise vs impact:
  * novelty = how different the mechanism is from the current baseline and cited evidence.
  * surprise = how non-obvious the idea feels to a strong researcher before seeing it, even after accounting for the topic and available evidence.
  * impact = how much the idea could move the core problem if it works.
  * A proposal can be novel but not very surprising if it is an obvious next step that simply has not been tried yet.
  * A proposal can be surprising without being high quality if it feels unexpected but weakly justified; do not reward incoherent weirdness.
- Distinguish feasibility vs risk:
  * feasibility = can this be implemented and tested with the stated assumptions/resources.
  * risk = how likely/severe the remaining failure modes are even if implementation is possible.
- Distinguish conciseness vs complexity_penalty:
  * conciseness = how focused and non-bloated the proposal description and mechanism are.
  * complexity_penalty = how much avoidable system/evaluation burden the proposal introduces.

Metric-specific anchors:
- novelty (0-5; higher is more original)
  * 0 = trivial restatement, cosmetic rewrite, or no real mechanism change.
  * 3 = meaningful recombination or scoped mechanism change, but still close to known patterns.
  * 5 = clear mechanism-level departure or unusually strong recombination with concrete rationale.
  * Evaluator-only, contract-only, or auxiliary-control-wrapper-only changes without a substantive mechanism change should usually score <= 2.
- surprise (0-5; higher is more unexpectedly insightful)
  * 0 = the idea feels like an obvious next step once the topic and baseline are known.
  * 3 = the idea contains at least one non-obvious move, reframing, or recombination that many researchers would not immediately propose.
  * 5 = the idea is genuinely "I would not have thought of that" surprising while remaining coherent, mechanism-grounded, and relevant.
  * Do not reward randomness, vagueness, or incoherent novelty theater. Surprise requires justified unexpectedness.
- feasibility (0-5; higher is more practical)
  * 0 = underspecified, implausible, or depends on missing capabilities/data.
  * 3 = implementable with reasonable effort, but has notable execution assumptions.
  * 5 = concrete, technically coherent, and straightforward to implement/evaluate from the given plan.
- clarity (0-5; higher is clearer)
  * 0 = vague objective, unclear mechanism, or missing causal story.
  * 3 = mostly understandable, but some steps/roles/claims remain ambiguous.
  * 5 = precise objective, explicit component roles, and clear causal chain from edit to outcome.
- impact (0-5; higher is more valuable)
  * 0 = addresses a peripheral issue or offers negligible upside.
  * 3 = could improve an important sub-problem, but expected gains are bounded/local.
  * 5 = directly targets a central bottleneck and could materially improve performance, reliability, or insight.
  * Audit or evaluator layers that mostly improve diagnosis/measurement should usually score <= 3 unless they materially change the system's learning or decision behavior.
- risk (0-5; higher is riskier)
  * 0 = major failure modes are already controlled or low consequence.
  * 3 = meaningful risks remain, but they are identifiable and partially mitigated.
  * 5 = serious unresolved failure modes, fragile assumptions, or high likelihood of invalid conclusions.
- conciseness (0-5; higher is more focused)
  * 0 = bloated, diffuse, feature-dumped, or padded with unnecessary machinery.
  * 3 = moderately focused, but includes some extra scope or wording that is not essential.
  * 5 = tightly scoped, minimal mechanism surface area, and no obvious unnecessary additions.
- alignment_score (0-5; higher is more aligned)
  * 0 = clearly drifts from the topic, root-domain, or mature-idea constraints.
  * 3 = generally relevant, but some elements feel weakly connected or partially off-target.
  * 5 = directly and tightly aligned with the topic constraints and the intended search direction.
- complexity_penalty (0-5; higher is more excessive)
  * 0 = no meaningful excess complexity beyond what the objective requires.
  * 3 = some added moving parts or coordination cost, but still arguable.
  * 5 = avoidable architectural sprawl, weakly justified extra modules, or major evaluation burden.
  * Extra evaluators, auditors, contract layers, or auxiliary control wrappers count as overhead unless they are strictly necessary to support a clearly identified core mechanism.
- protocol_score (0-5; higher is more rigorous)
  * 0 = no meaningful validation plan, or only vague evaluation claims.
  * 3 = includes at least one solid validation axis, but coverage is incomplete.
  * 5 = explicit regression, ablation, and stress tests with clear fairness/control expectations.
  * protocol_score reflects validation quality only. It must not compensate for weak novelty, weak impact, or evaluator-only ideas.

Return STRICT JSON (no prose):
{{
  "novelty": 0-5,
  "surprise": 0-5,
  "feasibility": 0-5,
  "clarity": 0-5,
  "impact": 0-5,
  "risk": 0-5,
  "conciseness": 0-5,
  "alignment_score": 0-5,
  "complexity_penalty": 0-5,
  "protocol_score": 0-5,
  "confidence": 0-1,
  "failure_modes": ["at least one concrete failure mode"],
  "fairness_protocol": "How fairness/control experiments are enforced or what is missing",
  "feedback": "Actionable critique in neutral mechanism terms, referencing defects, skill choice, and component edits",
  "defect_fix_summary": "Which defect was addressed and why the selected skill helps, written in neutral mechanism terms",
  "detected_defects": ["canonical_defect_tag_1", "canonical_defect_tag_2"]
}}
"""


CONCEPTUAL_SURPRISE_MCTS_EVALUATION_PROMPT = MCTS_IDEA_EVALUATION_PROMPT.replace(
    "- Distinguish novelty vs surprise vs impact:\n"
    "  * novelty = how different the mechanism is from the current baseline and cited evidence.\n"
    "  * surprise = how non-obvious the idea feels to a strong researcher before seeing it, even after accounting for the topic and available evidence.\n"
    "  * impact = how much the idea could move the core problem if it works.\n"
    "  * A proposal can be novel but not very surprising if it is an obvious next step that simply has not been tried yet.\n"
    "  * A proposal can be surprising without being high quality if it feels unexpected but weakly justified; do not reward incoherent weirdness.\n",
    """- Distinguish novelty vs surprise vs impact:
  * novelty = how different the mechanism is from the current baseline and cited evidence.
  * surprise = how non-obvious the central thesis feels to a strong researcher before seeing it, especially when the idea repairs a weak assumption, sharpens a principle, or reframes the problem while staying coherent and relevant.
  * impact = how much the idea could move the core problem if it works.
  * A proposal can be novel but not very surprising if it is an obvious next step that simply has not been tried yet.
  * A proposal can be moderately novel yet highly surprising if the main value comes from a strong thesis-level insight rather than a large architectural departure.
  * A proposal can be surprising without being high quality if it feels unexpected but weakly justified; do not reward incoherent weirdness.
""",
).replace(
    "- novelty (0-5; higher is more original)\n"
    "  * 0 = trivial restatement, cosmetic rewrite, or no real mechanism change.\n"
    "  * 3 = meaningful recombination or scoped mechanism change, but still close to known patterns.\n"
    "  * 5 = clear mechanism-level departure or unusually strong recombination with concrete rationale.\n"
    "  * Evaluator-only, contract-only, or auxiliary-control-wrapper-only changes without a substantive mechanism change should usually score <= 2.\n",
    """- novelty (0-5; higher is more original)
  * 0 = trivial restatement, cosmetic rewrite, or no real mechanism change.
  * 3 = meaningful recombination or scoped mechanism change, but still close to known patterns.
  * 5 = clear mechanism-level departure or unusually strong recombination with concrete rationale.
  * Evaluator-only, contract-only, or auxiliary-control-wrapper-only changes without a substantive mechanism change should usually score <= 2.
  * A new thesis without any concrete mechanism support should not automatically score high on novelty.
""",
).replace(
    "- surprise (0-5; higher is more unexpectedly insightful)\n"
    "  * 0 = the idea feels like an obvious next step once the topic and baseline are known.\n"
    "  * 3 = the idea contains at least one non-obvious move, reframing, or recombination that many researchers would not immediately propose.\n"
    "  * 5 = the idea is genuinely \"I would not have thought of that\" surprising while remaining coherent, mechanism-grounded, and relevant.\n"
    "  * Do not reward randomness, vagueness, or incoherent novelty theater. Surprise requires justified unexpectedness.\n",
    """- surprise (0-5; higher is more unexpectedly insightful)
  * 0 = the idea feels like an obvious next step once the topic and baseline are known.
  * 2 = the idea is mostly a local wrapper tweak, monitor, or protocol refinement without a stronger thesis-level insight.
  * 3 = the idea contains at least one non-obvious move, reframing, assumption repair, or principled recombination that many researchers would not immediately propose.
  * 5 = the idea is genuinely "I would not have thought of that" surprising because it changes how the problem is interpreted or what principle should govern the method, while remaining coherent, mechanism-grounded, and relevant.
  * Do not reward randomness, vagueness, or incoherent novelty theater. Surprise requires justified unexpectedness.
""",
).replace(
    "Scoring policy:\n",
    """Scoring policy:
- In `conceptual_surprise` mode, treat thesis-level insight as the primary source of surprise. A child should score well on surprise when it sharpens the parent idea's scientific claim, repairs a weak assumption, or introduces a better principle for the same method axis.
- A pure implementation tweak can still be useful, but if it does not improve the scientific thesis, its surprise should usually remain limited.
""",
)


def get_mcts_evaluation_prompt(mode: Optional[str] = None) -> str:
    if is_conceptual_surprise_mode(mode):
        return CONCEPTUAL_SURPRISE_MCTS_EVALUATION_PROMPT
    return MCTS_IDEA_EVALUATION_PROMPT
