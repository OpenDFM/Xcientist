MCTS_IDEA_EVALUATION_PROMPT = """
You score research ideas encountered during a memory-guided MCTS search.
Topic: {topic}
Fixed root domains for this MCTS run: {root_domains}
Mature idea (optional alignment anchor):
{mature_idea}
Latest analysis + critiques:
{analysis}

Current run idea pool snapshot:
{idea_pool_context}

Relevant literature evidence from the current paper cache:
{paper_context}

Compiled component-level edit plan for this node:
{edit_plan}

Candidate idea:
{idea}

Rewrite path:
{path_summary}

Defect registry:
{defect_registry}

Component-level symbolic memory insights (from past MCTS iterations):
{symbolic_memory_hints}

Scoring policy:
- Prefer concrete mechanism-level edits over vague incremental changes.
- Treat evaluator, contract, audit, or gate additions as auxiliary scaffolding unless they clearly enable or verify a distinct core mechanism change in the generator, objective, representation, planner, or data path.
- Reward validation only when it is the lightest protocol set needed to falsify the core mechanism. Do not reward protocol bulk by itself.
- Penalize evaluator-first proposals whose main contribution is auditing, gating, or benchmarking rather than changing the task-solving mechanism.
- Penalize plans that add gates merely to manage extra machinery they just introduced. A gate can improve feasibility, but it should not by itself raise novelty or impact.
- Penalize feature dumping and unsupported complexity jumps.
- If the idea drifts from topic constraints or the fixed root domains above, reduce alignment_score.
- If the proposal mostly improves diagnosis, measurement, or guardrails without changing the task-solving path, clarity may improve, but novelty and impact should stay limited.
- When symbolic memory hints are available, use them to calibrate scores:
  * Positive-delta records for the same component family suggest the approach is promising — credit novelty/impact.
  * Negative-delta records or listed anti-patterns signal known failure modes — increase risk/complexity_penalty accordingly.
  * If the edit plan contradicts a high-confidence anti-pattern, note it explicitly in feedback.
- In "detected_defects", list ALL defect tags from the registry above that still apply to this idea AFTER the proposed edit. Choose only from the canonical tags.

Scoring rubric (use this exact rubric for consistency):
- Use the full scale. Avoid defaulting to 3 unless the evidence is genuinely mixed.
- All 0-5 metric scores must be integers only. Do not return decimals such as 3.5 or 4.2.
- Positive metrics where HIGHER is better: novelty, surprise, feasibility, clarity, impact, conciseness, alignment_score, protocol_score.
- Penalty metrics where HIGHER is worse: risk, complexity_penalty.
- Score each metric independently. Do not let a strong score in one metric automatically raise another.
- Distinguish novelty vs surprise vs impact:
  * novelty = how different the mechanism is from the current baseline/path and cited evidence.
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
  * Evaluator-only, contract-only, or gate-only changes without a substantive mechanism change should usually score <= 2.
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
  * 0 = clearly drifts from the topic, path, or mature-idea constraints.
  * 3 = generally relevant, but some elements feel weakly connected or partially off-target.
  * 5 = directly and tightly aligned with the topic constraints and the intended search direction.
- complexity_penalty (0-5; higher is more excessive)
  * 0 = no meaningful excess complexity beyond what the objective requires.
  * 3 = some added moving parts, coordination cost, or budget pressure, but still arguable.
  * 5 = avoidable architectural sprawl, weakly justified extra modules, or major evaluation burden.
  * Extra evaluators, auditors, contract layers, or gates count as overhead unless they are strictly necessary to support a clearly identified core mechanism.
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
  "feedback": "Actionable critique referencing defects, skill choice, and component edits",
  "defect_fix_summary": "Which defect was addressed and why the selected skill helps",
  "detected_defects": ["canonical_defect_tag_1", "canonical_defect_tag_2"],
  "lift_estimate": 0-100
}}
"""
