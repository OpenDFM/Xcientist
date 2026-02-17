MCTS_IDEA_EVALUATION_PROMPT = """
You score research ideas encountered during a memory-guided MCTS search.
Topic: {topic}
Mature idea (optional alignment anchor):
{mature_idea}
Latest analysis + critiques:
{analysis}

Relevant literature evidence from the current paper cache:
{paper_context}

Compiled component-level edit plan for this node:
{edit_plan}

Current skill prior / constraints for the chosen edit-operator skill:
{skill_prior}

Candidate idea (JSON):
{idea}

Rewrite path:
{path_summary}

{defect_registry}

Scoring policy:
- Prefer concrete mechanism-level edits over vague incremental changes.
- Reward plans that include explicit ADD_PROTOCOL-based regression, ablation, and stress tests.
- Penalize plans that add components without clear gating when budget risk is visible.
- Penalize feature dumping and unsupported complexity jumps.
- If the idea drifts from topic constraints, reduce alignment_score.
- In "detected_defects", list ALL defect tags from the registry above that still apply to this idea AFTER the proposed edit. Choose only from the canonical tags.

Return STRICT JSON (no prose):
{{
  "novelty": 0-5,
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
