MCTS_IDEA_EVALUATION_PROMPT = """
You score research ideas encountered during a memory-guided MCTS search.
Topic: {topic}
Latest analysis + critiques:
{analysis}

Relevant literature evidence from the current paper cache:
{paper_context}

Candidate idea (JSON):
{idea}

If available, the rewrite path describing applied operators:
{path_summary}

ICML bar reminders:
- Reward ideas that introduce brand-new mechanisms, training contracts, or cross-domain theory transfers with higher novelty/impact.
- Penalize "just add a gate/MoE/ensemble" tweaks (novelty ≤ 1, impact ≤ 2) unless the idea explicitly offers a new scientific insight.
- Elevate proposals that overhaul evaluation contracts or surface new failure science; note this inside feedback.
- Explicitly call out if the idea fails the ICML bar due to lack of mechanism clarity, missing evaluation, or excessive incrementalism.
- If the core contribution is only a new protocol/benchmark/dataset, cap novelty ≤ 1 and impact ≤ 1 unless the response also introduces a concrete algorithmic method enabled by that protocol.

Judge the idea across multi-dimensional criteria. Enforce fairness (explicit baselines, ablations), guard against resource dumping, and highlight uncovered failure modes. Reward concrete algorithmic/mechanistic innovations; penalize responses that only add analysis/instrumentation/protocol tweaks without a new intervention (novelty <= 2 in those cases).

Return STRICT JSON (no prose) using:
{{
  "novelty": 0-5,
  "feasibility": 0-5,
  "clarity": 0-5,
  "impact": 0-5,
  "risk": 0-5,  # higher means riskier
  "conciseness": 0-5,
  "confidence": 0-1,
  "failure_modes": ["list at least one concrete failure mode"],
  "fairness_protocol": "How fairness / control experiments are enforced or what's missing",
  "feedback": "Actionable critique referencing defects/operators",
  "defect_fix_summary": "Which defect was addressed and why the operator helps",
  "lift_estimate": 0-100  # expected % improvement over parent baseline
}}
"""
