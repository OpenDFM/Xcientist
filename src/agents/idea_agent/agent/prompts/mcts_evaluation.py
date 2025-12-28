MCTS_IDEA_EVALUATION_PROMPT = """
You score research ideas encountered during a memory-guided MCTS search.
Topic: {topic}
Latest analysis + critiques:
{analysis}

Candidate idea (JSON):
{idea}

If available, the rewrite path describing applied operators:
{path_summary}

Judge the idea across multi-dimensional criteria. Enforce fairness (explicit baselines, ablations), guard against resource dumping, and highlight uncovered failure modes.

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
