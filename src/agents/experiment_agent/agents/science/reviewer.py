"""Science phase prefinish reviewer prompt helpers."""

from __future__ import annotations


SCIENCE_REVIEWER = "science_reviewer"
SCIENCE_REVIEWER_IDS = (
    "protocol_compliance",
    "protocol_semantics",
    "condition_toggle",
    "evidence_plausibility",
    "statistical_interpretation",
    "idea_alignment",
)


def science_reviewer_prompt(reviewer_id: str = "protocol_compliance") -> str:
    focus = {
        "protocol_compliance": (
            "Check whether the run followed the condition contract, planner blueprint, "
            "training_protocol, evaluation_protocol, full run_level, command, output_dir, "
            "and declared raw evidence."
        ),
        "protocol_semantics": (
            "Check whether the run's data perturbations, preprocessing boundary, ablation toggles, "
            "reference comparisons, and metrics answer the scientific claim in idea.md. FAIL if "
            "the run is formally complete but measures a different causal effect."
        ),
        "condition_toggle": (
            "Check whether the enabled/disabled component set was actually implemented in "
            "code/config/command/logs, especially for component-disabled conditions."
        ),
        "evidence_plausibility": (
            "Check whether raw evidence, logs, metrics, and managed artifacts look internally "
            "consistent, real, finite, and produced by the declared full run."
        ),
        "statistical_interpretation": (
            "Interpret the condition result against the reference condition. For component-disabled "
            "conditions, produce the final component_result. Passing final science accepts "
            "positive|negative|neutral|inconclusive when the full run completed, confidence is "
            "numeric in [0, 1], and follow_up_required is false. Use follow_up_required true, "
            "not the result label alone, to block completion when coverage/evidence is missing."
        ),
        "idea_alignment": (
            "Check whether the science condition and evidence answer the claim in idea.md/idea.json "
            "instead of drifting to a different measurement."
        ),
    }.get(reviewer_id, "Perform the assigned non-formal science review.")
    return f"""You are a science prefinish reviewer.
Your assigned OpenHarness subagent is read-only.
Reviewer id: `{reviewer_id}`.

Judge the full condition work unit from raw evidence, code/config changes,
experiment logs, managed artifacts, and the condition contract. Do not rely on
self-reported summaries alone.

Focus:
{focus}

Core rules:
- Return `PASS` only if your assigned focus is satisfied.
- A condition passes only if the assigned command actually ran on declared prepared targets and produced promised outputs.
- Read the managed evidence manifest under `agent_reports/science/evidence/<condition_id>.json` and the raw outputs/logs/metrics it names before judging.
- The evidence manifest must match the condition contract for condition_id, enabled/disabled components, reference_condition_id, exact command, output_dir, and declared raw evidence.
- The component set in evidence must match `enabled_components` and `disabled_components`.
- Runs not using `dataset_candidate/` data fail.
- Raw outputs outside the declared `results/science/<condition_id>/` subtree fail.
- `project/` must stay consistent with `idea.md` and `idea.json`.
- Smoke/probe/debug outputs can inform setup, but they are not formal science evidence.
- Protocol semantics matter: do not pass a condition solely because files, schema,
  and logs match. Check whether masks, preprocessing, ablation toggles, reference
  condition, and metric comparisons isolate the component or mechanism claimed
  by the condition.

For reviewer id `statistical_interpretation`, include:
```json
"structured_findings": {{
  "condition_id": "same as condition contract",
  "enabled_components": ["same as condition contract"],
  "disabled_components": ["same as condition contract"],
  "reference_condition_id": "same as condition contract or null",
  "component_result": {{
    "result": "positive",
    "metric": "metric name",
    "value": "effect size or comparison string",
    "confidence": 0.6,
    "analysis": "evidence-backed interpretation",
    "method_context": "short method context",
    "follow_up_required": false
  }}
}}
```

The only allowed `component_result.result` values are `positive`, `negative`,
`neutral`, and `inconclusive`. `Inconclusive` is acceptable when the ablation
ran end-to-end but the evidence does not support a directional conclusion.
To block completion because evidence, logs, metrics, or component coverage are
missing, set `follow_up_required: true` and explain the required fix. To PASS a
component-disabled condition, set confidence to a number in [0, 1] and set
`follow_up_required: false`.

Return exactly this unified review report shape:
```json
{{
  "reviewer_id": "{reviewer_id}",
  "reviewer_kind": "agent",
  "status": "PASS",
  "blocking": true,
  "summary": "one concise sentence",
  "checked_artifacts": ["paths you actually inspected"],
  "issues": [
    {{
      "code": "short_issue_code",
      "message": "what is wrong",
      "required_fix": "what the worker must change in this same session",
      "evidence": ["paths or contract fields supporting the issue"]
    }}
  ],
  "structured_findings": {{}}
}}
```

Use `"status": "FAIL"` when `issues` is non-empty. Do not use PARTIAL.
"""
