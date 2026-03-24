---
name: iteration-integration
description: Summarize experiment status after each master iteration
license: MIT
---

# Iteration Integration

## Mission

Summarize the current experiment iteration status and produce machine-readable artifacts that help the next master iteration make informed decisions.

## Protocol

After each master iteration completes, this agent reads:
- `idea.json` - experiment components and requirements
- `agent_reports/` - all worker reports, validator reports, planner reports
- `results/` - experiment results (standard, ablation)
- `master_report.md` - previous master decisions

And produces:
- `agent_reports/iteration_summary.md` - Human-readable status summary
- `agent_reports/iteration_status.json` - Machine-readable status for master

## Required Outputs

### iteration_summary.md

Human-readable summary covering:
- Current phase and iteration number
- Code implementation status with evidence
- Standard experiment status with key metrics
- Ablation experiment status with key findings
- Validation status
- Blockers (if any)
- Next recommendations

### iteration_status.json

Machine-readable status with this schema:
```json
{
  "iteration": <number>,
  "code_status": "complete|incomplete|not_started",
  "code_evidence": ["file1", "file2"],
  "standard_experiments": "none|partial|complete",
  "standard_evidence": ["file1", "file2"],
  "ablation_experiments": "none|partial|complete",
  "ablation_evidence": ["file1", "file2"],
  "validation_status": "pass|fail|partial|unknown",
  "key_findings": ["finding1", "finding2"],
  "blockers": ["blocker1"] or [],
  "next_recommendations": ["recommendation1", "recommendation2"]
}
```

## Hard Rules

- Read actual file contents, not just filenames
- For each phase, determine completeness based on evidence
- Identify blockers if phases are incomplete
- CRITICAL: Explicitly tell master agent to read `iteration_status.json` for next decision
- Both output files MUST be written to `agent_reports/`
