# Science Plan (Iteration v###): [TITLE]
**Date**: [DATE]  
**Spec**: `spec.md`  
**Idea**: `idea.md`  

## Summary

What we will test in this iteration and why.

## Inputs & Environment

- **Project Root**: [absolute path at runtime]
- **Dataset Directory**: [absolute path at runtime; do not hard-code]
- **Entry Point**: ...

## Experiment Design

- **Baseline**: ...
- **Ablations**: ...
- **Diagnostics**: ...

## Metrics (explicit sources)

For each metric: file path + key/regex + rationale.

## Execution Constraints (Worker must follow)

- Only write under each task's `result_dir`
- Persist stdout/stderr/meta/metrics into result_dir
- Do not modify repository code outside result_dir

## Resume & Traceability

- This iteration is identified as `v###`
- All run artifacts under `_science_runs/step_XXXX/`
- All summaries under `result/science/iter_v###/`



