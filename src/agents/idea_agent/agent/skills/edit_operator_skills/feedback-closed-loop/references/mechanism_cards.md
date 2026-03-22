# Feedback-Closed-Loop Mechanism Cards

Load this file only when instantiating the `feedback-closed-loop` skill.

## Closed-Loop Threshold Controller

- Best for defects: `silent_failure`, `drift`, `open_loop_fragility`, mild `feature_dumping`
- Injection point: one existing decision surface such as write threshold, commit threshold, retrieval cutoff, or escalation gate
- Inputs: recent utility signal, error rate, rollback count, budget pressure, short rolling history
- Update rule: change one control variable with a bounded delta after the monitored signal stays above or below a floor for `k` windows
- Keep fixed: the main data path, storage format, router, and retrieval backbone unless they already appear in the compiled plan
- Minimal ablation: keep probes/logging/monitoring active, disable only the control update
- Failure boundary: noisy proxy signal causes oscillation; require corroboration across multiple windows and cap per-update magnitude

## Probe-Gated Rollback Loop

- Best for defects: `silent_failure`, `drift`, `validation_gap`
- Injection point: after a risky write, routing, or control action that may degrade future retrieval or task success
- Inputs: probe outcome, held-out retrieval quality, rollback budget, recent control deltas
- Update rule: if post-action quality falls below baseline for `k` checks, rollback the last bounded action and cool down future updates
- Keep fixed: probe protocol and instrumentation so rollback/no-rollback runs stay comparable
- Minimal ablation: same probes and diagnostics, but replace rollback trigger with a no-op recorder
- Failure boundary: rollback fires on short-term noise; require a small confirmation window rather than a single bad observation
