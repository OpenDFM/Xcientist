# Feedback-Closed-Loop Mechanism Cards

Use these cards as soft mechanism patterns and failure-mode checks for `feedback-closed-loop`.
Reuse the pattern, not the literal module names.

## Closed-Loop Update Stabilizer

- Best for defects: `silent_failure`, `drift`, `open_loop_fragility`
- Injection point: one existing update surface such as a write rule, commit rule, retrieval cutoff, or allocation variable
- Inputs: recent utility signal, error rate, rollback count, operating pressure, short rolling history
- Update rule: change one existing update variable with a bounded delta after the monitored signal stays above or below a floor for `k` windows
- Keep fixed: the main data path, storage format, and retrieval backbone unless they already appear in the compiled plan
- Minimal ablation: keep probes/logging/monitoring active, disable only the adaptive update
- Failure boundary: noisy proxy signal causes oscillation; require corroboration across multiple windows and cap per-update magnitude

## Probe-Validated Rollback Rule

- Best for defects: `silent_failure`, `drift`, `open_loop_fragility`
- Injection point: after a risky write, retrieval, or update action that may degrade future retrieval or task success
- Inputs: probe outcome, held-out retrieval quality, rollback allowance, recent update deltas
- Update rule: if post-action quality falls below baseline for `k` checks, rollback the last bounded action and cool down future updates
- Keep fixed: probe protocol and instrumentation so rollback/no-rollback runs stay comparable
- Minimal ablation: same probes and diagnostics, but replace rollback trigger with a no-op recorder
- Failure boundary: rollback fires on short-term noise; require a small confirmation window rather than a single bad observation
