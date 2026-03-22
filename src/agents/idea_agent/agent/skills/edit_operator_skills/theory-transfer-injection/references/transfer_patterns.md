# Theory-Transfer Injection Patterns

Load this file only when instantiating the `theory-transfer-injection` skill.

## Lagrangian Budget Controller

- Source principle: constrained optimization with a dual variable
- Imported invariant: performance improves only while a resource or quality constraint stays satisfied
- Injection point: one scalar decision rule such as commit threshold, retrieval budget, or facet quota
- Concrete rewrite: maintain a multiplier that raises or lowers the target threshold when the observed budget or quality constraint is violated
- Keep fixed: backbone modules, memory schema, and retrieval stack unless the compiled plan explicitly edits them
- Minimal validation: compare against a static threshold and a heuristic controller with the same monitoring budget
- Negative transfer signal: the dual variable reacts faster than the environment changes and starts chasing noise instead of the true constraint

## Hysteresis-Stabilized Switching

- Source principle: control systems avoid switch chatter by using separate enter/exit boundaries
- Imported invariant: state transitions should require stronger evidence to flip back than to stay in the current regime
- Injection point: mode switch, escalate/fallback gate, or multi-path coordinator
- Concrete rewrite: replace one symmetric threshold with a pair of asymmetric thresholds plus a short persistence window
- Keep fixed: the scorer that produces the switching signal; only alter the switching law
- Minimal validation: compare against a single-threshold switch under drift and bursty noise
- Negative transfer signal: hysteresis masks real rapid changes and delays necessary adaptation
