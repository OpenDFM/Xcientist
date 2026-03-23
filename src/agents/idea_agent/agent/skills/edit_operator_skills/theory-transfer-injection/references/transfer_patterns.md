# Theory-Transfer Injection Patterns

Use these cards as soft transfer patterns and negative-transfer checks for `theory-transfer-injection`.
Reuse the imported principle, not the literal names from the source domain.

## Lagrangian Constraint Coupling

- Source principle: constrained optimization with a dual variable
- Imported invariant: performance improves only while a resource or quality constraint stays satisfied
- Injection point: one scalar rule such as retrieval scope, allocation weight, or facet allocation
- Concrete rewrite: maintain a multiplier that reweights the target variable when the observed quality or consistency constraint is violated
- Keep fixed: backbone modules, memory schema, and retrieval stack unless the compiled plan explicitly edits them
- Minimal validation: compare against a static rule and a heuristic reweighting scheme with the same monitoring setup
- Negative transfer signal: the dual variable reacts faster than the environment changes and starts chasing noise instead of the true constraint

## Asymmetric Persistence Rule

- Source principle: control systems avoid switch chatter by using separate enter/exit boundaries
- Imported invariant: state transitions should require stronger evidence to flip back than to stay in the current regime
- Injection point: mode switch, fallback criterion, or multi-path interface
- Concrete rewrite: replace one symmetric trigger rule with asymmetric entry and exit conditions plus a short persistence window
- Keep fixed: the scorer that produces the switching signal; only alter the switching law
- Minimal validation: compare against a single symmetric trigger rule under drift and bursty noise
- Negative transfer signal: hysteresis masks real rapid changes and delays necessary adaptation
