# Mechanism-Commit Patterns

Use these cards as soft mechanism patterns and failure-mode checks for `mechanism-commit-innovation`.
Reuse the pattern, not the literal module names.

## Single Mechanism Surface Commitment

- Best for defects: `stagnant_novelty`, `unclear_mechanism`, `validation_gap`
- Core move: commit the novelty to one existing mechanism surface such as write/update/evict, retrieve/re-rank, or allocation
- Keep fixed: supporting modules around that boundary so the causal story stays narrow
- Concrete output shape: one main mechanism, scorer, or invariant carrier that changes what the system does at that surface
- Minimal ablation: turn off only that one mechanism while keeping surrounding instrumentation identical
- Hard rule: if the parent idea is not already centered on threshold/control logic, do not default to thresholding, gating, suppression, or quota-style patches.
- Anti-pattern: introducing a second equally important novelty path such as a new fallback branch plus a new adaptation rule plus a new memory store

## One Invariant, One Adjustment Rule

- Best for defects: `unclear_mechanism`, `validation_gap`
- Core move: define one measurable invariant, then add one adjustment rule that keeps the system near that invariant
- Good invariants: duplicate rate, marginal utility floor, retrieval calibration gap, constraint violation rate
- Good adjustment action: bounded reweighting coefficient, retention penalty, merge rule, rollback trigger
- Keep fixed: metrics not needed by the chosen invariant; do not add extra dashboards or evaluators unless they support the single adjustment rule
- Minimal ablation: keep the invariant measurement but freeze the adjustment rule
- Anti-pattern: measuring many metrics and changing many modules without a single dominant decision rule
