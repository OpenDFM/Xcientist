# Mechanism-Commit Patterns

Load this file only when instantiating the `mechanism-commit-innovation` skill.

## Single Decision Surface Commitment

- Best for defects: `stagnant_novelty`, `unclear_mechanism`, `feature_dumping`
- Core move: commit the novelty to one existing decision boundary such as write/update/evict, route/escalate, or retrieve/re-rank
- Keep fixed: supporting modules around that boundary so the causal story stays narrow
- Concrete output shape: one main controller, scorer, or invariant that changes what the system does at that boundary
- Minimal ablation: turn off only that one mechanism while keeping surrounding instrumentation identical
- Anti-pattern: introducing a second equally important novelty path such as a new router plus a new controller plus a new memory store

## One Invariant, One Controller

- Best for defects: `unclear_mechanism`, `validation_gap`
- Core move: define one measurable invariant, then add one controller that keeps the system near that invariant
- Good invariants: duplicate rate, marginal utility floor, retrieval calibration gap, budget violation rate
- Good controller action: bounded threshold shift, quota adjustment, reweighting coefficient, rollback trigger
- Keep fixed: metrics not needed by the chosen invariant; do not add extra dashboards or evaluators unless they support the single controller
- Minimal ablation: keep the invariant measurement but freeze the controller
- Anti-pattern: measuring many metrics and changing many modules without a single dominant decision rule
