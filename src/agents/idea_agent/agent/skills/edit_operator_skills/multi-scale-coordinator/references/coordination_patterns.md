# Multi-Scale Coordinator Patterns

Use these cards as soft mechanism patterns and failure-mode checks for `multi-scale-coordinator`.
Reuse the pattern, not the literal module names.

## Coarse-to-Fine Consistency Coupling

- Best for defects: `scale_mismatch`, `coordination_failure`, `latency_bottleneck`
- Core move: combine a cheap coarse signal with a finer expensive path, and use one explicit consistency rule to decide when fine-scale input should revise the coarse view
- Injection point: an existing scale interface, summary-refinement stage, or retrieval-depth interface
- Keep fixed: local experts and the global backbone; the novelty is the consistency rule across scales
- Minimal validation: at matched compute, compare no coupling, always-fine, and cross-scale consistency coupling
- Failure boundary: coordination overhead or a stale coarse signal dominates the gain from fine-scale specialization

## Fast-Slow Timescale Coupling

- Best for defects: `coordination_failure`, `scale_mismatch`
- Core move: let a fast path react per step while a slower path updates shared summaries, quotas, or constraints over a longer horizon
- Injection point: between fast execution decisions and slower planning or summary signals
- Keep fixed: scoring functions where possible; change the schedule of influence, not every module at once
- Minimal validation: stress with bursty load or regime shifts to show when the slow signal helps and when it lags
- Failure boundary: the slow path becomes a hidden second mechanism and creates conflicting objectives
