# Hierarchical-Decomposition Patterns

Use these cards as soft mechanism patterns and failure-mode checks for `hierarchical-decomposition`.
Reuse the pattern, not the literal module names.

## Planner-Executor Split

- Best for defects: `responsibility_entanglement`, `monolithic_design`, `scale_mismatch`
- Core move: separate long-horizon choice from short-horizon execution so each layer owns a different decision timescale
- Injection point: replace a flat pipeline or single module that currently handles both strategy and action selection
- Keep fixed: low-level primitives if they are already adequate; the hierarchy should change responsibility allocation first
- Minimal validation: compare against the flat version under the same primitives and supervision
- Failure boundary: upper and lower levels relearn the same policy and the hierarchy adds ceremony without clearer role separation

## Summary-Then-Refine Hierarchy

- Best for defects: `responsibility_entanglement`, `scale_mismatch`
- Core move: first compress or summarize global state, then let a lower layer refine local actions conditioned on that summary
- Injection point: before a dense execution path that currently consumes all signals at once
- Keep fixed: the task objective and most downstream operators unless the compiled plan explicitly edits them
- Minimal validation: measure whether high-level summaries improve downstream decisions or just duplicate existing features
- Failure boundary: the summary layer is too lossy or too unconstrained, causing brittle downstream behavior
