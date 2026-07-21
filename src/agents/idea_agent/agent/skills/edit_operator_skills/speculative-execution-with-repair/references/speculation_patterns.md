# Speculative-Execution-with-Repair Patterns

Use these cards as soft mechanism patterns and failure-mode checks for `speculative-execution-with-repair`.
Reuse the pattern, not the literal module names.

## Draft-and-Verify Fast Path

- Best for defects: `over_conservative_execution`, `latency_bottleneck`, `rollback_blindspot`
- Core move: run a cheap optimistic draft path first, then use a bounded check to decide whether to accept, repair, or escalate
- Injection point: before a slow exact path, expensive retrieval stage, or heavyweight planner
- Keep fixed: the conservative baseline so the gain can be attributed to speculation rather than broader redesign
- Minimal validation: compare accept rate, repair rate, and end-to-end cost at matched quality
- Failure boundary: the acceptance check is too weak and silently locks in bad drafts, or too strict and removes the latency benefit

## Partial Rollback with Local Repair

- Best for defects: `rollback_blindspot`, `over_conservative_execution`
- Core move: when speculation fails, undo or patch only the local consequence instead of restarting the whole pipeline
- Injection point: after a speculative write, route choice, or partial plan expansion
- Keep fixed: rollback scope and accounting so repair cost stays comparable across variants
- Minimal validation: measure how often local repair succeeds before full fallback is needed
- Failure boundary: the repair handler grows into a second full execution path and hides the true cost of mis-speculation
