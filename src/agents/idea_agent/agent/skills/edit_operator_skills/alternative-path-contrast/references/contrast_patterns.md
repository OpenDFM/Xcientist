# Alternative-Path Contrast Patterns

Use these cards as soft mechanism patterns and failure-mode checks for `alternative-path-contrast`.
Reuse the pattern, not the literal module names.

## Failure-Specific Contrast Path

- Best for defects: `brittle_single_path`, `weak_fallback_behavior`
- Core move: keep one primary regime for common cases and add one alternative treatment whose advantage is defined by a concrete failure condition
- Injection point: an existing failure boundary, escalation criterion, or commit step
- Keep fixed: the primary regime and the evidence used to identify the failure condition; the new contribution is the explicit contrast treatment, not a second full system
- Minimal validation: compare primary-only vs primary+alternative under the failure regime that motivated the contrast
- Failure boundary: a vague failure condition sends too many normal cases down the alternative path and turns the design into hidden duplication

## Rare-Regime Specialist Treatment

- Best for defects: `rare_regime_failure`, `brittle_single_path`
- Core move: define one identifiable rare regime and give it one specialized treatment whose advantage can be stated before training or tuning
- Injection point: after a regime descriptor, horizon-length signal, or uncertainty estimate
- Keep fixed: shared representation and evaluation target unless the compiled plan already edits them
- Minimal validation: report branch coverage, regime-specific win rate, and what happens when the specialist never fires
- Failure boundary: the "rare regime" is vague, post-hoc, or so broad that the system effectively becomes multi-expert feature accumulation
