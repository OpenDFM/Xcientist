# Surgical-Modularity Patterns

Use these cards as soft mechanism patterns and failure-mode checks for `surgical-modularity`.
Reuse the pattern, not the literal module names.

## Weak-Block Swap

- Best for defects: `feature_dumping`, `monolithic_design`, `harder_to_ablate`
- Core move: replace one weak block with one modular block that exposes a clearer input-output interface and a narrower causal claim
- Injection point: an existing scoring, aggregation, or update block already blamed for failure
- Keep fixed: upstream and downstream modules, training setup, and evaluation suite wherever possible
- Minimal validation: swap only this block back to the baseline while keeping interfaces and instrumentation unchanged
- Failure boundary: the "localized" edit silently changes multiple behaviors such as data flow, objective, and scheduler at once

## Interface-First Refactor

- Best for defects: `monolithic_design`, `harder_to_ablate`
- Core move: rewrite a tangled block around one explicit downstream contract so the module can be studied or removed without rewriting the rest of the pipeline
- Injection point: immediately before an existing downstream interface or consumer module
- Keep fixed: the semantics expected by the downstream consumer unless the compiled plan explicitly changes them
- Minimal validation: report whether the module can be bypassed or replaced without collapsing the surrounding system
- Failure boundary: the new module is only a wrapper around the old behavior and does not isolate a real mechanism
