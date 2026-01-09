# Paper Agent Plan (exploration OS)

This plan is an **operating system** for writer behavior: iterative exploration → claim testing → drafting → compile+VLM+review.
It should not be a checklist of paragraph content.

## 0) Top-level objective

Produce a conference-style academic paper with:
- complete structure (as designed in `spec.md`),
- coherent argumentation (thesis → method → evidence → limitations),
- appropriate visualizations/tables,
- citations **only** via literature subagent,
- always compile with VLM layout review.

## 1) Execution loop (repeat until done)

### Loop A — Evidence discovery & indexing

Goal: quickly build an index of what evidence exists.

- Use bash to locate:
  - result directories, summary files, run reports, logs
  - any scripts/configs that define metrics
- Build/refresh an **Evidence Index** (lightweight):
  - paths, file types, key fields, metric definitions
  - “best candidates” for main results, ablations, efficiency, limitations

Exit condition:
- You can name 3–6 candidate claims and point to plausible evidence sources for each.

### Loop B — Claim testing (one claim at a time)

Pick 1–2 claims from the claims backlog and test them.

- Validate:
  - metric meaning, units, splits, aggregation
  - whether results support the claim or contradict it
- If supported:
  - decide the **paper artifact** that will carry the claim (table/figure/paragraph)
- If not supported:
  - rewrite/downgrade claim (negative result is acceptable)
  - update section design accordingly

Exit condition:
- each “main contribution claim” has a concrete artifact plan and provenance path.

### Loop C — Artifact production (analysis/viz subagents)

Trigger conditions:
- If you need to convert raw logs to tables/figures → call **analysis subagent**.
- If you need a plot/diagram with a clear spec → call **viz subagent**.

Requirements:
- Figures must be PDFs under `artifact_dir/assets/figures/`.
- Tables must be LaTeX under `artifact_dir/assets/tables/` (or integrated cleanly).
- Every produced artifact must be referenced in the evidence ledger.

### Loop D — Writing (LaTeX edits guided by section design)

Write by section **intent** (from `spec.md`), not by dumping results.

Rules:
- Each section must match its designed intent and key points.
- Any quantitative sentence must point to evidence ledger entries.
- Keep prose academic and structured (motivation → method → evidence → implications).

### Loop E — Citations (literature subagent gate)

Trigger conditions:
- Before adding any citation or “related work” claim → call **literature subagent**.
- Use its outputs to:
  - create/update `references.bib`
  - draft related work paragraphs

Rule:
- No citation may appear in LaTeX unless it is justified by literature subagent output.

### Loop F — Compile + VLM + review (always)

Trigger conditions:
- After major writing changes, or after adding tables/figures/citations → call **review subagent**.

Rules:
- Compilation must always include VLM layout review (no compile-only runs).
- Apply fixes in small diffs; re-run review as needed.

## 2) What the writer should do first (bootstrap)

1. Read `spec.md` and `plan.md`.
2. Identify LaTeX entry `.tex` and confirm build pipeline.
3. Build initial evidence index.
4. Draft the section design (if architect left gaps, ask for clarification rather than guessing).

## 3) Completion gates (paper readiness)

- Structure complete and aligned with `spec.md` section design.
- Claims supported or appropriately qualified; limitations clearly stated.
- Visuals/tables are readable and properly placed.
- Citations are literature-subagent-backed and compile cleanly.
- PDF compiles; VLM review issues are addressed or consciously accepted.

