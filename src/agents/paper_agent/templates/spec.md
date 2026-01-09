# Paper Agent Spec (paper-constitution)

This file is the **constitution** for writing an academic paper (not a lab report).
It must be high-level, academically oriented, and **must not pre-write the paper**.
It should guide exploration and writing while preserving freedom to revise claims.

## 0) Non-negotiables (quality + integrity)

- **Academic paper, not report**: the output must read like a conference paper (clear thesis, related work, method, experiments, limitations, conclusion).
- **No new experiments**: do NOT run training/experiments. Only analyze existing artifacts in `project_dir` and `experiment_workspace_dir`.
- **Evidence-first**: every quantitative claim must be backed by concrete files under `project_dir` (path + key/row/line if applicable).
- **Citation gate**: citations are allowed **only** if obtained via the **literature subagent** (Semantic Scholar). No “memory citations”.
- **Always compile with VLM layout review**: every compile iteration must include VLM layout review (no compile-only mode).

## 1) Inputs (runtime)

- `idea.md`: research idea and positioning
- `project_dir`: repository containing code + results
- `paper_dir`: LaTeX template workspace (entry tex may vary)
- `artifact_dir`: analysis/viz/review artifacts (figures/tables/subagent reports)
- `specs_dir`: this spec + plan

## 2) Thesis & central tension (editable)

Write 1–2 sentences:
- **Central tension**: what tradeoff/constraint matters (e.g., accuracy vs efficiency, flexibility vs stability).
- **Thesis**: what we claim to contribute *if evidence supports it*.

> Freedom clause: thesis may be rewritten if evidence contradicts it.

## 3) Paper structure (architect must design)

Define the target paper structure as a list of sections.
For each section, specify **intent**, **reader takeaways**, and **evidence needs** (not prose).

### Required sections (can add/remove subsections)

1. **Abstract**
2. **Introduction**
3. **Related Work**
4. **Method**
5. **Experiments**
6. **Limitations**
7. **Conclusion**

### Section design template (repeat per section)

- **Section name**:
  - **Intent**: what this section must achieve for the reader.
  - **Key points (bullets)**: the minimum set of claims/ideas.
  - **Evidence needed**: what must be true in artifacts for these claims to stand.
  - **Artifacts to produce**: figures/tables/appendix items (if any).
  - **Failure path**: if evidence is missing/weak, how to rewrite this section (e.g., soften claim, present negative result, reposition contribution).

## 4) Claims backlog (guides exploration; not final)

List candidate claims. Each claim is a hypothesis to be tested by exploration.

### Claim template

- **Claim**: (one sentence; falsifiable)
  - **Role in narrative**: (motivation | method novelty | empirical finding | systems result | limitation)
  - **Evidence required** (2–4 bullets): e.g., result_summary, ablation, runtime breakdown, failure case.
  - **Where to look first**: (directories/files patterns; not exhaustive)
  - **Alternative explanations to rule out**: (e.g., data leakage, metric definition mismatch)
  - **If not supported**: (rewrite strategy)

## 5) Visualization & table requirements (paper-level, not “fill tables”)

Specify **types** of figures/tables required, not the exact numbers.
Examples:
- One “method schematic” figure (explains mechanism intuitively).
- One “main results” figure or table (primary comparison).
- One “efficiency/latency” table (if claiming efficiency).
- One “failure case / limitation” figure or table (if applicable).

Each item must include:
- **Purpose** (what question it answers)
- **Evidence source** (what files feed it)
- **Form** (figure/table; expected axes/columns)

## 6) Evidence ledger requirement (provenance)

The paper must be auditable. Maintain a ledger (can be a markdown table) mapping:
- **Claim / number / figure / table** → **source file path(s)** → **extraction method** (manual, script path) → **notes/assumptions**

## 7) Subagent usage policy (when to call which)

- **analysis subagent**: call when you need to turn raw results into paper-ready quantitative narratives and/or generate tables/figures from structured logs.
- **viz subagent**: call when a figure is needed (PDF under `artifact_dir/assets/figures/`) or when a plot spec is unclear.
- **literature subagent** (**mandatory for citations**): call before adding or updating any citation. It must output `references.bib` entries or citation-ready metadata.
- **review subagent**: call after each major writing iteration to compile+VLM+review and produce a prioritized revision plan.

## 8) Output expectations

The writer must produce:
- A complete LaTeX paper that matches the above structure and reads as an academic paper.
- Evidence ledger (or equivalent) showing provenance of quantitative statements.

