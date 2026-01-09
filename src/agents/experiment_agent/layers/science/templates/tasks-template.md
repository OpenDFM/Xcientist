---
description: "Science experiment tasks (Worker execution plan)"
---

# Science Tasks (Iteration v###): [TITLE]

**Inputs**: `spec.md`, `plan_v###.md`, `idea.md`  
**Output root**: `_science_runs/` (raw) + `result/` (summaries)

## Execution Rules (Worker)

- Read constraints first: spec.md, plan.md, tasks.md, and previous report/feedback if present.
- Run from project root; use absolute paths.
- Only write inside each task's `result_dir`.
- Persist orchestration artifacts: `stdout_{task_id}.txt`, `stderr_{task_id}.txt`, `meta_{task_id}.json`, `metrics_{task_id}.json`.

## Task List (human-readable)

- T001 ...
- T002 ...

## Command Blocks (no parser required)

Put each runnable command in its own fenced block. The Worker will execute them sequentially.

### T001: [title]

```bash
# Run from project root
# Must write all artifacts under result/science/iter_v###/runs/T001/
echo "TODO"
```
