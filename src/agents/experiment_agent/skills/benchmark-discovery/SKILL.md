---
name: benchmark-discovery
description: Discover the concrete benchmark surfaces and entrypoints required for evaluation
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Benchmark Discovery

## Mission
Translate datasets and repository evidence into the exact benchmark surfaces that later phases must execute.

## Protocol
- Extract benchmark candidates from cloned repositories, dataset READMEs, and the idea description.
- Keep only benchmarks that are directly relevant to validating the idea.
- Mark each benchmark as `ready` or `blocked`.
- Include required science conditions using the current unified condition model:
  an all-components reference condition and component-disabled conditions for
  each canonical idea component.
- For each benchmark, record the concrete real data path and the definition of a full run.
- **CRITICAL**: The benchmark data files from `dataset_candidate/` MUST be used in experiments. Record the exact file paths so later phases use them.
- Feed the benchmark evidence back into the managed prepare handoff `agent_reports/prepare/artifacts/idea.md` under `## Dataset Usage Guidance` and `## Repository-to-Dataset Mapping` so the human-readable document matches the validated worker evidence.
- If a repository has no confirmed benchmark linkage, record `no direct mapping` instead of inventing one.

## Required Output
- Final worker JSON response requested by the planner; the runtime persists the worker report.

## Failure Conditions
- No benchmark list is produced.
- Benchmarks are listed without conditions or metrics.
- Benchmarks are listed without concrete entrypoints or without saying what a full run means.
- Benchmarks are marked ready without any local evidence.
