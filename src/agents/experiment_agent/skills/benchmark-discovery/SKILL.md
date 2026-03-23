---
name: benchmark-discovery
description: Discover the concrete benchmark surfaces and entrypoints required for evaluation
license: MIT
---

# Benchmark Discovery

## Mission
Translate datasets and repository evidence into the exact benchmark surfaces that later phases must execute.

## Protocol
- Extract benchmark candidates from cloned repositories, dataset READMEs, and the idea description.
- Keep only benchmarks that are directly relevant to validating the idea.
- Mark each benchmark as `ready` or `blocked`.
- Include `required_conditions` such as `baseline`, `full_method`, and stress conditions.
- For each benchmark, record the concrete real data path and the definition of a full run.
- Feed the benchmark evidence back into `agent_reports/prepare_idea.md` under `## Repository-to-Dataset Mapping` so the human-readable document matches the validated worker evidence.
- If a repository has no confirmed benchmark linkage, record `no direct mapping` instead of inventing one.

## Required Output
- Stage worker report requested by the planner

## Failure Conditions
- No benchmark list is produced.
- Benchmarks are listed without conditions or metrics.
- Benchmarks are listed without concrete entrypoints or without saying what a full run means.
- Benchmarks are marked ready without any local evidence.
