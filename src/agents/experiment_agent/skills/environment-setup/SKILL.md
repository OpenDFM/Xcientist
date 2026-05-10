---
name: environment-setup
description: Create and verify the experiment runtime environment
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Environment Setup

## Mission
Prepare the local execution environment required by the experiment workspace.

## Protocol
- Create or validate the project virtual environment.
- Install only the dependencies required for the planned runs.
- Record which real models are API-backed versus local-checkpoint-backed so prepare can declare them in its validator-backed handoff reports.
- Record environment status, commands, and blockers in a structured report.

## Hard Rules
- Do not mark the environment ready unless the target venv exists and basic activation/import checks pass.
- Never print secret values; only record variable names and purposes.
- If the formal path needs local checkpoints, do not leave the model target unspecified; record it clearly so the dataset/model acquisition step can fetch it.
