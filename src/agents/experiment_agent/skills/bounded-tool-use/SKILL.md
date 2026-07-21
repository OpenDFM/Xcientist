---
name: bounded-tool-use
description: Context-efficient file and terminal usage for experiment agents
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Bounded Tool Use

## Mission
Keep tool observations small and high-signal so planning and reporting decisions do not waste context on whole-file or whole-log dumps.

## File Protocol
- Prefer `read_json` for machine-readable artifacts such as reviewer reports, planner reports, and status files.
- Prefer `search` before `view` when you only need a finding, metric, path, or key phrase.
- Use `view` only with a narrow purpose. If you do not know the exact lines yet, search first, then read the smallest useful window.
- Treat directory inspection as bounded discovery only. Do not repeatedly rescan whole trees once key artifacts are known.
- Never request a whole file when a field, match, or line window is sufficient.

## Terminal Protocol
- Prefer targeted commands such as `rg`, `head`, `tail`, `jq`, focused test commands, and explicit file paths.
- Avoid broad `cat`, recursive dumps, or commands that print entire logs when a filtered command would do.
- When a command produces long output, rely on the saved raw output path and follow up with targeted file inspection instead of rerunning a broad command.

## Decision Rules
- Use `think` before multi-step edits, after reviewer failures, and after surprising tool output when you need to compare hypotheses or repair strategies.
- Use `think` to summarize the current state before you choose the next command, task delegation, or retry.
- For master and reporting work: machine-readable status files come first, raw evidence second.
- For planning work: validated handoff artifacts and reviewer JSON define the canonical execution surface.
- For execution work: keep raw outputs on disk, and put only the smallest necessary evidence excerpt into reports.
