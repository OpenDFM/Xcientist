---
name: component-coverage
description: Normalize idea components into explicit ablation requirements
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Component Coverage

## Mission
Convert every component in the idea into a required validation target.

## Protocol
- `idea.json.components` is canonical: use its component names verbatim and preserve its order.
- Every component in `idea.json.components` must appear in the science plan and in validator-backed ablation evidence.
- Do not rename, merge, split, omit, or reorder components.
- Carry each component's explanation forward so later phases can build `method_context` from the original idea statement.
- Assign an `ablation_mode` that describes how the component will be tested.

## Required Output
- Component coverage reflected in the science plan or validator-backed ablation report

## Hard Rule
- There is no implicit exemption for “fixed” components. If a component is not meant to be removed directly, define a degraded fallback or mark it blocked with a reason.
- Every final ablation result must include `method_context` for the exact canonical component it evaluates.
