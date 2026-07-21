---
name: component-coverage
description: Normalize idea components into explicit component-disable requirements
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Component Coverage

## Mission
Convert every component in the idea into a required validation target.

## Protocol
- `idea.json.components` is canonical: use its component names verbatim and preserve its order.
- Every component in `idea.json.components` must appear in the science plan and in reviewer-approved component-disabled condition evidence.
- Do not rename, merge, split, omit, or reorder components.
- Carry each component's explanation forward so later phases can build `method_context` from the original idea statement.
- Assign a `disable_mode` that describes how the component will be disabled for testing.

## Required Output
- Component coverage reflected in the science plan or reviewer-approved component-disabled condition report

## Hard Rule
- There is no implicit exemption for “fixed” components. If a component is not meant to be removed directly, define an explicit disable/substitution mode for testing or mark it blocked with a reason.
- Every final ablation result must include `method_context` for the exact canonical component it evaluates.
