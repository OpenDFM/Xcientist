"""Parsing helpers for edit-operator skill metadata and blueprints."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_ATOMIC_OP_VALUES = {
    "ADD_COMPONENT",
    "REMOVE_COMPONENT",
    "REPLACE_COMPONENT",
    "REWIRE",
    "ADD_PROTOCOL",
}


def split_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text

    lines = stripped.splitlines()
    frontmatter: Dict[str, str] = {}
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            body = "\n".join(lines[idx + 1 :])
            return frontmatter, body
        line = lines[idx]
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    return frontmatter, text


def parse_markdown_sections(body: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip().lower().replace(" ", "_")
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            sections[current].append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s", stripped):
            sections[current].append(re.sub(r"^\d+\.\s", "", stripped).strip())
    return sections


def parse_blueprint_step(step: str) -> Optional[Dict[str, Any]]:
    raw = (step or "").strip()
    if not raw:
        return None
    match = re.match(r"^(?P<op>[A-Z_]+)\((?P<body>.*)\)$", raw)
    if not match:
        return None

    op = match.group("op").strip()
    body = match.group("body").strip()
    if op not in _ATOMIC_OP_VALUES:
        return None

    if op == "ADD_PROTOCOL":
        protocols = [token.strip().lower() for token in body.split(",") if token.strip()]
        return {"op": op, "protocols": protocols or ["regression", "ablation", "stress"]}

    if op == "REWIRE":
        source, target = [segment.strip() for segment in body.split("->", 1)] if "->" in body else (body, "downstream")
        return {
            "op": op,
            "component": source,
            "target": target,
            "details": f"Rewire {source} -> {target}",
        }

    if op == "REPLACE_COMPONENT":
        old_component, new_component = [segment.strip() for segment in body.split("->", 1)] if "->" in body else (body, f"{body}_replacement")
        return {
            "op": op,
            "component": new_component,
            "target": old_component,
            "details": f"Replace {old_component} with {new_component}",
        }

    component = body.strip() or "component"
    return {"op": op, "component": component, "details": f"{op} on {component}"}
