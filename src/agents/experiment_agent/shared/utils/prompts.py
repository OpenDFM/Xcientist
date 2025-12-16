"""
Prompt file loader utilities (UTF-8).

We store large system prompts in external files under:
- layers/code/prompts/
- layers/science/prompts/

This module provides:
- UTF-8 safe loading
- Optional lightweight templating using ${var} placeholders (safe for JSON braces and $ENV).
"""

import re
from typing import Dict, Optional


def load_prompt_text(prompt_path: str) -> str:
    if not isinstance(prompt_path, str) or not prompt_path.strip():
        raise ValueError("prompt_path must be a non-empty string")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt_template(
    template_text: str, variables: Optional[Dict[str, str]] = None
) -> str:
    variables = variables or {}
    safe_vars: Dict[str, str] = {}
    for k, v in variables.items():
        if k is None:
            continue
        key = str(k)
        safe_vars[key] = "" if v is None else str(v)
    # Only substitute ${var} patterns. Leave $ENV_VAR intact to avoid clobbering
    # common shell env references inside prompts (e.g., "$SCIENCE_RESULT_DIR").
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

    def _repl(match: "re.Match[str]") -> str:
        name = match.group(1)
        return safe_vars.get(name, match.group(0))

    return pattern.sub(_repl, template_text)


def load_and_render_prompt(
    prompt_path: str, variables: Optional[Dict[str, str]] = None
) -> str:
    text = load_prompt_text(prompt_path)
    return render_prompt_template(text, variables=variables)
