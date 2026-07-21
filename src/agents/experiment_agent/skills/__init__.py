"""
Minimal skill metadata loader for debugging.
Skills are discovered natively by OpenHarness/project skills at runtime.
"""

from pathlib import Path
from typing import Dict, Optional

SKILLS_DIR = Path(__file__).parent


def list_skills() -> Dict[str, str]:
    """Return {name: description} for each SKILL.md under skills/."""
    result: Dict[str, str] = {}
    if not SKILLS_DIR.is_dir():
        return result
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        md = skill_dir / "SKILL.md"
        if not md.is_file():
            continue
        name: Optional[str] = None
        description: Optional[str] = None
        with open(md, "r", encoding="utf-8") as f:
            in_frontmatter = False
            for line in f:
                stripped = line.strip()
                if stripped == "---":
                    if not in_frontmatter:
                        in_frontmatter = True
                    else:
                        break
                    continue
                if in_frontmatter:
                    if stripped.startswith("name:"):
                        name = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("description:"):
                        description = stripped.split(":", 1)[1].strip()
        if name:
            result[name] = description or ""
    return result


if __name__ == "__main__":
    for name, desc in list_skills().items():
        print(f"  {name}: {desc}")
