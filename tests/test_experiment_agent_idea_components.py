import json

import pytest

from src.agents.experiment_agent.runtime.idea_components import (
    IDEA_COMPONENTS_HEADING,
    canonical_component_names,
    format_canonical_components_markdown,
    load_canonical_components,
)


def test_load_canonical_components_preserves_order_and_explanation(tmp_path):
    payload = {
        "title": "demo",
        "components": [
            {"component": "component_a", "explanation": "first"},
            {"component": "component_b", "explanation": "second"},
        ],
    }
    (tmp_path / "idea.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    components = load_canonical_components(str(tmp_path))

    assert components == [
        {"component": "component_a", "explanation": "first", "index": "1"},
        {"component": "component_b", "explanation": "second", "index": "2"},
    ]
    assert canonical_component_names(str(tmp_path)) == ["component_a", "component_b"]


def test_load_canonical_components_rejects_duplicates(tmp_path):
    payload = {
        "components": [
            {"component": "component_a", "explanation": "first"},
            {"component": "component_a", "explanation": "duplicate"},
        ]
    }
    (tmp_path / "idea.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate component"):
        load_canonical_components(str(tmp_path))


def test_format_canonical_components_markdown_lists_components_in_order():
    rendered = format_canonical_components_markdown(
        [
            {"component": "component_a", "explanation": "first", "index": "1"},
            {"component": "component_b", "explanation": "second", "index": "2"},
        ]
    )

    assert IDEA_COMPONENTS_HEADING == "## Idea Components"
    assert rendered.splitlines()[0] == "1. `component_a`"
    assert rendered.splitlines()[2] == "2. `component_b`"
