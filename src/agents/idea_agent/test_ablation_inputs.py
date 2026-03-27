import json

import pytest

from src.agents.idea_agent.utils.core.ablation_inputs import (
    ingest_ablation_results_if_available,
)


class _FakeAgent:
    def __init__(self) -> None:
        self.calls = []

    def ingest_ablation_results(self, payload):
        self.calls.append(payload)


def test_ingest_ablation_results_skips_empty_payload(tmp_path):
    ablation_dir = tmp_path / "ablation_input"
    ablation_dir.mkdir()
    ablation_path = ablation_dir / "ablation_results.json"
    ablation_path.write_text("{}", encoding="utf-8")
    agent = _FakeAgent()

    injected = ingest_ablation_results_if_available(
        agent,
        {"ablation_results_path": str(ablation_dir)},
    )

    assert injected is False
    assert agent.calls == []


def test_ingest_ablation_results_uses_non_empty_payload(tmp_path):
    payload = {
        "components": {
            "memory_router": {
                "result": "negative",
                "metric": "accuracy",
                "value": "-1.4",
                "confidence": 0.8,
                "analysis": "Removing the component hurt accuracy.",
                "method_context": "Routes memory updates into the long-term slot.",
            }
        }
    }
    ablation_dir = tmp_path / "ablation_input"
    ablation_dir.mkdir()
    ablation_path = ablation_dir / "ablation_results.json"
    ablation_path.write_text(json.dumps(payload), encoding="utf-8")
    agent = _FakeAgent()

    injected = ingest_ablation_results_if_available(
        agent,
        {"ablation_results_path": str(ablation_dir)},
    )

    assert injected is True
    assert agent.calls == [payload]


def test_ingest_ablation_results_rejects_multiple_json_files(tmp_path):
    ablation_dir = tmp_path / "ablation_input"
    ablation_dir.mkdir()
    (ablation_dir / "a.json").write_text("{}", encoding="utf-8")
    (ablation_dir / "b.json").write_text("{}", encoding="utf-8")
    agent = _FakeAgent()

    with pytest.raises(ValueError, match="exactly one JSON file"):
        ingest_ablation_results_if_available(
            agent,
            {"ablation_results_path": str(ablation_dir)},
        )
