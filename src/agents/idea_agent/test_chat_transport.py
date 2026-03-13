from __future__ import annotations

from types import SimpleNamespace

from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.utils.core.chat_transport import (
    ensure_default_max_output_tokens,
    resolve_chat_transport,
)


class _FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(output_text="responses transport")


class _FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="chat transport"))]
        )


class _FakeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.responses = _FakeResponsesAPI()
        self.chat = SimpleNamespace(completions=_FakeChatCompletionsAPI())


def test_agent_base_uses_chat_completions_for_coding_plan(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
    monkeypatch.delenv("OPENAI_API_STYLE", raising=False)
    client = _FakeClient("https://coding.dashscope.aliyuncs.com/v1")
    agent = AgentBase(chat_model=client)

    text = agent.chat(
        "hello",
        model="kimi-k2.5",
        reasoning={"effort": "low"},
        max_output_tokens=65536,
        temperature=0.2,
    )

    assert text == "chat transport"
    assert not client.responses.calls
    assert len(client.chat.completions.calls) == 1
    request = client.chat.completions.calls[0]
    assert request["model"] == "kimi-k2.5"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["max_tokens"] == 65536
    assert "max_output_tokens" not in request
    assert "reasoning" not in request


def test_agent_base_keeps_responses_for_openai_gpt5(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("OPENAI_API_STYLE", raising=False)
    client = _FakeClient("https://api.openai.com/v1")
    agent = AgentBase(chat_model=client)

    text = agent.chat(
        "hello",
        model="gpt-5-mini",
        reasoning={"effort": "medium"},
        max_output_tokens=65536,
        temperature=1.0,
    )

    assert text == "responses transport"
    assert not client.chat.completions.calls
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5-mini"
    assert request["input"] == [{"role": "user", "content": "hello"}]
    assert request["reasoning"] == {"effort": "medium"}
    assert request["max_output_tokens"] == 65536


def test_resolve_chat_transport_can_be_forced_to_responses() -> None:
    transport = resolve_chat_transport(
        "https://coding.dashscope.aliyuncs.com/v1",
        api_style="responses",
    )

    assert transport == "responses"


def test_ensure_default_max_output_tokens_preserves_existing_limit() -> None:
    explicit = ensure_default_max_output_tokens({"max_output_tokens": 2048}, 65536)
    inherited = ensure_default_max_output_tokens({}, 65536)

    assert explicit["max_output_tokens"] == 65536
    assert inherited["max_output_tokens"] == 65536
