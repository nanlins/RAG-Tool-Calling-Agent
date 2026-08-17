import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent import Agent
from backend.embedding_client import EmbeddingClient
from backend.models import ProviderConfig
from backend.retriever import Retriever

CUSTOM = ProviderConfig(
    name="my-gateway",
    display_name="My Gateway",
    base_url="https://gw.example.com/v1",
    api_key="sk-custom-secret",
    chat_model="my-chat",
    embedding_model="my-embed",
    embedding_dimensions=128,
    supports_chat=True,
    supports_embedding=True,
)


def test_agent_uses_dynamic_provider_config(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, api_key=None, base_url=None, timeout=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    agent = Agent(provider_config=CUSTOM)
    assert agent.provider == "my-gateway"
    assert agent.model == "my-chat"
    assert captured["api_key"] == "sk-custom-secret"
    assert captured["base_url"] == "https://gw.example.com/v1"


def test_agent_anthropic_style_dynamic(monkeypatch):
    import anthropic

    captured = {}

    class FakeAnthropic:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropic)
    config = CUSTOM.model_copy(update={"anthropic_style": True})
    agent = Agent(provider_config=config)
    assert agent.anthropic_style is True
    assert captured["api_key"] == "sk-custom-secret"
    assert captured["base_url"] == "https://gw.example.com/v1"


def test_agent_requires_dynamic_api_key():
    import pytest

    config = ProviderConfig(name="gw", base_url="https://gw.example.com/v1", chat_model="m")
    with pytest.raises(ValueError, match="API Key"):
        Agent(provider_config=config)


def test_embedding_client_uses_dynamic_provider(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    client = EmbeddingClient(provider_config=CUSTOM)
    assert client.model == "my-embed"
    assert client.get_dimensions() == 128
    assert captured["api_key"] == "sk-custom-secret"
    assert captured["base_url"] == "https://gw.example.com/v1"


class _DefaultEmbedding:
    def embed(self, text):
        return [0.1, 0.2], {"prompt_tokens": 1, "total_tokens": 1}


class _EmptyStore:
    def __init__(self):
        self.collection = object()

    def query(self, query_embedding, top_k, score_threshold, doc_id=None):
        return []


class _DynamicEmbedding:
    def __init__(self):
        self.called = False

    def embed(self, text):
        self.called = True
        return [0.9, 0.9], {"prompt_tokens": 1, "total_tokens": 1}


def test_retriever_accepts_dynamic_embedding_client():
    dynamic = _DynamicEmbedding()
    retriever = Retriever(_EmptyStore(), _DefaultEmbedding())
    chunks, elapsed, usage = retriever.retrieve("hello", top_k=3, hybrid=False, embedding_client=dynamic)
    assert dynamic.called is True
    assert chunks == []
    assert elapsed >= 0
    assert usage["total_tokens"] == 1
