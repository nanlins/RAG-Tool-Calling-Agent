import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import agent as agent_module
from backend.agent import Agent

FAKE_CONFIG = {
    "api_key": "test-key",
    "base_url": "http://fake",
    "chat_model": "fake-model",
    "embedding_model": "",
}


def make_fake_llm(responses):
    state = {"count": 0}

    async def fake_openai_call(self, messages, tools=None, on_token=None):
        index = state["count"]
        state["count"] += 1
        msg, usage = responses[index]
        if on_token and msg.get("content"):
            await on_token(msg["content"])
        return {"message": msg}, usage

    return fake_openai_call, state


def test_agent_tool_loop(monkeypatch):
    monkeypatch.setitem(agent_module.LLM_CONFIGS, "dashscope", FAKE_CONFIG)
    responses = [
        (
            {
                "role": "assistant",
                "content": "先计算",
                "tool_calls": [{
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": json.dumps({"a": 1, "b": 2, "operation": "add"})},
                }],
            },
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        ),
        ({"role": "assistant", "content": "结果是 3"}, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    ]
    fake, state = make_fake_llm(responses)
    monkeypatch.setattr(Agent, "_openai_call", fake)
    agent = Agent(provider="dashscope")
    result = asyncio.run(agent.chat("1+2=?"))

    assert result["success"]
    assert "3" in result["answer"]
    assert state["count"] == 2
    assert any(s["step_type"] == "action" and s["tool_name"] == "calculator" for s in result["steps"])
    assert any(m.get("role") == "tool" for m in result["messages"])
    assert result["conversation_id"]


def test_agent_malformed_tool_args(monkeypatch):
    monkeypatch.setitem(agent_module.LLM_CONFIGS, "dashscope", FAKE_CONFIG)
    responses = [
        (
            {
                "role": "assistant",
                "content": "尝试调用",
                "tool_calls": [{
                    "id": "t2",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{bad json"},
                }],
            },
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        ),
        (
            {"role": "assistant", "content": "参数有误，无法计算"},
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        ),
    ]
    fake, state = make_fake_llm(responses)
    monkeypatch.setattr(Agent, "_openai_call", fake)
    agent = Agent(provider="dashscope")
    result = asyncio.run(agent.chat("算一下"))

    assert state["count"] == 2
    assert any("参数解析失败" in s.get("content", "") for s in result["steps"])


def test_agent_stream_events(monkeypatch):
    monkeypatch.setitem(agent_module.LLM_CONFIGS, "dashscope", FAKE_CONFIG)
    responses = [
        ({"role": "assistant", "content": "你好"}, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    ]
    fake, _ = make_fake_llm(responses)
    monkeypatch.setattr(Agent, "_openai_call", fake)
    agent = Agent(provider="dashscope")

    async def collect():
        events = []
        async for event in agent.stream_chat("hi"):
            events.append(event)
        return events

    events = asyncio.run(collect())
    types = [e["type"] for e in events]
    assert "start" in types
    assert "token" in types
    assert "answer" in types
    assert "result" in types