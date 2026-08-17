import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeAgent:
    provider = "dashscope"
    model = "fake-model"

    async def stream_chat(self, message, history=None, conversation_id=None):
        yield {"type": "start", "conversation_id": "c1"}
        yield {"type": "token", "content": "你好"}
        yield {"type": "answer", "content": "你好"}
        yield {"type": "done", "conversation_id": "c1"}
        yield {
            "type": "result",
            "result": {
                "conversation_id": "c1",
                "messages": [],
                "steps": [],
                "answer": "你好",
                "usage": {},
                "elapsed_ms": 1,
                "success": True,
                "warning": "",
            },
        }


def test_stream_endpoint_returns_sse(client, monkeypatch):
    import backend.main as main

    monkeypatch.setattr(main, "_create_agent", lambda provider, model, provider_config=None: FakeAgent())
    resp = client.post("/v1/agent/chat/stream", json={"message": "hi", "stream": True})
    assert resp.status_code == 200
    text = "".join(resp.iter_text())
    assert "data: " in text
    assert '"type": "token"' in text
    assert "[DONE]" in text