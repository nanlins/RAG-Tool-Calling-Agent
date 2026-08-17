import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _provider(**overrides):
    data = {
        "name": "custom-gw",
        "display_name": "Custom Gateway",
        "base_url": "https://gw.example.com/v1",
        "api_key": "sk-super-secret-key",
        "chat_model": "chat-model",
        "embedding_model": "embed-model",
        "embedding_dimensions": 64,
        "supports_chat": True,
        "supports_embedding": True,
        "anthropic_style": False,
        "native_json_mode": True,
    }
    data.update(overrides)
    return data


class TestProviderTestEndpoint:
    def test_chat_success(self, client, monkeypatch):
        import backend.main as main

        async def fake_test(config, test_type, sample_message):
            assert config.api_key == "sk-super-secret-key"
            assert test_type == "chat"
            assert sample_message == "ping"
            return {
                "ok": True,
                "provider": "custom-gw",
                "model": "chat-model",
                "reply": "pong",
                "elapsed_ms": 12.0,
            }

        monkeypatch.setattr(main, "test_provider", fake_test)
        resp = client.post("/v1/providers/test", json={"provider_config": _provider()})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["reply"] == "pong"

    def test_embedding_success(self, client, monkeypatch):
        import backend.main as main

        async def fake_test(config, test_type, sample_message):
            assert test_type == "embedding"
            assert config.embedding_model == "embed-model"
            return {
                "ok": True,
                "provider": "custom-gw",
                "model": "embed-model",
                "dimensions": 64,
                "elapsed_ms": 8.0,
            }

        monkeypatch.setattr(main, "test_provider", fake_test)
        resp = client.post(
            "/v1/providers/test",
            json={"provider_config": _provider(), "test_type": "embedding"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["dimensions"] == 64

    def test_missing_api_key_returns_400(self, client):
        resp = client.post(
            "/v1/providers/test",
            json={"provider_config": _provider(api_key="")},
        )
        assert resp.status_code == 400
        assert "API Key" in resp.json()["error"]

    def test_missing_base_url_returns_400(self, client):
        resp = client.post(
            "/v1/providers/test",
            json={"provider_config": _provider(base_url="")},
        )
        assert resp.status_code == 400
        assert "Base URL" in resp.json()["error"]

    def test_auth_error_redacts_key(self, client, monkeypatch):
        import backend.main as main

        async def fake_test(config, test_type, sample_message):
            raise RuntimeError("401 invalid api key sk-super-secret-key")

        monkeypatch.setattr(main, "test_provider", fake_test)
        resp = client.post(
            "/v1/providers/test",
            json={"provider_config": _provider(), "test_type": "chat"},
        )
        assert resp.status_code == 401
        assert "sk-super-secret-key" not in resp.text

    def test_gateway_error_redacts_key(self, client, monkeypatch):
        import backend.main as main

        async def fake_test(config, test_type, sample_message):
            raise RuntimeError("gateway 500 sk-super-secret-key")

        monkeypatch.setattr(main, "test_provider", fake_test)
        resp = client.post(
            "/v1/providers/test",
            json={"provider_config": _provider(), "test_type": "chat"},
        )
        assert resp.status_code == 502
        assert "sk-super-secret-key" not in resp.text


class TestVectorStoreProfileAndRebuild:
    def _fake_store(self):
        class FakeStore:
            def __init__(self):
                self.cleared = False
                self.profile = {
                    "embedding_provider": "dashscope",
                    "embedding_model": "text-embedding-v3",
                    "embedding_dimensions": 1536,
                    "hnsw:space": "cosine",
                    "total_chunks": 20,
                    "total_documents": 2,
                }

            def get_profile(self):
                return dict(self.profile)

            def clear_all(self):
                self.cleared = True
                self.profile = {
                    "embedding_provider": "",
                    "embedding_model": "",
                    "embedding_dimensions": None,
                    "hnsw:space": "cosine",
                    "total_chunks": 0,
                    "total_documents": 0,
                }

        return FakeStore()

    def test_profile(self, client, monkeypatch):
        import backend.main as main

        store = self._fake_store()
        monkeypatch.setattr(main, "vector_store", store)
        resp = client.get("/v1/vector-store/profile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["embedding_model"] == "text-embedding-v3"
        assert body["data"]["total_chunks"] == 20

    def test_rebuild_requires_confirm(self, client, monkeypatch):
        import backend.main as main

        store = self._fake_store()
        monkeypatch.setattr(main, "vector_store", store)
        resp = client.post("/v1/vector-store/rebuild", json={"confirm": False})
        assert resp.status_code == 400
        assert "confirm=true" in resp.json()["error"]
        assert store.cleared is False

    def test_rebuild_clears_profile(self, client, monkeypatch):
        import backend.main as main

        store = self._fake_store()
        monkeypatch.setattr(main, "vector_store", store)
        resp = client.post("/v1/vector-store/rebuild", json={"confirm": True})
        assert resp.status_code == 200
        assert store.cleared is True
        assert resp.json()["data"]["profile"]["total_chunks"] == 0


class TestUploadWithProviderConfig:
    def _install_fakes(self, monkeypatch, tmp_path, fail_add=False):
        import backend.main as main

        class FakeDoc:
            id = "doc-1"
            filename = "note.md"
            doc_type = "md"
            title = "note"
            content = "hello rag"
            uploaded_at = "2026-08-16T00:00:00"
            chunk_count = 1

            def to_dict(self):
                return {
                    "id": self.id,
                    "filename": self.filename,
                    "doc_type": self.doc_type,
                    "title": self.title,
                    "content": self.content,
                    "uploaded_at": self.uploaded_at,
                    "chunk_count": self.chunk_count,
                }

        class FakeEmbedding:
            provider = "custom-gw"
            model = "embed-model"

            def __init__(self, provider_config=None):
                self.provider_config = provider_config

            def embed_batch(self, texts):
                return [[0.1] * 64 for _ in texts], {
                    "prompt_tokens": 1,
                    "total_tokens": 1,
                }

        class FakeStore:
            def __init__(self):
                self.added = 0

            def add_documents(self, chunks, embeddings, **kwargs):
                if fail_add:
                    raise ValueError(
                        "向量库由 Embedding 模型 old-model 生成，请先调用 /v1/vector-store/rebuild 重建索引"
                    )
                self.added += len(chunks)
                return len(chunks)

            def get_statistics(self):
                return {
                    "total_chunks": self.added,
                    "total_documents": 1 if self.added else 0,
                }

            def get_profile(self):
                return {
                    "embedding_model": "embed-model",
                    "embedding_dimensions": 64,
                    "total_chunks": self.added,
                    "total_documents": 1 if self.added else 0,
                }

        store = FakeStore()

        def fake_load(file_path, original_filename):
            doc = FakeDoc()
            doc.filename = original_filename
            chunks = [
                {
                    "id": "c1",
                    "text": "hello rag",
                    "metadata": {"doc_id": doc.id, "filename": original_filename},
                }
            ]
            return doc, chunks

        monkeypatch.setattr(main, "EmbeddingClient", FakeEmbedding)
        monkeypatch.setattr(main, "vector_store", store)
        monkeypatch.setattr(main, "retriever", object())
        monkeypatch.setattr(main, "embedding_client", None)
        monkeypatch.setattr(main, "_load_and_split", fake_load)
        monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
        monkeypatch.setattr(main, "save_document", lambda data: None)
        return store

    def _post(self, client, provider_json):
        return client.post(
            "/v1/documents/upload",
            data={"provider_config_json": provider_json},
            files={"file": ("note.md", b"hello rag", "text/markdown")},
        )

    def test_upload_with_provider_config_success(self, client, monkeypatch, tmp_path):
        store = self._install_fakes(monkeypatch, tmp_path)
        resp = self._post(client, json.dumps(_provider()))
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["chunks_added"] == 1
        assert store.added == 1
        assert "sk-super-secret-key" not in resp.text

    def test_upload_invalid_provider_config(self, client, tmp_path):
        import backend.main as main

        monkeypatch = __import__("pytest").MonkeyPatch()
        monkeypatch.setattr(main, "vector_store", None)
        monkeypatch.setattr(main, "retriever", None)
        monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
        try:
            resp = self._post(client, "{bad json")
            assert resp.status_code == 400
            assert "provider_config_json" in resp.json()["error"]
        finally:
            monkeypatch.undo()

    def test_upload_embedding_mismatch_returns_409(self, client, monkeypatch, tmp_path):
        self._install_fakes(monkeypatch, tmp_path, fail_add=True)
        resp = self._post(client, json.dumps(_provider()))
        assert resp.status_code == 409
        assert "重建索引" in resp.json()["error"]
        assert "sk-super-secret-key" not in resp.text


class TestAgentErrorRedaction:
    def test_agent_chat_error_redacts_key(self, client, monkeypatch):
        import backend.main as main

        class FakeAgent:
            provider = "custom-gw"
            model = "chat-model"

            async def chat(self, message, history, conversation_id):
                return {
                    "success": False,
                    "answer": "answer sk-super-secret-key",
                    "error": "boom sk-super-secret-key",
                    "conversation_id": "c1",
                    "steps": [],
                    "retrieved_chunks": [],
                    "usage": {},
                    "elapsed_ms": 1,
                    "messages": [],
                    "warning": "",
                }

        monkeypatch.setattr(
            main,
            "_create_agent",
            lambda provider, model, provider_config=None: FakeAgent(),
        )
        resp = client.post(
            "/v1/agent/chat",
            json={"message": "hi", "provider_config": _provider()},
        )
        assert resp.status_code == 502
        assert "sk-super-secret-key" not in resp.text
