import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestHealthEndpoint:
    def test_health_ok(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "embedding_client", object())
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_degraded(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "embedding_client", None)
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert "data" in resp.json()


class TestToolsEndpoint:
    def test_list_tools(self, client):
        resp = client.get("/v1/tools")
        assert resp.status_code == 200
        d = resp.json()
        assert d["success"]
        assert d["count"] >= 4

    def test_execute_calculator(self, client):
        resp = client.post("/v1/tools/execute", json={
            "tool_name": "calculator",
            "arguments": {"a": 10, "b": 3, "operation": "divide"},
        })
        assert resp.status_code == 200
        assert resp.json()["result"]["result"] == 10 / 3

    def test_execute_time(self, client):
        resp = client.post("/v1/tools/execute", json={
            "tool_name": "get_current_time",
            "arguments": {"timezone": "Asia/Shanghai"},
        })
        assert resp.status_code == 200
        assert "datetime" in resp.json()["result"]

    def test_execute_business_error_status(self, client):
        resp = client.post("/v1/tools/execute", json={
            "tool_name": "calculator",
            "arguments": {"a": 1, "b": 0, "operation": "divide"},
        })
        assert resp.status_code == 422
        assert resp.json()["success"] is False


class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        resp = client.get("/v1/documents")
        assert resp.status_code == 200
        assert "success" in resp.json()

    def test_upload_invalid_file(self, client):
        resp = client.post("/v1/documents/upload", files={
            "file": ("test.exe", b"fake", "application/x-msdownload"),
        })
        assert resp.status_code == 400


class FakeRetriever:
    def retrieve(self, query, top_k=5, score_threshold=0.0, doc_id=None):
        return ([{
            "id": "chunk-1",
            "doc_id": "doc-1",
            "text": "test content",
            "metadata": {"filename": "test.md"},
            "score": 0.8,
            "hybrid_score": 0.8,
        }], 1.0, {"prompt_tokens": 1, "total_tokens": 1})


class TestRAGEndpoint:
    def test_rag_query(self, client, monkeypatch):
        from backend import main as backend_main
        monkeypatch.setattr(backend_main, "retriever", FakeRetriever())
        resp = client.post("/v1/rag/query", json={"query": "test", "top_k": 3})
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] == 1


class TestStatsEndpoint:
    def test_stats(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        assert "success" in resp.json()


class TestRedisEndpoint:
    def test_redis_health_fallback(self, client):
        resp = client.get("/v1/redis/health")
        assert resp.status_code == 200
        assert resp.json()["redis_available"] is False


class TestAuthGuard:
    def test_auth_disabled_allows_requests(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_missing_token_refused_when_auth_enabled(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "APP_API_TOKEN", "")
        monkeypatch.setattr(main, "AUTH_DISABLED", False)
        resp = client.get("/v1/models")
        assert resp.status_code == 503
        assert "APP_API_TOKEN" in resp.json()["error"]


class TestRateLimit:
    def test_rate_limit_returns_429(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "RATE_LIMIT_ENABLED", True)
        monkeypatch.setattr(main, "check_rate_limit", lambda *args, **kwargs: False)
        resp = client.get("/v1/tools")
        assert resp.status_code == 429
        assert resp.json()["error"]

    def test_rate_limit_memory_fallback(self):
        from backend.cache import check_rate_limit

        key = "unit-test-ip"
        assert check_rate_limit(key, max_requests=2, window=60) is True
        assert check_rate_limit(key, max_requests=2, window=60) is True
        assert check_rate_limit(key, max_requests=2, window=60) is False