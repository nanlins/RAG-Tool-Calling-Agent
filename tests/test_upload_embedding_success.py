import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedding:
    provider = "dashscope"
    model = "text-embedding-v3"

    def __init__(self, fail=False):
        self.fail = fail

    def embed_batch(self, texts):
        if self.fail:
            raise RuntimeError("embedding service down")
        return [[0.1, 0.2] for _ in texts], {"prompt_tokens": len(texts), "total_tokens": len(texts)}


class FakeStore:
    def __init__(self):
        self.added = 0

    def add_documents(self, chunks, embeddings, embedding_provider="", embedding_model="", embedding_dimensions=None):
        self.added = len(chunks)
        return self.added

    def delete_document(self, doc_id):
        return self.added


class FakeRetriever:
    pass


class PartialStore(FakeStore):
    def add_documents(self, chunks, embeddings, embedding_provider="", embedding_model="", embedding_dimensions=None):
        self.added = min(1, len(chunks))
        return self.added


def _upload(client, filename, content):
    return client.post(
        "/v1/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/markdown")},
    )


def test_upload_indexed_and_delete(client, monkeypatch, tmp_path):
    import backend.main as main

    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path / "docs")
    monkeypatch.setattr(main, "embedding_client", FakeEmbedding())
    monkeypatch.setattr(main, "vector_store", FakeStore())
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    resp = _upload(client, "guide.md", "# 指南\n\n这是用于测试检索的知识内容。")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["ingest_status"] == "indexed"
    assert data["chunks_added"] > 0
    assert data["chunk_count"] == data["chunks_added"]

    deleted = client.delete(f"/v1/documents/{data['id']}")
    assert deleted.status_code == 200


def test_upload_embedding_failure_rolls_back(client, monkeypatch, tmp_path):
    import backend.main as main

    before = client.get("/v1/documents").json()["count"]
    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(main, "DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(main, "embedding_client", FakeEmbedding(fail=True))
    monkeypatch.setattr(main, "vector_store", FakeStore())
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    resp = _upload(client, "broken.md", "# 无法入库\n\n这段内容不会进入知识库。")
    assert resp.status_code == 502
    assert client.get("/v1/documents").json()["count"] == before
    assert not list(docs_dir.glob("*"))


def test_upload_partial_returns_202(client, monkeypatch, tmp_path):
    import backend.main as main

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(main, "DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(main, "embedding_client", FakeEmbedding())
    monkeypatch.setattr(main, "vector_store", PartialStore())
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    content = "# 部分入库\n\n" + ("测试内容 " * 300)
    resp = _upload(client, "partial.md", content)
    assert resp.status_code == 202
    body = resp.json()
    assert body["success"] is True
    assert body["warning"]
    assert body["data"]["ingest_status"] == "partial"
    assert body["data"]["chunks_added"] < body["data"]["chunk_count"]