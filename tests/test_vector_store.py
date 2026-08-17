import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.vector_store import VectorStore


def _chunks_and_embeddings():
    chunks = [{"id": "c1", "text": "hello world", "metadata": {"doc_id": "d1", "filename": "a.md"}}]
    embeddings = [[0.1, 0.2, 0.3]]
    return chunks, embeddings


def test_first_add_writes_profile(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "db"))
    chunks, embeddings = _chunks_and_embeddings()
    added = store.add_documents(
        chunks,
        embeddings,
        embedding_provider="dashscope",
        embedding_model="text-embedding-v3",
        embedding_dimensions=3,
    )
    assert added == 1
    profile = store.get_profile()
    assert profile["embedding_provider"] == "dashscope"
    assert profile["embedding_model"] == "text-embedding-v3"
    assert profile["embedding_dimensions"] == 3
    assert profile["total_chunks"] == 1


def test_model_mismatch_rejected(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "db"))
    chunks, embeddings = _chunks_and_embeddings()
    store.add_documents(
        chunks, embeddings, embedding_provider="dashscope", embedding_model="m1", embedding_dimensions=3
    )
    with pytest.raises(ValueError, match="重建索引"):
        store.add_documents(
            [{"id": "c2", "text": "other", "metadata": {"doc_id": "d1"}}],
            [[0.5, 0.6, 0.7]],
            embedding_provider="other",
            embedding_model="m2",
            embedding_dimensions=3,
        )


def test_dimension_mismatch_rejected(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "db"))
    chunks, embeddings = _chunks_and_embeddings()
    store.add_documents(
        chunks, embeddings, embedding_provider="dashscope", embedding_model="m1", embedding_dimensions=3
    )
    with pytest.raises(ValueError, match="维度"):
        store.add_documents(
            [{"id": "c2", "text": "other", "metadata": {"doc_id": "d1"}}],
            [[0.5, 0.6, 0.7, 0.8]],
            embedding_provider="dashscope",
            embedding_model="m1",
            embedding_dimensions=4,
        )


def test_clear_all_resets_profile(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "db"))
    chunks, embeddings = _chunks_and_embeddings()
    store.add_documents(
        chunks, embeddings, embedding_provider="dashscope", embedding_model="m1", embedding_dimensions=3
    )
    store.clear_all()
    profile = store.get_profile()
    assert profile["total_chunks"] == 0
    assert profile["embedding_model"] == ""
    assert profile["embedding_dimensions"] is None
