import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retriever import BM25, Retriever, tokenize


class FakeEmbedding:
    def embed(self, text):
        return [0.1, 0.2], {"prompt_tokens": 1, "total_tokens": 1}


class FakeCollection:
    def __init__(self, texts):
        self.texts = texts

    def get(self):
        return {
            "ids": [f"c{i}" for i in range(len(self.texts))],
            "documents": self.texts,
            "metadatas": [{"doc_id": "d1", "filename": "a.md"} for _ in self.texts],
        }


class FakeStore:
    def __init__(self, texts):
        self.collection = FakeCollection(texts)

    def query(self, query_embedding, top_k, score_threshold, doc_id=None):
        return [
            {
                "id": f"c{i}",
                "text": t,
                "metadata": {"doc_id": "d1", "filename": "a.md"},
                "score": max(0.5, 0.95 - i * 0.2),
            }
            for i, t in enumerate(self.collection.texts[:top_k])
        ]


def test_tokenize_chinese_bigrams():
    tokens = tokenize("人工智能")
    assert "人工智能" in tokens
    assert "人工" in tokens


def test_bm25_ranks_relevant_doc():
    bm = BM25()
    bm.fit(["apple banana", "apple apple apple"])
    scores = bm.score("apple banana")
    assert scores[0] > scores[1]


def test_hybrid_retrieve():
    store = FakeStore(["RAG 检索增强生成", "今日天气晴朗"])
    retriever = Retriever(store, FakeEmbedding())
    chunks, elapsed, usage = retriever.retrieve("RAG 是什么", top_k=1)
    assert chunks
    assert "hybrid_score" in chunks[0]
    assert elapsed >= 0
    assert usage["total_tokens"] == 1