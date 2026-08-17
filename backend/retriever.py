# Retrieval module - hybrid search (vector + BM25)
import math
import re
import time
from collections import Counter

from backend.config import BM25_WEIGHT, SCORE_THRESHOLD, TOP_K, VECTOR_WEIGHT
from backend.embedding_client import EmbeddingClient
from backend.vector_store import VectorStore


def tokenize(text: str) -> list[str]:
    """轻量中英文分词：英文按词，中文按单字 + 二元组"""
    tokens = []
    for m in re.finditer(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        part = m.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) >= 2:
                tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            tokens.append(part)
        else:
            tokens.append(part)
    return tokens


class BM25:
    """BM25 scoring for keyword-based retrieval"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.avgdl = 0
        self.N = 0
        self.corpus = []

    def fit(self, corpus: list[str]):
        self.corpus = corpus
        self.N = len(corpus)
        self.doc_lens = [len(tokenize(d)) for d in corpus]
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)
        for d in corpus:
            for term in set(tokenize(d)):
                self.doc_freqs[term] += 1

    def score(self, query: str) -> list[float]:
        scores = []
        q_terms = tokenize(query)
        for i, doc in enumerate(self.corpus):
            score = 0
            dl = self.doc_lens[i]
            doc_terms = tokenize(doc)
            doc_counter = Counter(doc_terms)
            for qt in q_terms:
                if qt not in self.doc_freqs:
                    continue
                idf = math.log((self.N - self.doc_freqs[qt] + 0.5) / (self.doc_freqs[qt] + 0.5) + 1)
                freq = doc_counter.get(qt, 0)
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * num / den if den else 0
            scores.append(score)
        return scores


class Retriever:
    """Hybrid retriever: vector (Chroma) + keyword (BM25)"""

    def __init__(self, store: VectorStore, embedding_client: EmbeddingClient | None = None):
        self.store = store
        self.embedding_client = embedding_client or EmbeddingClient()
        self.bm25 = None
        self._chunks_cache = {"ids": [], "documents": [], "metadatas": []}

    def _build_bm25(self, force=False):
        if self.bm25 is None or force:
            results = self.store.collection.get()
            texts = results.get("documents", [])
            if texts:
                self.bm25 = BM25()
                self.bm25.fit(texts)
                self._chunks_cache = {
                    "ids": results.get("ids", []),
                    "documents": texts,
                    "metadatas": results.get("metadatas", []) or [],
                }

    def retrieve(self, query: str, top_k: int = TOP_K,
                 score_threshold: float = SCORE_THRESHOLD,
                 doc_id: str | None = None,
                 hybrid: bool = True,
                 embedding_client: EmbeddingClient | None = None):
        start = time.time()
        client = embedding_client or self.embedding_client
        query_embedding, usage = client.embed(query)

        vec_chunks = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k * 2 if hybrid else top_k,
            score_threshold=score_threshold,
            doc_id=doc_id,
        )

        if hybrid:
            self._build_bm25()
            if self.bm25 and self._chunks_cache.get("documents"):
                bm25_scores = self.bm25.score(query)
                corpus_map = {}
                for i, t in enumerate(self._chunks_cache["documents"]):
                    meta = self._chunks_cache["metadatas"][i] if i < len(self._chunks_cache["metadatas"]) else {}
                    cid = self._chunks_cache["ids"][i] if i < len(self._chunks_cache["ids"]) else ""
                    corpus_map[t] = {
                        "bm25": bm25_scores[i] if i < len(bm25_scores) else 0,
                        "id": cid,
                        "metadata": meta,
                    }

                seen = set()
                for c in vec_chunks:
                    entry = corpus_map.get(c["text"], {})
                    bm25 = entry.get("bm25", 0)
                    vec_score = c.get("score", 0)
                    c["hybrid_score"] = round(
                        vec_score * VECTOR_WEIGHT + min(bm25 / 5, 1) * BM25_WEIGHT, 4
                    )
                    seen.add(c["text"])

                for text, entry in sorted(corpus_map.items(), key=lambda x: -x[1]["bm25"])[:top_k * 2]:
                    if text not in seen and entry["bm25"] > 0:
                        vec_chunks.append({
                            "id": entry["id"],
                            "text": text[:500],
                            "metadata": entry["metadata"],
                            "score": min(entry["bm25"] / 5, 1),
                            "hybrid_score": round(min(entry["bm25"] / 5, 1), 4),
                        })

                vec_chunks = [
                    c for c in vec_chunks
                    if c.get("hybrid_score", c.get("score", 0)) >= score_threshold
                ]
                vec_chunks.sort(key=lambda x: -x.get("hybrid_score", x.get("score", 0)))
                vec_chunks = vec_chunks[:top_k]
        else:
            vec_chunks.sort(key=lambda x: -x.get("score", 0))
            vec_chunks = vec_chunks[:top_k]

        elapsed = (time.time() - start) * 1000
        return vec_chunks, elapsed, usage

    def format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "unknown")
            headers = meta.get("headers", "")
            score = chunk.get("hybrid_score", chunk.get("score", 0))
            h_info = f" [{headers}]" if headers else ""
            parts.append(
                f"[Source {i}] File: {filename}{h_info} (relevance: {score:.2f})\n"
                f"{chunk['text']}\n"
            )
        return "\n---\n".join(parts)