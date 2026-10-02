# 向量存储模块 - 基于 Chroma
import logging
import os

from backend.config import CHROMA_DB_DIR, EMBEDDING_PROVIDER, LLM_CONFIGS

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储，封装 Chroma 操作"""

    def __init__(self, collection_name: str = "documents", persist_dir: str | None = None):
        self.persist_dir = str(persist_dir or CHROMA_DB_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = collection_name
        self._init_chroma()
        self._init_embedding_function()

    def _init_chroma(self):
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _init_embedding_function(self):
        """初始化默认 Embedding 函数 (使用 Chroma 内置的 OpenAI)"""
        # Chroma 的 OpenAI 嵌入函数会在 add/query 时自动调用
        # 我们自行管理 embedding, 存入预计算向量
        self._openai_available = bool(LLM_CONFIGS.get(EMBEDDING_PROVIDER, {}).get("embedding_model"))

    def _current_metadata(self) -> dict:
        """返回集合级 metadata（始终返回新 dict，避免污染 Chroma 对象）"""
        return dict(self.collection.metadata or {})

    def get_profile(self) -> dict:
        """返回当前向量库 Embedding profile 与统计"""
        stats = self.get_statistics()
        meta = self._current_metadata()
        return {
            "embedding_provider": meta.get("embedding_provider", ""),
            "embedding_model": meta.get("embedding_model", ""),
            "embedding_dimensions": meta.get("embedding_dimensions"),
            "hnsw:space": meta.get("hnsw:space", "cosine"),
            "total_chunks": stats["total_chunks"],
            "total_documents": stats["total_documents"],
        }

    def _ensure_profile(self, embeddings: list[list[float]], provider: str,
                        model: str, dimensions: int | None):
        """首次入库写入 profile；已有索引时校验 Embedding 模型与维度一致性"""
        stats = self.get_statistics()
        meta = self._current_metadata()
        new_dims = dimensions
        if new_dims is None and embeddings:
            new_dims = len(embeddings[0])

        if stats["total_chunks"] == 0:
            metadata = {}
            if provider:
                metadata["embedding_provider"] = provider
            if model:
                metadata["embedding_model"] = model
            if new_dims is not None:
                metadata["embedding_dimensions"] = int(new_dims)
            self.collection.modify(metadata=metadata)
            return

        existing_model = meta.get("embedding_model", "")
        existing_dims = meta.get("embedding_dimensions")
        if not existing_model or existing_dims is None:
            raise ValueError(
                "向量库缺少 Embedding 配置信息，请先调用 /v1/vector-store/rebuild 重建索引"
            )
        if model and existing_model != model:
            raise ValueError(
                f"向量库由 Embedding 模型 {existing_model} 生成，当前模型为 {model}，"
                "请先调用 /v1/vector-store/rebuild 重建索引"
            )
        if new_dims is not None and int(existing_dims) != int(new_dims):
            raise ValueError(
                f"向量维度不一致：索引维度 {existing_dims}，当前维度 {new_dims}，"
                "请先调用 /v1/vector-store/rebuild 重建索引"
            )

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]],
                      embedding_provider: str = "", embedding_model: str = "",
                      embedding_dimensions: int | None = None) -> int:
        """添加文档片段到向量库；首次入库写入 Embedding profile，模型不一致时拒绝"""
        if not chunks or not embeddings:
            return 0
        self._ensure_profile(embeddings, embedding_provider, embedding_model, embedding_dimensions)

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # 检查是否已存在
        existing = self.collection.get(ids=ids)
        if existing and existing["ids"]:
            # 跳过已存在的
            new_ids = [i for i in ids if i not in existing["ids"]]
            new_texts = [t for i, t in zip(ids, texts, strict=False) if i in new_ids]
            new_metadatas = [m for i, m in zip(ids, metadatas, strict=False) if i in new_ids]
            new_embeddings = [e for i, e in zip(ids, embeddings, strict=False) if i in new_ids]
            if not new_ids:
                return 0
            ids, texts, metadatas, embeddings = new_ids, new_texts, new_metadatas, new_embeddings

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        return len(ids)

    def query(self, query_embedding: list[float], top_k: int = 5,
              score_threshold: float = 0.0, doc_id: str | None = None) -> list[dict]:
        """检索最相似的文档片段"""
        where = {}
        if doc_id:
            where = {"doc_id": doc_id}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                score = 1 - results["distances"][0][i] if results["distances"] else 1.0
                if score_threshold > 0 and score < score_threshold:
                    continue
                chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": round(score, 4),
                })

        return chunks

    def delete_document(self, doc_id: str) -> int:
        """删除指定文档的所有片段"""
        results = self.collection.get(where={"doc_id": doc_id})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def get_document_ids(self) -> list[str]:
        """获取所有文档 ID (去重)"""
        results = self.collection.get()
        doc_ids = set()
        if results and results.get("metadatas"):
            for m in results["metadatas"]:
                if m and "doc_id" in m:
                    doc_ids.add(m["doc_id"])
        return list(doc_ids)

    def get_statistics(self) -> dict:
        """获取向量库统计信息"""
        results = self.collection.get()
        count = len(results.get("ids", []))
        doc_ids = set()
        docs_info = {}
        if results.get("metadatas"):
            for m in results["metadatas"]:
                if m:
                    did = m.get("doc_id", "")
                    fname = m.get("filename", "")
                    doc_ids.add(did)
                    if did not in docs_info:
                        docs_info[did] = {"filename": fname, "chunks": 0}
                    docs_info[did]["chunks"] += 1

        return {
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "documents": docs_info,
        }

    def clear_all(self):
        """清空所有向量数据并移除 Embedding profile，供重建索引使用"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning("清空向量库失败: %s", e)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

# 修改记录：
#   2026-10-01 Embedding 供应商判断改用 EMBEDDING_PROVIDER（与聊天供应商解耦）
