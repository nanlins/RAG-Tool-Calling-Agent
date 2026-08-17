# RAG 使用示例
"""
RAG 完整链路演示:
1. 加载文档 -> 2. 切分 -> 3. Embedding -> 4. 向量存储 -> 5. 检索 -> 6. 回答
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.document_loader import load_document
from backend.text_splitter import split_document
from backend.embedding_client import EmbeddingClient
from backend.vector_store import VectorStore
from backend.retriever import Retriever


def demo_rag_pipeline():
    """演示 RAG 完整链路"""
    print("=" * 60)
    print("RAG 完整链路演示")
    print("=" * 60)
    
    # Step 1: 加载文档
    print("\n[Step 1] 加载文档...")
    doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents", "AI学习指南.md")
    if not os.path.exists(doc_path):
        print(f"  文档不存在: {doc_path}")
        return
    doc = load_document(doc_path)
    print(f"  文档: {doc.filename} ({len(doc.content)} 字符)")
    
    # Step 2: 切分
    print("\n[Step 2] 文本切分...")
    chunks = split_document(doc, strategy="recursive", chunk_size=200, chunk_overlap=20)
    print(f"  切分成 {len(chunks)} 个片段")
    for i, c in enumerate(chunks[:3]):
        print(f"  片段 {i+1}: {c['text'][:60]}...")
    
    # Step 3-4: Embedding + 向量存储
    print("\n[Step 3+4] 生成向量并存储到 Chroma...")
    try:
        embedding_client = EmbeddingClient()
        vector_store = VectorStore(collection_name="demo")
        
        texts = [c["text"] for c in chunks]
        embeddings, usage = embedding_client.embed_batch(texts)
        print(f"  向量维度: {len(embeddings[0])}")
        print(f"  已存储 {vector_store.add_documents(chunks, embeddings)} 个片段")
        
        # Step 5-6: 检索 + 展示
        print("\n[Step 5+6] 检索 + 格式化上下文...")
        retriever = Retriever(vector_store, embedding_client)
        
        test_queries = [
            "什么是 RAG?",
            "后端工程包含哪些技术?",
            "Agent 的核心是什么?",
        ]
        
        for q in test_queries:
            print(f"\n--- 查询: {q} ---")
            chunks, elapsed, usage = retriever.retrieve(q, top_k=2)
            context = retriever.format_context(chunks)
            print(f"  耗时: {elapsed:.1f}ms")
            print(f"  检索结果 ({len(chunks)} 条):")
            print(context[:300])
    
    except Exception as e:
        print(f"  错误: {e}")
        print("  请确保已配置 API Key")
    
    print("\n" + "=" * 60)
    print("RAG 链路演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_rag_pipeline()
