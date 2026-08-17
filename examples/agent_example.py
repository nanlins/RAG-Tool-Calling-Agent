# Agent 使用示例
"""
演示 Agent 对话、工具调用和 RAG 检索
"""
import os
import sys
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def demo_agent():
    from backend.agent import Agent
    from backend.vector_store import VectorStore
    from backend.embedding_client import EmbeddingClient
    from backend.retriever import Retriever
    
    print("=" * 60)
    print("Agent 对话演示")
    print("=" * 60)
    
    # 初始化
    try:
        vector_store = VectorStore()
        embedding_client = EmbeddingClient()
        retriever = Retriever(vector_store, embedding_client)
        agent = Agent(retriever=retriever)
    except Exception as e:
        print(f"初始化失败: {e}")
        print("使用无检索模式...")
        agent = Agent()
        retriever = None
    
    test_cases = [
        "现在几点了?",
        "计算 123 + 456 等于多少?",
        "搜索一下什么是 AI Agent",
    ]
    
    for question in test_cases:
        print(f"\n{'=' * 50}")
        print(f"用户: {question}")
        print(f"{'=' * 50}")
        
        result = await agent.chat(question)
        
        print(f"\n=== 推理过程 ===")
        for step in result.get("steps", []):
            st = step.get("step_type", "")
            content = step.get("content", "")
            if st == "thought" and content:
                print(f"  [思考] {content[:150]}")
            elif st == "action":
                tn = step.get("tool_name", "")
                ta = step.get("tool_args", {})
                print(f"  [行动] 调用: {tn}({ta})")
            elif st == "observation":
                print(f"  [观察] 工具返回结果")
        
        print(f"\n=== 最终回答 ===\n{result.get('answer', '')[:300]}")
        usage = result.get("usage", {})
        print(f"\n[用量] {usage.get('total_tokens', 0)} tokens | "
              f"耗时: {result.get('elapsed_ms', 0):.0f}ms")
    
    print("\n" + "=" * 60)
    print("Agent 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_agent())
