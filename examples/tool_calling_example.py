# 工具调用示例
"""
演示工具调用的基本用法
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools import execute_tool, ALL_TOOLS


def demo_tools():
    print("=" * 60)
    print("工具调用演示")
    print("=" * 60)
    
    # 列出所有可用工具
    print("\n可用工具:")
    for t in ALL_TOOLS:
        fn = t.get("function", {})
        params = fn.get("parameters", {}).get("properties", {})
        print(f"  - {fn['name']}: {fn.get('description', '')}")
        print(f"    参数: {list(params.keys())}")
    
    # 测试各个工具
    test_cases = [
        ("calculator", {"a": 100, "b": 25, "operation": "divide"}),
        ("calculator", {"a": 2, "b": 10, "operation": "power"}),
        ("get_current_time", {"timezone": "Asia/Shanghai"}),
        ("get_current_time", {"timezone": "America/New_York"}),
        ("web_search", {"query": "2026 AI technology trends"}),
    ]
    
    for tool_name, args in test_cases:
        print(f"\n--- 调用 {tool_name}({json.dumps(args, ensure_ascii=False)}) ---")
        result = execute_tool(tool_name, args)
        print(f"  结果: {json.dumps(result, ensure_ascii=False)[:200]}")
    
    print("\n" + "=" * 60)
    print("工具调用演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_tools()
