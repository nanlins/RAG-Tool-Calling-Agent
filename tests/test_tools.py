# 工具单元测试
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.tools import execute_tool


class TestCalculator:
    def test_add(self):
        result = execute_tool("calculator", {"a": 1, "b": 2, "operation": "add"})
        assert result["success"]
        assert result["result"] == 3

    def test_divide(self):
        result = execute_tool("calculator", {"a": 10, "b": 2, "operation": "divide"})
        assert result["success"]
        assert result["result"] == 5

    def test_divide_by_zero(self):
        result = execute_tool("calculator", {"a": 10, "b": 0, "operation": "divide"})
        assert not result["success"]

    def test_power(self):
        result = execute_tool("calculator", {"a": 2, "b": 10, "operation": "power"})
        assert result["success"]
        assert result["result"] == 1024


class TestTimeTool:
    def test_get_time(self):
        result = execute_tool("get_current_time", {"timezone": "Asia/Shanghai"})
        assert result["success"]
        assert result["timezone"] == "Asia/Shanghai"
        assert "datetime" in result


class TestWebSearch:
    def test_search(self, monkeypatch):
        from backend import tools
        monkeypatch.setattr(
            tools,
            "execute_web_search",
            lambda args, retriever=None: {
                "success": True,
                "results": [{"title": "mock", "url": "https://example.com", "snippet": "mock"}],
            },
        )
        result = execute_tool("web_search", {"query": "test"})
        assert result["success"]
        assert result["results"][0]["url"] == "https://example.com"


class TestQueryDocs:
    def test_no_retriever(self):
        result = execute_tool("query_documents", {"query": "test"}, retriever=None)
        assert not result["success"]
