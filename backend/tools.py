# 工具定义与执行模块
import logging
import math
import urllib.parse
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ===== 工具 Schema 定义 (用于 Function Calling) =====

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行数学运算 (加、减、乘、除、幂运算)",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个操作数"},
                "b": {"type": "number", "description": "第二个操作数"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt", "abs"],
                    "description": "运算类型",
                },
            },
            "required": ["a", "b", "operation"],
        },
    },
}

CURRENT_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取指定时区的当前日期和时间",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区名称，如 Asia/Shanghai、America/New_York、Europe/London、Asia/Tokyo",
                    "default": "Asia/Shanghai",
                },
            },
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索互联网上的实时信息。当需要获取最新新闻、实时数据或文档中不包含的信息时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或问题"},
            },
            "required": ["query"],
        },
    },
}

QUERY_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "query_documents",
        "description": "在已导入的知识库文档中搜索相关信息。当用户问题与已上传文档内容相关时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或问题"},
                "top_k": {"type": "number", "description": "返回结果数量", "default": 5},
            },
            "required": ["query"],
        },
    },
}

# 可用工具列表 (按 Agent 需要暴露)
ALL_TOOLS = [CALCULATOR_TOOL, CURRENT_TIME_TOOL, WEB_SEARCH_TOOL, QUERY_DOCS_TOOL]

# 工具名称到参数的映射
TOOL_DEFINITIONS = {
    "calculator": CALCULATOR_TOOL,
    "get_current_time": CURRENT_TIME_TOOL,
    "web_search": WEB_SEARCH_TOOL,
    "query_documents": QUERY_DOCS_TOOL,
}


def execute_calculator(args: dict[str, Any]) -> dict:
    """执行计算器工具"""
    op = args.get("operation", "add")
    a = args.get("a", 0)
    b = args.get("b", 0)
    result = None
    error = None

    try:
        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        elif op == "divide":
            if b == 0:
                error = "除数不能为零"
            else:
                result = a / b
        elif op == "power":
            result = math.pow(a, b)
        elif op == "sqrt":
            if a < 0:
                error = "负数不能开平方"
            else:
                result = math.sqrt(a)
                b = None
        elif op == "abs":
            result = abs(a)
            b = None
        else:
            error = f"未知运算: {op}"
    except Exception as e:
        error = str(e)

    return {"success": error is None, "result": result, "operation": op, "a": a, "b": b, "error": error}


def execute_get_current_time(args: dict[str, Any]) -> dict:
    """执行时间查询工具，基于 IANA 时区"""
    tz_name = args.get("timezone", "Asia/Shanghai")
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        offset_hours = now.utcoffset().total_seconds() / 3600 if now.utcoffset() else 0
        offset_text = f"UTC{offset_hours:+.0f}"
        return {
            "success": True,
            "timezone": tz_name,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
            "utc_offset": offset_text,
        }
    except Exception as e:
        return {"success": False, "error": f"未知时区 {tz_name}: {e}", "timezone": tz_name}


def execute_web_search(args: dict[str, Any], retriever=None) -> dict:
    """执行网页搜索工具 (多引擎: 优先 DuckDuckGo, 降级 Bing)"""
    query = args.get("query", "")
    if not query:
        return {"success": False, "error": "搜索关键词不能为空"}

    # Try 1: DuckDuckGo via ddgs (short timeout)
    try:
        from ddgs import DDGS
        results = []
        for r in DDGS(timeout=5).text(query, max_results=8):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        if results:
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "source": "duckduckgo",
            }
    except Exception as e:
        logger.warning("DuckDuckGo 搜索失败: %s", e)

    # Try 2: Bing via HTML scrape (no API key needed)
    try:
        import httpx
        from lxml import html
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
        r = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        tree = html.fromstring(r.text)
        results = []
        for li in tree.cssselect("li.b_algo")[:8]:
            title_el = li.cssselect("h2 a")
            snippet_el = li.cssselect(".b_caption p")
            if title_el:
                results.append({
                    "title": title_el[0].text_content().strip(),
                    "url": title_el[0].get("href", ""),
                    "snippet": snippet_el[0].text_content().strip() if snippet_el else "",
                })
        if results:
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "source": "bing",
            }
    except Exception as e:
        logger.warning("Bing 搜索失败: %s", e)

    return {"success": False, "error": "搜索服务暂时不可用(已尝试DuckDuckGo和Bing)", "query": query}


def execute_query_documents(args: dict[str, Any], retriever=None) -> dict:
    """执行文档查询工具"""
    query = args.get("query", "")
    top_k = int(args.get("top_k", 5))

    if not query:
        return {"success": False, "error": "查询关键词不能为空"}

    if not retriever:
        return {"success": False, "error": "知识库检索器未初始化", "results": []}

    try:
        chunks, elapsed, _usage = retriever.retrieve(query, top_k=top_k)

        results = []
        for c in chunks:
            meta = c.get("metadata", {})
            results.append({
                "chunk_id": c.get("id", ""),
                "doc_id": meta.get("doc_id", c.get("doc_id", "")),
                "filename": meta.get("filename", "未知"),
                "headers": meta.get("headers", ""),
                "text": c["text"][:500],
                "score": c.get("score", c.get("hybrid_score", 0)),
                "metadata": meta,
            })

        return {
            "success": True,
            "query": query,
            "results": results,
            "total": len(results),
            "elapsed_ms": round(elapsed, 2),
        }
    except Exception as e:
        logger.exception("知识库检索失败")
        return {"success": False, "error": str(e), "results": []}


def execute_tool(tool_name: str, args: dict[str, Any], retriever=None) -> dict:
    """通用工具执行入口"""
    if tool_name == "calculator":
        return execute_calculator(args)
    elif tool_name == "get_current_time":
        return execute_get_current_time(args)
    elif tool_name == "web_search":
        return execute_web_search(args, retriever)
    elif tool_name == "query_documents":
        return execute_query_documents(args, retriever)
    else:
        return {"success": False, "error": f"未知工具: {tool_name}"}