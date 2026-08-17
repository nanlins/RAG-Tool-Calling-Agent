# Agent 核心模块 - ReAct 循环实现
import asyncio
import contextlib
import json
import time
import traceback
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import uuid4

from backend.config import DEFAULT_PROVIDER, LLM_CONFIGS
from backend.retriever import Retriever
from backend.tools import ALL_TOOLS, execute_tool


def build_agent_system_prompt() -> str:
    """动态生成 Agent System Prompt，日期始终取服务器当前时间"""
    now = datetime.now().strftime("%Y年%m月%d日")
    return f"""当前日期: {now}。你是一个智能 AI 助手，可以调用工具、检索知识库来回答用户问题。

【能力范围】
1. 直接回答：如果你确定用户问题在你的知识范围内，可以直接回答
2. 检索知识库：调用 query_documents 工具在已导入文档中搜索相关信息
3. 调用工具：使用 calculator 进行数学计算，get_current_time 获取时间，web_search 搜索网络信息
4. 组合使用：可以先检索知识库，再结合工具计算或搜索

【规则】
- 必须基于可信信息回答，不确定时明确说明
- 每次调用工具前先说明为什么要调用
- 工具结果返回后，整合信息给出完整、自然的回答
- 如果知识库和工具都无法获取相关信息，直接说不知道，不要编造
- 引用来源时标注文档名称和片段

【思考格式】
每次行动前思考：
- 用户需要什么？
- 我现在掌握哪些信息？
- 我需要检索知识库还是调用工具？
- 调用什么工具最合适？
"""


# RAG 专用系统提示
RAG_SYSTEM_PROMPT = """你是一个 AI 知识助手，基于提供的文档内容回答用户问题。

【规则】
- 只基于提供的文档内容回答
- 必须引用来源（文档名+原文片段）
- 如果文档中没有相关信息，明确说"在已导入文档中未找到相关信息"
- 不确定时不要猜测或者编造
- 回答时保持简洁、清晰
"""


def _default_result(conversation_id: str = "") -> dict:
    return {
        "success": False,
        "answer": "Agent 未返回结果",
        "steps": [],
        "conversation_id": conversation_id or "",
        "retrieved_chunks": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "elapsed_ms": 0,
        "messages": [],
        "warning": "",
    }


class Agent:
    """AI Agent - 支持 RAG 检索、工具调用和多轮对话"""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 system_prompt: str | None = None, retriever: Retriever | None = None,
                 provider_config: dict | None = None):
        self.provider_config = provider_config
        self.system_prompt = system_prompt or build_agent_system_prompt()
        self.retriever = retriever
        if provider_config is not None:
            self.provider = provider or self._pc_get(provider_config, "name", "")
            self.model = model or self._pc_get(provider_config, "chat_model", "")
            if not self.provider:
                raise ValueError("provider_config 缺少 name")
            if not self.model:
                raise ValueError("provider_config 未配置聊天模型")
            self.anthropic_style = bool(self._pc_get(provider_config, "anthropic_style", False))
            self._init_dynamic_client(provider_config)
            return

        self.provider = provider or DEFAULT_PROVIDER
        config = LLM_CONFIGS.get(self.provider)
        if not config:
            raise ValueError(f"未知供应商: {self.provider}，可选: {list(LLM_CONFIGS.keys())}")
        self.model = model or config.get("chat_model", "")
        self.anthropic_style = self.provider == "anthropic"
        self._init_client(config)

    @staticmethod
    def _pc_get(config, key: str, default=None):
        """从 ProviderConfig 模型或 dict 中取值"""
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _init_client(self, config: dict):
        """初始化 LLM 客户端"""
        api_key = config["api_key"]
        if not api_key:
            raise ValueError(f"请设置 {self.provider.upper()}_API_KEY 环境变量")

        if self.anthropic_style:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key, base_url=config["base_url"], timeout=120)

    def _init_dynamic_client(self, config):
        """初始化动态供应商客户端，base_url/api_key 全部来自请求"""
        api_key = self._pc_get(config, "api_key", "") or ""
        base_url = self._pc_get(config, "base_url", "") or ""
        if not api_key:
            raise ValueError(f"请设置供应商 {self.provider} 的 API Key")
        if not base_url:
            raise ValueError(f"供应商 {self.provider} 未配置 Base URL")

        if self.anthropic_style:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120)

    async def chat(self, message: str, history: list[dict] | None = None,
                   conversation_id: str | None = None) -> dict:
        """Agent 对话入口，返回完整结果"""
        return await self._run_agent(message, history, conversation_id=conversation_id)

    async def stream_chat(self, message: str, history: list[dict] | None = None,
                          conversation_id: str | None = None) -> AsyncIterator[dict]:
        """Agent 流式对话入口，逐事件产出 SSE 数据"""
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def emit(event: dict) -> None:
            await queue.put(event)

        task = asyncio.create_task(
            self._run_agent(message, history, conversation_id=conversation_id, emit=emit)
        )
        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "result":
                    break
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run_agent(self, message: str, history: list[dict] | None = None,
                         conversation_id: str | None = None, emit: Any | None = None) -> dict:
        """ReAct 主循环；emit 为可选的异步事件回调"""
        start = time.time()
        messages: list[dict] = []
        steps: list[dict] = []
        conv_id = conversation_id or uuid4().hex[:12]
        retrieved_chunks: list[dict] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        warning = ""

        if history:
            messages = history.copy()
        messages.append({"role": "user", "content": message})

        max_iterations = 5
        iteration = 0
        loop_exhausted = False

        async def emit_event(event: dict):
            if emit:
                await emit(event)

        await emit_event({"type": "start", "conversation_id": conv_id})

        try:
            while iteration < max_iterations:
                iteration += 1

                async def on_token(token: str):
                    await emit_event({"type": "token", "content": token})

                response, usage = await self._llm_call(
                    messages,
                    tools=ALL_TOOLS,
                    on_token=on_token if emit else None,
                )
                self._merge_usage(total_usage, usage)

                msg = response["message"]
                messages.append(msg)

                if msg.get("content"):
                    step = {"step_type": "thought", "content": msg["content"]}
                    steps.append(step)
                    await emit_event({"type": "thought", "content": msg["content"]})

                tool_calls = msg.get("tool_calls", [])
                if not tool_calls:
                    break

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            raise TypeError("工具参数必须是 JSON 对象")
                    except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                        error_text = f"工具参数 JSON 解析失败: {parse_error}。原始参数: {raw_args}。请修正后重试。"
                        steps.append({
                            "step_type": "action",
                            "content": f"调用工具: {tool_name}（参数解析失败）",
                            "tool_name": tool_name,
                            "tool_args": {},
                        })
                        await emit_event({
                            "type": "action",
                            "tool_name": tool_name,
                            "tool_args": {},
                            "error": error_text,
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": error_text,
                        })
                        continue

                    steps.append({
                        "step_type": "action",
                        "content": f"调用工具: {tool_name}",
                        "tool_name": tool_name,
                        "tool_args": args,
                    })
                    await emit_event({
                        "type": "action",
                        "tool_name": tool_name,
                        "tool_args": args,
                    })

                    try:
                        tool_result = await asyncio.to_thread(execute_tool, tool_name, args, self.retriever)
                    except Exception as e:
                        tool_result = {"success": False, "error": f"工具执行异常: {e}"}

                    steps.append({
                        "step_type": "observation",
                        "content": f"工具 {tool_name} 返回结果",
                        "tool_name": tool_name,
                        "tool_args": args,
                        "tool_result": json.dumps(tool_result, ensure_ascii=False)[:1000],
                    })
                    await emit_event({
                        "type": "observation",
                        "tool_name": tool_name,
                        "tool_result": json.dumps(tool_result, ensure_ascii=False)[:1000],
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

                    if tool_name == "query_documents" and tool_result.get("success"):
                        results = tool_result.get("results", [])
                        for r in results:
                            meta = r.get("metadata", {}) or {}
                            retrieved_chunks.append({
                                "id": r.get("chunk_id", ""),
                                "doc_id": r.get("doc_id", "") or meta.get("doc_id", ""),
                                "text": r.get("text", ""),
                                "source": r.get("filename", "") or meta.get("filename", ""),
                                "score": r.get("score", 0),
                                "metadata": meta,
                            })
                        self.system_prompt = RAG_SYSTEM_PROMPT + (
                            f"\n当前日期: {datetime.now().strftime('%Y年%m月%d日')}。"
                        )
                        if self.retriever:
                            context_text = self.retriever.format_context(results)
                        else:
                            context_lines = []
                            for i, r in enumerate(results, 1):
                                context_lines.append(
                                    f"[Source {i}] File: {r.get('filename', '未知')} "
                                    f"(relevance: {r.get('score', 0):.2f})\n{r.get('text', '')}"
                                )
                            context_text = "\n---\n".join(context_lines)
                        messages.append({
                            "role": "user",
                            "content": "以下是知识库检索结果，请优先基于这些内容回答并引用来源：\n\n" + context_text,
                        })
                        await emit_event({"type": "context", "content": context_text[:2000]})

                if iteration >= max_iterations:
                    loop_exhausted = True

            final_answer = ""
            if messages and messages[-1]["role"] == "assistant":
                final_answer = messages[-1].get("content", "")

            if loop_exhausted and (not final_answer or messages[-1].get("tool_calls")):
                final_answer = f"尝试了 {max_iterations} 轮仍未完成回答，请重试或换个问法。"
                warning = f"ReAct 循环达到最大轮数 {max_iterations}"

            elapsed = (time.time() - start) * 1000

            result = {
                "success": True,
                "answer": final_answer,
                "steps": steps,
                "conversation_id": conv_id,
                "retrieved_chunks": retrieved_chunks,
                "usage": total_usage,
                "elapsed_ms": round(elapsed, 2),
                "messages": messages,
                "warning": warning,
            }
            await emit_event({"type": "answer", "content": final_answer})
            await emit_event({
                "type": "done",
                "conversation_id": conv_id,
                "steps": steps,
                "retrieved_chunks": retrieved_chunks,
                "usage": total_usage,
                "elapsed_ms": round(elapsed, 2),
                "warning": warning,
            })
            await emit_event({"type": "result", "result": result})
            return result

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            error_result = {
                "success": False,
                "answer": f"Agent 处理出错: {e!s}",
                "steps": steps,
                "conversation_id": conv_id,
                "retrieved_chunks": retrieved_chunks,
                "usage": total_usage,
                "elapsed_ms": round(elapsed, 2),
                "error": traceback.format_exc(),
                "messages": messages,
                "warning": warning,
            }
            await emit_event({"type": "error", "error": str(e)})
            await emit_event({"type": "result", "result": error_result})
            return error_result

    async def _llm_call(self, messages: list[dict], tools: list[dict] | None = None,
                        on_token: Any | None = None):
        """调用 LLM (兼容 OpenAI 和 Anthropic)"""
        if self.anthropic_style:
            return await self._anthropic_call(messages, tools)
        return await self._openai_call(messages, tools, on_token=on_token)

    async def _openai_call(self, messages: list[dict], tools: list[dict] | None = None,
                           on_token: Any | None = None):
        """调用 OpenAI 兼容 API，支持真实流式输出"""
        api_messages = [{"role": "system", "content": self.system_prompt}]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id", ""),
                    "content": content,
                })
            elif role == "assistant" and m.get("tool_calls"):
                api_messages.append({
                    "role": "assistant",
                    "content": m.get("content", ""),
                    "tool_calls": m["tool_calls"],
                })
            else:
                api_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools

        streaming = on_token is not None
        if streaming:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

        resp = await self.client.chat.completions.create(**kwargs)

        if not streaming:
            choice = resp.choices[0]
            msg = choice.message
            result = {"message": {"role": "assistant", "content": msg.content or ""}}
            if msg.tool_calls:
                result["message"]["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }
            return result, usage

        content_parts = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async for chunk in resp:
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue
            if delta.content:
                content_parts.append(delta.content)
                await on_token(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            acc["name"] += tc.function.name
                        if tc.function.arguments:
                            acc["arguments"] += tc.function.arguments

        message = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls_acc:
            ordered = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            message["tool_calls"] = [
                {
                    "id": acc["id"],
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                }
                for acc in ordered
            ]
        return {"message": message}, usage

    async def _anthropic_call(self, messages: list[dict], tools: list[dict] | None = None):
        """调用 Anthropic Claude API"""
        api_messages = []
        system_text = self.system_prompt

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_text = content
            elif role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": content,
                        }
                    ],
                })
            elif role == "assistant" and m.get("tool_calls"):
                content_blocks = [{"type": "text", "text": m.get("content", "")}]
                for tc in m["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {"_parse_error": tc["function"]["arguments"]}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                api_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "system": system_text,
            "max_tokens": 4096,
        }

        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", {})
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            kwargs["tools"] = anthropic_tools

        msg = await self.client.messages.create(**kwargs)

        result = {"message": {"role": "assistant", "content": ""}}
        tool_calls = []

        for block in msg.content:
            if block.type == "text":
                result["message"]["content"] = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        if tool_calls:
            result["message"]["tool_calls"] = tool_calls

        usage = {
            "prompt_tokens": msg.usage.input_tokens,
            "completion_tokens": msg.usage.output_tokens,
            "total_tokens": msg.usage.input_tokens + msg.usage.output_tokens,
        }

        return result, usage

    def _merge_usage(self, total: dict, add: dict):
        """合并 Token 用量"""
        for key in total:
            total[key] = total.get(key, 0) + add.get(key, 0)