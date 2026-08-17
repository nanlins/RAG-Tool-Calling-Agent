# LLM 供应商连接测试 - OpenAI 兼容 / Anthropic 风格
import time


async def test_provider(config, test_type: str = "chat", sample_message: str = "ping"):
    """测试供应商连接：chat 或 embedding，返回模型、耗时与维度"""
    if not config.api_key:
        raise ValueError("未配置 API Key")
    if not config.base_url:
        raise ValueError("未配置 Base URL")

    if test_type == "chat":
        if not config.chat_model:
            raise ValueError("该供应商未配置聊天模型")
        start = time.time()
        if config.anthropic_style:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=config.api_key, base_url=config.base_url)
            resp = await client.messages.create(
                model=config.chat_model,
                messages=[{"role": "user", "content": sample_message}],
                max_tokens=16,
            )
            reply = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url, timeout=30)
            resp = await client.chat.completions.create(
                model=config.chat_model,
                messages=[{"role": "user", "content": sample_message}],
                max_tokens=16,
            )
            reply = (resp.choices[0].message.content or "")[:200]
        return {
            "ok": True,
            "provider": config.name,
            "model": config.chat_model,
            "reply": reply,
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }

    if test_type == "embedding":
        if not config.embedding_model:
            raise ValueError("该供应商未配置 Embedding 模型")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url, timeout=30)
        start = time.time()
        resp = await client.embeddings.create(model=config.embedding_model, input=sample_message)
        return {
            "ok": True,
            "provider": config.name,
            "model": config.embedding_model,
            "dimensions": len(resp.data[0].embedding),
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }

    raise ValueError(f"不支持的测试类型: {test_type}")
