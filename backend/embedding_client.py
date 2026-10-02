# Embedding 客户端 - 支持多供应商与动态自定义配置
import time

from backend.config import (
    EMBEDDING_MAX_BATCH,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_PROVIDER,
    EMBEDDING_TIMEOUT,
    LLM_CONFIGS,
)


class EmbeddingClient:
    """Embedding 向量生成客户端"""

    def __init__(self, provider: str | None = None, api_key: str | None = None,
                 provider_config=None):
        self.provider_config = provider_config
        if provider_config is not None:
            self.provider = provider or self._pc_get(provider_config, "name", "")
            self.model = self._pc_get(provider_config, "embedding_model", "") or ""
            self.dimensions = self._pc_get(provider_config, "embedding_dimensions", None)
            if not self.provider:
                raise ValueError("provider_config 缺少 name")
            if not self.model:
                raise ValueError(f"供应商 {self.provider} 不支持 Embedding")
            self._init_dynamic_client(provider_config, api_key)
            return

        self.provider = provider or EMBEDDING_PROVIDER
        config = LLM_CONFIGS.get(self.provider)
        if not config:
            raise ValueError(f"未知供应商: {self.provider}，可选: {list(LLM_CONFIGS.keys())}")

        model = config.get("embedding_model")
        if not model:
            raise ValueError(
                f"供应商 {self.provider} 未配置 Embedding 模型"
                f"（请设置 {self.provider.upper()}_EMBEDDING_MODEL，或在网页供应商管理中指定）"
            )

        self.model = model
        self.dimensions = None
        self._init_client(config, api_key)

    @staticmethod
    def _pc_get(config, key: str, default=None):
        """从 ProviderConfig 模型或 dict 中取值"""
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _init_client(self, config: dict, api_key: str | None = None):
        """初始化 OpenAI 兼容客户端"""
        from openai import OpenAI
        key = api_key or config["api_key"]
        if not key:
            raise ValueError(f"请设置 {self.provider.upper()}_API_KEY 环境变量")
        self.client = OpenAI(
            api_key=key,
            base_url=config["base_url"],
            timeout=EMBEDDING_TIMEOUT,
            max_retries=EMBEDDING_MAX_RETRIES,
        )

    def _init_dynamic_client(self, config, api_key: str | None = None):
        """初始化动态供应商客户端，base_url/api_key 全部来自请求"""
        from openai import OpenAI
        key = api_key or self._pc_get(config, "api_key", "") or ""
        base_url = self._pc_get(config, "base_url", "") or ""
        if not key:
            raise ValueError(f"请设置供应商 {self.provider} 的 API Key")
        if not base_url:
            raise ValueError(f"供应商 {self.provider} 未配置 Base URL")
        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=EMBEDDING_TIMEOUT,
            max_retries=EMBEDDING_MAX_RETRIES,
        )

    def _call_with_retry(self, fn):
        """带退避重试的 Embedding 调用"""
        last_error = None
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < EMBEDDING_MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise last_error

    def embed(self, text: str):
        """生成单个文本的向量"""
        resp = self._call_with_retry(lambda: self.client.embeddings.create(model=self.model, input=text))
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "total_tokens": resp.usage.total_tokens}
        return resp.data[0].embedding, usage

    def embed_batch(self, texts: list[str]):
        """批量生成向量，自动分批并带重试"""
        embeddings = []
        usage = {"prompt_tokens": 0, "total_tokens": 0}
        for i in range(0, len(texts), EMBEDDING_MAX_BATCH):
            batch = texts[i:i + EMBEDDING_MAX_BATCH]
            resp = self._call_with_retry(lambda b=batch: self.client.embeddings.create(model=self.model, input=b))
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            embeddings.extend(d.embedding for d in sorted_data)
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["total_tokens"] += resp.usage.total_tokens
        return embeddings, usage

    def get_dimensions(self) -> int:
        """返回向量维度；未显式配置时通过一次轻量 embedding 探测"""
        if self.dimensions:
            return int(self.dimensions)
        embedding, _usage = self.embed("ping")
        self.dimensions = len(embedding)
        return self.dimensions

# 修改记录：
#   2026-10-01 Embedding 供应商改用 EMBEDDING_PROVIDER（与聊天 LLM_PROVIDER 解耦，
#              修复 deepseek 无嵌入接口导致知识库初始化失败）
#   2026-10-01 模型名不再内置，未配置时提示设置 *_EMBEDDING_MODEL（区分"未配置"与"不支持"）
