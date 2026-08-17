# Pydantic 数据模型
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """文档元数据"""
    id: str = Field(description="文档唯一标识")
    filename: str = Field(description="原始文件名")
    source: str = Field(description="文件路径")
    doc_type: str = Field(description="文档类型: pdf/md/txt")
    uploaded_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    chunk_count: int = Field(0, description="切分后片段数")
    content_preview: str = Field("", description="内容预览")


class DocumentChunk(BaseModel):
    """文档片段"""
    id: str = Field(description="片段唯一 ID")
    doc_id: str = Field(description="所属文档 ID")
    text: str = Field(description="片段文本")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class RetrievedChunk(BaseModel):
    """检索到的文档片段 (含相似度)"""
    id: str = Field(description="片段 ID")
    doc_id: str = Field(description="所属文档 ID")
    text: str = Field(description="片段文本")
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(0.0, description="相似度得分")
    source: str = Field("", description="来源文件名")


class ProviderConfig(BaseModel):
    """网页端自定义 OpenAI 兼容供应商配置"""

    name: str = Field(description="唯一标识，如 kimi、zhipu、my-gateway")
    display_name: str = Field("", description="前端显示名")
    base_url: str = Field(description="OpenAI 兼容 Base URL")
    api_key: str = Field("", description="请求级 Key，日志必须脱敏")
    chat_model: str = Field("", description="聊天模型")
    embedding_model: str = Field("", description="Embedding 模型，留空表示不支持")
    embedding_dimensions: int | None = Field(None, description="可选，用于维度校验")
    supports_chat: bool = Field(True, description="是否支持聊天")
    supports_embedding: bool = Field(False, description="是否支持 Embedding")
    anthropic_style: bool = Field(False, description="是否走 Anthropic 原生协议")
    native_json_mode: bool = Field(True, description="是否原生支持 response_format")


class AgentRequest(BaseModel):
    """Agent 对话请求"""
    message: str = Field(description="用户消息")
    conversation_id: str | None = Field(None, description="会话 ID (续对话)")
    stream: bool = Field(False, description="是否流式响应")
    provider: str | None = Field(None, description="供应商")
    model: str | None = Field(None, description="模型名称")
    provider_config: ProviderConfig | None = Field(None, description="网页自定义供应商配置")


class AgentStep(BaseModel):
    """Agent 单步记录"""
    step_type: str = Field(description="步骤类型: thought/action/observation/answer")
    content: str = Field(description="步骤内容")
    tool_name: str | None = Field(None, description="工具名称")
    tool_args: dict[str, Any] | None = Field(None, description="工具参数")
    tool_result: str | None = Field(None, description="工具结果")


class AgentResponse(BaseModel):
    """Agent 对话响应"""
    success: bool = True
    answer: str = Field(description="最终回答")
    steps: list[AgentStep] = Field(default_factory=list, description="推理步骤")
    conversation_id: str = Field(description="会话 ID")
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list, description="检索到的片段")
    usage: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
    provider: str = Field("", description="实际使用的供应商")
    model: str = Field("", description="实际使用的模型")
    warning: str = Field("", description="警告信息")
    error: str = Field("", description="错误信息")


class RAGQueryRequest(BaseModel):
    """RAG 检索请求"""
    query: str = Field(description="检索查询")
    top_k: int = Field(5, description="返回结果数", ge=1, le=20)
    doc_id: str | None = Field(None, description="限定文档 ID")
    provider_config: ProviderConfig | None = Field(None, description="网页自定义 Embedding 供应商配置")


class RAGQueryResponse(BaseModel):
    """RAG 检索响应"""
    success: bool = True
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_chunks: int = 0
    elapsed_ms: float = 0.0
    error: str = Field("", description="错误信息")


class ToolExecuteRequest(BaseModel):
    """工具执行请求"""
    tool_name: str = Field(description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class ToolExecuteResponse(BaseModel):
    """工具执行响应"""
    success: bool = True
    result: Any = Field(description="执行结果")
    tool_name: str = Field(description="工具名称")
    error: str = Field("", description="错误信息")


class AgentLog(BaseModel):
    """Agent 日志"""
    id: int = 0
    conversation_id: str = ""
    question: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str = ""
    model: str = ""
    total_tokens: int = 0
    elapsed_ms: float = 0.0
    timestamp: str = ""
    success: bool = True
    error: str = ""


class LogListResponse(BaseModel):
    """日志列表响应"""
    success: bool = True
    data: list[AgentLog] = Field(default_factory=list)
    count: int = 0


class ProviderTestRequest(BaseModel):
    """供应商连接测试请求"""

    provider_config: ProviderConfig = Field(description="待测试的供应商配置")
    test_type: str = Field("chat", description="chat / embedding")
    sample_message: str = Field("ping", description="chat 测试消息")


class VectorStoreRebuildRequest(BaseModel):
    """向量库重建请求"""

    confirm: bool = Field(False, description="必须为 true 才执行重建")