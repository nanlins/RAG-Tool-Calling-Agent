# FastAPI 应用入口 - RAG + Tool Calling Agent
import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from backend.agent import Agent
from backend.cache import (
    check_rate_limit,
    get_conversation,
    get_redis_info,
    is_available,
    save_conversation,
)
from backend.config import (
    APP_API_TOKEN,
    AUTH_DISABLED,
    CORS_ORIGINS,
    DEFAULT_PROVIDER,
    DOCUMENTS_DIR,
    HOST,
    LLM_CONFIGS,
    MAX_UPLOAD_BYTES,
    PORT,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_MAX,
    RATE_LIMIT_TRUST_XFF,
    RATE_LIMIT_WINDOW,
    SCORE_THRESHOLD,
)
from backend.database import (
    delete_document_record,
    get_all_documents,
    get_document_record,
    init_db,
    save_document,
)
from backend.document_loader import load_document
from backend.embedding_client import EmbeddingClient
from backend.logger import get_logs, log_agent_call
from backend.models import (
    AgentLog,
    AgentRequest,
    AgentResponse,
    AgentStep,
    LogListResponse,
    ProviderConfig,
    ProviderTestRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunk,
    ToolExecuteRequest,
    ToolExecuteResponse,
    VectorStoreRebuildRequest,
)
from backend.providers import test_provider
from backend.retriever import Retriever
from backend.text_splitter import split_document
from backend.tools import TOOL_DEFINITIONS, execute_tool
from backend.vector_store import VectorStore

logger = logging.getLogger("rag-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}
SSE_EVENT_TYPES = {"start", "token", "thought", "action", "observation", "context", "answer", "done", "error"}

# 全局实例
vector_store = None
retriever = None
embedding_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    global vector_store, retriever, embedding_client

    init_db()

    if not APP_API_TOKEN and not AUTH_DISABLED:
        raise RuntimeError(
            "安全配置缺失：未设置 APP_API_TOKEN，服务拒绝启动。"
            "本地演示请设置 APP_API_TOKEN=local-demo，公网部署请使用强随机 Token。"
        )

    # 初始化向量存储和检索器
    try:
        vector_store = VectorStore()
        embedding_client = EmbeddingClient()
        retriever = Retriever(vector_store, embedding_client)
        stats = vector_store.get_statistics()
        logger.info(
            "[RAG Agent] 初始化完成: 供应商=%s, 向量库=%s, %s 片段, %s 文档",
            DEFAULT_PROVIDER,
            vector_store.persist_dir,
            stats["total_chunks"],
            stats["total_documents"],
        )
    except Exception as e:
        logger.warning("[RAG Agent] 知识库组件初始化失败，上传/检索功能暂不可用: %s", e)

    yield


app = FastAPI(title="RAG + Tool Calling Agent", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_ip(request: Request) -> str:
    """返回真实客户端 IP；仅当显式配置可信代理时才信任 X-Forwarded-For"""
    if RATE_LIMIT_TRUST_XFF:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """启用后按客户端 IP 对 /v1/* 接口限流，Redis 不可用时使用进程内计数兜底"""
    if RATE_LIMIT_ENABLED and request.url.path.startswith("/v1/"):
        client_ip = _client_ip(request)
        allowed = await asyncio.to_thread(
            check_rate_limit, client_ip, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"success": False, "error": "请求过于频繁，请稍后再试", "code": 429},
            )
    return await call_next(request)


def _error(status_code: int, message: str, **extra: Any) -> JSONResponse:
    """统一错误响应体"""
    body = {"success": False, "error": message, "code": status_code}
    body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """把 HTTPException 统一为项目的 JSON 错误体"""
    return _error(exc.status_code, str(exc.detail))


async def require_token(
    authorization: str | None = Header(None),
    x_api_token: str | None = Header(None),
) -> None:
    """API Token 鉴权；AUTH_DISABLED=true 时显式关闭，其余环境必须配置"""
    if not APP_API_TOKEN:
        if AUTH_DISABLED:
            return
        raise HTTPException(
            status_code=503,
            detail="服务未配置 APP_API_TOKEN，拒绝开放接口；请设置 APP_API_TOKEN",
        )
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_token:
        token = x_api_token.strip()
    if token and hmac.compare_digest(token.encode("utf-8"), APP_API_TOKEN.encode("utf-8")):
        return
    raise HTTPException(status_code=401, detail="未授权，请提供 APP_API_TOKEN")


def _create_agent(provider: str | None, model: str | None, provider_config=None) -> Agent:
    if provider_config is not None:
        try:
            return Agent(provider=provider, model=model, retriever=retriever, provider_config=provider_config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    provider_name = provider or DEFAULT_PROVIDER
    config = LLM_CONFIGS.get(provider_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"未知供应商: {provider_name}")
    if not config.get("api_key"):
        raise HTTPException(status_code=503, detail=f"供应商 {provider_name} 未配置 API Key")
    return Agent(provider=provider_name, model=model, retriever=retriever)


def _redact_api_keys(text: str, provider_config=None) -> str:
    """从错误/回答文本中移除请求级 API Key"""
    if not text or provider_config is None:
        return text
    key = getattr(provider_config, "api_key", "") or ""
    if key:
        text = text.replace(key, "***")
    return text


def _embedding_mismatch(embed_client, provider_config=None) -> str | None:
    """自定义 Embedding 配置与已建索引不一致时返回原因，阻止静默错配"""
    if vector_store is None:
        return None
    try:
        profile = vector_store.get_profile()
    except Exception:
        return None
    if not profile.get("total_chunks"):
        return None
    meta_model = profile.get("embedding_model", "")
    if not meta_model:
        return "向量库缺少 Embedding 配置信息，请先重建索引"
    if meta_model != embed_client.model:
        return (
            f"向量库由 Embedding 模型 {meta_model} 生成，当前查询使用 {embed_client.model}，"
            "请先重建索引"
        )
    meta_dims = profile.get("embedding_dimensions")
    dims = provider_config.embedding_dimensions if provider_config is not None else None
    if dims is not None and meta_dims is not None and int(dims) != int(meta_dims):
        return f"向量维度不一致：索引维度 {meta_dims}，当前维度 {dims}，请先重建索引"
    return None


def _write_exclusive(path: Path, content: bytes) -> None:
    with open(path, "xb") as f:
        f.write(content)


def _safe_unlink(path: Path) -> None:
    path.unlink(missing_ok=True)


def _load_and_split(file_path: Path, original_filename: str):
    """在后台线程中解析并切分文档"""
    doc = load_document(str(file_path))
    doc.filename = original_filename
    doc.title = Path(original_filename).stem
    chunks = split_document(doc, strategy="recursive")
    doc.chunk_count = len(chunks)
    return doc, chunks


def _build_agent_response(result: dict[str, Any], agent: Agent) -> AgentResponse:
    """把 Agent 内部结果映射为 API 响应"""
    steps = []
    for s in result.get("steps", []):
        steps.append(AgentStep(
            step_type=s.get("step_type", ""),
            content=s.get("content", ""),
            tool_name=s.get("tool_name"),
            tool_args=s.get("tool_args"),
            tool_result=s.get("tool_result"),
        ))

    retrieved = []
    for c in result.get("retrieved_chunks", []):
        meta = c.get("metadata", {}) or {}
        retrieved.append(RetrievedChunk(
            id=c.get("id", ""),
            doc_id=c.get("doc_id", "") or meta.get("doc_id", ""),
            text=c.get("text", ""),
            source=c.get("source", "") or meta.get("filename", ""),
            score=c.get("score", 0),
            metadata=meta,
        ))

    return AgentResponse(
        success=bool(result.get("success", True)),
        answer=result.get("answer", ""),
        steps=steps,
        conversation_id=result.get("conversation_id", ""),
        retrieved_chunks=retrieved,
        usage=result.get("usage", {}),
        elapsed_ms=result.get("elapsed_ms", 0),
        provider=agent.provider,
        model=agent.model,
        warning=result.get("warning", ""),
        error=result.get("error", ""),
    )


def _persist_agent_result(result: dict[str, Any], conv_id: str, question: str, agent: Agent) -> None:
    """保存会话并记录日志；Redis 不可用时由缓存层自动回退 SQLite"""
    save_conversation(conv_id, result.get("messages", []), ttl=3600)
    log_agent_call(
        conversation_id=conv_id,
        question=question,
        steps=result.get("steps", []),
        final_answer=result.get("answer", ""),
        model=f"{agent.provider}/{agent.model}",
        total_tokens=result.get("usage", {}).get("total_tokens", 0),
        elapsed_ms=result.get("elapsed_ms", 0),
        success=result.get("success", True),
        error=result.get("error", ""),
    )


# ===== 前端页面 =====

@app.get("/", include_in_schema=False)
async def serve_frontend():
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    try:
        content = await asyncio.to_thread(frontend_path.read_text, encoding="utf-8")
    except OSError:
        content = "<h1>RAG + Tool Calling Agent</h1><p>前端页面未找到</p>"
    return HTMLResponse(content=content)


# ===== 健康检查 =====

@app.get("/health")
async def health_check():
    status = "ok"
    details: dict[str, Any] = {}
    if vector_store:
        try:
            details["vector_store"] = await asyncio.to_thread(vector_store.get_statistics)
        except Exception as e:
            details["vector_store"] = {"error": str(e)}
    if embedding_client is None:
        status = "degraded"
        details["embedding"] = "unavailable"
    payload = {"status": status, "service": "RAG + Tool Calling Agent", "details": details}
    if status == "ok":
        return payload
    return JSONResponse(status_code=503, content=payload)


@app.get("/v1/models", dependencies=[Depends(require_token)])
async def list_models():
    """返回可用模型列表"""
    models = {}
    for provider, config in LLM_CONFIGS.items():
        if config.get("api_key") and config.get("chat_model"):
            models[provider] = {
                "chat_model": config.get("chat_model", ""),
                "embedding_model": config.get("embedding_model", ""),
            }
    return {"success": True, "data": models}


# ===== 供应商测试与向量库管理 =====

@app.post("/v1/providers/test", dependencies=[Depends(require_token)])
async def provider_test(request: ProviderTestRequest):
    """测试网页添加的供应商连接（chat / embedding）"""
    try:
        data = await test_provider(request.provider_config, request.test_type, request.sample_message)
    except ValueError as e:
        return _error(400, str(e))
    except Exception as e:
        message = str(e)
        low = message.lower()
        if "api key" in low or "unauthorized" in low or "invalid" in low or "401" in message:
            return _error(401, "供应商连接失败（鉴权错误），请检查 API Key")
        return _error(502, f"供应商连接失败: {_redact_api_keys(message, request.provider_config)}")
    return {"success": True, "data": data}


@app.get("/v1/vector-store/profile", dependencies=[Depends(require_token)])
async def vector_store_profile():
    """返回当前向量库 Embedding profile 与统计"""
    if vector_store is None:
        return _error(503, "向量库未初始化")
    try:
        profile = await asyncio.to_thread(vector_store.get_profile)
    except Exception as e:
        return _error(500, f"读取向量库 profile 失败: {e}")
    return {"success": True, "data": profile}


@app.post("/v1/vector-store/rebuild", dependencies=[Depends(require_token)])
async def vector_store_rebuild(request: VectorStoreRebuildRequest):
    """清空向量库并重建 metadata；必须显式 confirm=true"""
    if not request.confirm:
        return _error(400, "确认重建请携带 confirm=true")
    if vector_store is None:
        return _error(503, "向量库未初始化")
    try:
        await asyncio.to_thread(vector_store.clear_all)
        profile = await asyncio.to_thread(vector_store.get_profile)
    except Exception as e:
        return _error(500, f"重建向量库失败: {e}")
    return {
        "success": True,
        "data": {
            "message": "向量库已清空，下次上传将写入新的 Embedding 配置",
            "profile": profile,
        },
    }


# ===== 文档管理 =====

@app.post("/v1/documents/upload", dependencies=[Depends(require_token)])
async def upload_document(
    file: Annotated[UploadFile, File()],
    provider_config_json: Annotated[str | None, Form()] = None,
):
    """上传并处理文档；使用 UUID 服务端文件名，避免路径穿越和覆盖"""
    upload_provider_config = None
    dyn_embedding = None
    if provider_config_json:
        try:
            upload_provider_config = ProviderConfig.model_validate_json(provider_config_json)
        except Exception:
            return _error(400, "provider_config_json 格式错误或字段缺失")
        try:
            dyn_embedding = EmbeddingClient(provider_config=upload_provider_config)
        except ValueError as e:
            return _error(400, str(e))
    if not file.filename:
        return _error(400, "文件名为空")

    original_filename = os.path.basename(file.filename.replace("\\", "/"))
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error(400, "仅支持 PDF/Markdown/TXT 格式")

    if vector_store is None or retriever is None:
        return _error(503, "知识库组件未初始化")
    if embedding_client is None and not provider_config_json:
        return _error(503, "Embedding 服务未初始化，请配置 API Key 或使用网页供应商")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        return _error(413, f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制")

    server_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = DOCUMENTS_DIR / server_filename
    try:
        await asyncio.to_thread(_write_exclusive, file_path, content)
    except FileExistsError:
        return _error(500, "服务器文件命名冲突，请重试")

    try:
        doc, chunks = await asyncio.to_thread(_load_and_split, file_path, original_filename)
        texts = [c["text"] for c in chunks]
        embeddings = []
        ingest_error = ""
        embed = dyn_embedding or embedding_client
        if texts:
            try:
                embeddings, _usage = await asyncio.to_thread(embed.embed_batch, texts)
            except Exception as e:
                ingest_error = str(e)
                logger.warning("Embedding 失败: %s", ingest_error)

        added = 0
        if embeddings:
            try:
                added = await asyncio.to_thread(
                    vector_store.add_documents,
                    chunks,
                    embeddings,
                    embedding_provider=embed.provider,
                    embedding_model=embed.model,
                    embedding_dimensions=(
                        upload_provider_config.embedding_dimensions
                        if upload_provider_config else None
                    ),
                )
            except ValueError as e:
                await asyncio.to_thread(_safe_unlink, file_path)
                return _error(409, str(e))
            except Exception as e:
                ingest_error = str(e)
                logger.warning("向量写入失败: %s", ingest_error)

        if not chunks:
            status = "empty"
            ingest_error = ingest_error or "文档内容为空，未生成片段"
        elif not embeddings:
            status = "failed"
        elif added == len(chunks):
            status = "indexed"
        elif added > 0:
            status = "partial"
        else:
            status = "failed"
            ingest_error = ingest_error or "向量写入失败"

        doc_data = doc.to_dict()
        doc_data["server_filename"] = server_filename
        doc_data["chunk_count"] = len(chunks)
        doc_data["chunks_added"] = added
        doc_data["ingest_status"] = status
        await asyncio.to_thread(save_document, doc_data)

        if status in ("failed", "empty"):
            await asyncio.to_thread(delete_document_record, doc.id)
            await asyncio.to_thread(_safe_unlink, file_path)
            return _error(
                502 if status == "failed" else 422,
                ingest_error or f"文档处理状态: {status}",
                data={
                    "id": doc.id,
                    "filename": original_filename,
                    "chunk_count": len(chunks),
                    "chunks_added": added,
                    "ingest_status": status,
                },
            )

        if status == "partial":
            missing = len(chunks) - added
            return JSONResponse(
                status_code=202,
                content={
                    "success": True,
                    "warning": ingest_error or f"部分入库成功：{missing}/{len(chunks)} 个片段未写入",
                    "data": {
                        "id": doc.id,
                        "filename": original_filename,
                        "doc_type": doc.doc_type,
                        "title": doc.title,
                        "content_length": len(doc.content),
                        "chunk_count": len(chunks),
                        "chunks_added": added,
                        "ingest_status": status,
                        "uploaded_at": doc.uploaded_at,
                    },
                },
            )

        return {
            "success": True,
            "warning": ingest_error or "",
            "data": {
                "id": doc.id,
                "filename": original_filename,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "content_length": len(doc.content),
                "chunk_count": len(chunks),
                "chunks_added": added,
                "ingest_status": status,
                "uploaded_at": doc.uploaded_at,
            },
        }
    except Exception as e:
        logger.exception("文档处理失败")
        await asyncio.to_thread(_safe_unlink, file_path)
        return _error(500, f"文档处理失败: {e}")


@app.get("/v1/documents", dependencies=[Depends(require_token)])
async def list_documents():
    """获取文档列表"""
    docs = await asyncio.to_thread(get_all_documents)
    return {"success": True, "data": docs, "count": len(docs)}


@app.delete("/v1/documents/{doc_id}", dependencies=[Depends(require_token)])
async def delete_document(doc_id: str):
    """删除文档、向量与服务器文件"""
    record = await asyncio.to_thread(get_document_record, doc_id)
    server_filename = (record or {}).get("server_filename", "")

    deleted = 0
    if vector_store:
        deleted = await asyncio.to_thread(vector_store.delete_document, doc_id)
    record_deleted = await asyncio.to_thread(delete_document_record, doc_id)

    if server_filename:
        safe_path = (DOCUMENTS_DIR / Path(server_filename).name).resolve()
        if safe_path.parent == DOCUMENTS_DIR.resolve() and safe_path.is_file():
            await asyncio.to_thread(_safe_unlink, safe_path)

    if not record_deleted and deleted == 0:
        return _error(404, "文档不存在")
    return {"success": True, "data": {"deleted_chunks": deleted, "doc_id": doc_id}}


# ===== RAG 检索 =====

@app.post("/v1/rag/query", response_model=RAGQueryResponse, dependencies=[Depends(require_token)])
async def rag_query(request: RAGQueryRequest):
    """RAG 知识库检索"""
    if retriever is None:
        return _error(503, "检索器未初始化")

    start = time.time()
    try:
        dyn_client = None
        if request.provider_config is not None:
            dyn_client = EmbeddingClient(provider_config=request.provider_config)
            mismatch = _embedding_mismatch(dyn_client, request.provider_config)
            if mismatch:
                return _error(409, mismatch)
        retrieve_kwargs = {
            "query": request.query,
            "top_k": request.top_k,
            "score_threshold": SCORE_THRESHOLD,
            "doc_id": request.doc_id,
        }
        if dyn_client is not None:
            retrieve_kwargs["embedding_client"] = dyn_client
        chunks, _elapsed, _usage = await asyncio.to_thread(
            retriever.retrieve,
            **retrieve_kwargs,
        )

        result_chunks = []
        for c in chunks:
            meta = c.get("metadata", {}) or {}
            result_chunks.append(RetrievedChunk(
                id=c.get("id", ""),
                doc_id=c.get("doc_id", "") or meta.get("doc_id", ""),
                text=c.get("text", ""),
                metadata=meta,
                score=c.get("hybrid_score", c.get("score", 0)),
                source=c.get("source", "") or meta.get("filename", ""),
            ))

        return RAGQueryResponse(
            success=True,
            chunks=result_chunks,
            total_chunks=len(result_chunks),
            elapsed_ms=round((time.time() - start) * 1000, 2),
        )
    except ValueError as e:
        return _error(400, str(e))
    except Exception as e:
        logger.exception("RAG 检索失败")
        return _error(503, f"RAG 检索失败: {e}")


# ===== Agent 对话 =====

@app.post("/v1/agent/chat", dependencies=[Depends(require_token)])
async def agent_chat(request: AgentRequest):
    """Agent 对话 (含工具调用)，会话 ID 由请求传入以维持多轮上下文"""
    if request.stream:
        return await agent_chat_stream(request)
    try:
        agent = _create_agent(request.provider, request.model, request.provider_config)
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)

    conv_id = request.conversation_id or None
    history = await asyncio.to_thread(get_conversation, conv_id) if conv_id else None

    result = await agent.chat(
        message=request.message,
        history=history,
        conversation_id=conv_id,
    )
    if request.provider_config:
        result["error"] = _redact_api_keys(result.get("error", ""), request.provider_config)
        result["answer"] = _redact_api_keys(result.get("answer", ""), request.provider_config)

    cid = result.get("conversation_id") or conv_id or ""
    await asyncio.to_thread(_persist_agent_result, result, cid, request.message, agent)

    response = _build_agent_response(result, agent)
    if not result.get("success"):
        return _error(
            502,
            result.get("error") or result.get("answer") or "Agent 调用失败",
            **response.model_dump(exclude={"success", "error"}),
        )
    return response


@app.post("/v1/agent/chat/stream", dependencies=[Depends(require_token)])
async def agent_chat_stream(request: AgentRequest):
    """Agent 流式对话，SSE 实时输出 token 与思考/工具过程"""
    try:
        agent = _create_agent(request.provider, request.model, request.provider_config)
        conv_id = request.conversation_id or None
        history = await asyncio.to_thread(get_conversation, conv_id) if conv_id else None
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)

    async def event_stream():
        try:
            async for event in agent.stream_chat(
                message=request.message,
                history=history,
                conversation_id=conv_id,
            ):
                event_type = event.get("type")
                if request.provider_config:
                    if event_type == "result":
                        event["result"]["error"] = _redact_api_keys(
                            event["result"].get("error", ""), request.provider_config
                        )
                        event["result"]["answer"] = _redact_api_keys(
                            event["result"].get("answer", ""), request.provider_config
                        )
                    elif event_type == "error":
                        event["error"] = _redact_api_keys(event.get("error", ""), request.provider_config)
                if event_type == "result":
                    result = event["result"]
                    cid = result.get("conversation_id") or conv_id or ""
                    await asyncio.to_thread(_persist_agent_result, result, cid, request.message, agent)
                    continue
                if event_type in SSE_EVENT_TYPES:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("流式对话失败")
            error_text = _redact_api_keys(str(e), request.provider_config)
            yield f"data: {json.dumps({'type': 'error', 'error': error_text}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ===== 工具执行 =====

@app.post("/v1/tools/execute", dependencies=[Depends(require_token)])
async def tool_execute(request: ToolExecuteRequest):
    """执行指定工具"""
    try:
        result = await asyncio.to_thread(execute_tool, request.tool_name, request.arguments, retriever)
    except Exception as e:
        logger.exception("工具执行失败")
        return _error(500, f"工具执行失败: {e}")

    error = result.get("error") or ""
    if not result.get("success"):
        if "未知工具" in error:
            return _error(400, error)
        if "未初始化" in error:
            return _error(503, error)
        return _error(422, error or "工具执行失败", result=result)
    return ToolExecuteResponse(
        success=result.get("success", True),
        result=result,
        tool_name=request.tool_name,
        error=error,
    )


@app.get("/v1/tools", dependencies=[Depends(require_token)])
async def list_tools():
    """获取可用工具列表"""
    tools = []
    for name, schema in TOOL_DEFINITIONS.items():
        tools.append({
            "name": name,
            "description": schema.get("function", {}).get("description", ""),
            "parameters": schema.get("function", {}).get("parameters", {}),
        })
    return {"success": True, "data": tools, "count": len(tools)}


# ===== Agent 日志 =====

@app.get("/v1/agent/logs", response_model=LogListResponse, dependencies=[Depends(require_token)])
async def agent_logs(limit: int = Query(50, ge=1, le=200)):
    """获取 Agent 调用日志，字段白名单由 Pydantic 响应模型控制"""
    logs = await asyncio.to_thread(get_logs, limit)
    return LogListResponse(data=[AgentLog(**log) for log in logs], count=len(logs))


# ===== Redis 状态 =====

@app.get("/v1/redis/info", dependencies=[Depends(require_token)])
async def redis_info():
    """获取 Redis 连接状态与信息"""
    return {"success": True, "data": await asyncio.to_thread(get_redis_info)}


@app.get("/v1/redis/health", dependencies=[Depends(require_token)])
async def redis_health():
    """Ping Redis 检查健康状态"""
    return {"success": True, "redis_available": await asyncio.to_thread(is_available)}


# ===== 向量库统计 =====

@app.get("/v1/stats", dependencies=[Depends(require_token)])
async def get_stats():
    """获取系统统计信息"""
    stats: dict[str, Any] = {"vector_store": {}, "documents": {}, "logs": {}}

    if vector_store:
        try:
            stats["vector_store"] = await asyncio.to_thread(vector_store.get_statistics)
        except Exception as e:
            stats["vector_store"] = {"error": str(e)}

    docs = await asyncio.to_thread(get_all_documents)
    stats["documents"] = {"total": len(docs), "list": docs}

    logs = await asyncio.to_thread(get_logs, 5)
    stats["logs"] = {"recent": len(logs)}

    return {"success": True, "data": stats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)