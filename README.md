# RAG + Tool Calling Agent

[![CI](https://github.com/nanlins/RAG-Tool-Calling-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/nanlins/RAG-Tool-Calling-Agent/actions/workflows/ci.yml)

一个从 0 到 1 实现的 AI Agent 应用：支持文档上传、RAG 混合检索、ReAct 工具调用、多轮会话、流式输出和完整日志。

GitHub: https://github.com/nanlins/RAG-Tool-Calling-Agent

## 功能

- RAG：PDF/Markdown/TXT 解析、递归切分、Embedding、Chroma 向量存储、BM25 混合检索、来源引用
- Agent：ReAct 推理循环、calculator / get_current_time / web_search / query_documents 四个内置工具
- 多轮会话：Redis 缓存会话，Redis 不可用时自动回退 SQLite
- 流式输出：`/v1/agent/chat/stream` 逐 token SSE，前端实时展示思考、工具调用与回答
- 日志与统计：记录问题、步骤、工具结果、模型、Token 用量、耗时
- 安全：可选 API Token 鉴权、UUID 服务端文件名、CORS 白名单、前端安全 DOM 渲染

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 后端 | FastAPI + Pydantic | REST API 与请求/响应校验 |
| 向量检索 | Chroma + BM25 | 混合检索，支持中英文分词 |
| Embedding | Dashscope / OpenAI | 文本向量化 |
| 文档解析 | pymupdf | PDF 内容提取 |
| 文本切分 | langchain-text-splitters | 递归切分 |
| LLM | DeepSeek / 通义千问 / OpenAI / Claude | 多供应商兼容 |
| 会话缓存 | Redis + SQLite | Redis 可选，SQLite 兜底 |
| 前端 | 原生 HTML + JavaScript | 5 个功能页单页应用 |
| 测试 | pytest + pytest-asyncio | 60 个自动化测试 |

## 快速开始

```bash
pip install -r requirements.txt
```

复制并填写环境变量：

```bash
cp .env.example .env
```

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8081
```

打开 http://localhost:8081

Docker 方式：

```bash
docker compose up -d
```

## 环境变量

核心变量见 `.env.example`。**项目不内置任何模型名**，供应商与模型由用户显式配置（环境变量或网页「供应商管理」）。

- `LLM_PROVIDER`：聊天默认供应商，可填 `dashscope` / `deepseek` / `openai` / `anthropic`
- `EMBEDDING_PROVIDER`：Embedding 供应商（与 `LLM_PROVIDER` 解耦），可填 `dashscope` / `openai`
- `*_API_KEY`、`*_BASE_URL`、`*_CHAT_MODEL`、`*_EMBEDDING_MODEL`：各供应商的密钥/端点/模型，按需填写
- `APP_API_TOKEN`：未配置时服务拒绝启动；本地演示用 `local-demo`，公网部署必须设置强随机 Token
- `AUTH_DISABLED`：显式关闭鉴权，仅用于本地开发或自动化测试；生产环境必须为 `false`
- `REDIS_URL`、`REDIS_ENABLED`：会话缓存，Redis 不可用自动使用 SQLite
- `RATE_LIMIT_ENABLED`、`RATE_LIMIT_MAX`、`RATE_LIMIT_WINDOW`：可选限流；Redis 不可用时使用进程内计数兜底
- `RATE_LIMIT_TRUST_XFF`：启用后信任 `X-Forwarded-For` 的第一个 IP（仅在可信反向代理后开启）
- `CORS_ORIGINS`：允许访问的前端来源白名单

## 供应商配置示例

> RAG 链路需要 **两个能力**：聊天 LLM + Embedding。一个供应商可能只提供其一（如 DeepSeek 无 Embedding 接口），因此两者独立配置。

### 示例一：DeepSeek 聊天 + 阿里云 DashScope Embedding（推荐）

`.env`：

```env
# 聊天走 DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_CHAT_MODEL=deepseek-chat

# Embedding 走 DashScope
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-你的阿里云百炼密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3
```

### 示例二：全部用 DashScope（对话 + Embedding）

```env
LLM_PROVIDER=dashscope
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-你的阿里云百炼密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen-plus
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3
```

### 示例三：全部用 OpenAI

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-你的OpenAI密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 网页添加供应商（无需改 .env、无需重启）

在「供应商管理」页添加任意 OpenAI 兼容供应商（Kimi、智谱、DeepSeek、自定义网关等），保存后聊天/流式/RAG 检索/上传会自动携带该供应商配置。RAG 入库需要一个**配置了 Embedding 模型**的供应商。

> 各供应商能力差异：DeepSeek / Anthropic 仅对话（无 Embedding）；DashScope、OpenAI 同时支持对话与 Embedding。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 可用模型列表 |
| POST | `/v1/providers/test` | 测试网页添加的供应商连接（chat / embedding） |
| GET | `/v1/vector-store/profile` | 查看向量库 Embedding profile 与统计 |
| POST | `/v1/vector-store/rebuild` | 清空向量库并重建（需 confirm=true） |
| POST | `/v1/documents/upload` | 上传 PDF/MD/TXT，返回入库状态 |
| GET | `/v1/documents` | 文档列表 |
| DELETE | `/v1/documents/{doc_id}` | 删除文档与向量 |
| POST | `/v1/rag/query` | RAG 检索 |
| POST | `/v1/agent/chat` | Agent 普通对话 |
| POST | `/v1/agent/chat/stream` | Agent SSE 流式对话 |
| POST | `/v1/tools/execute` | 手动执行工具 |
| GET | `/v1/tools` | 工具列表 |
| GET | `/v1/agent/logs` | Agent 日志 |
| GET | `/v1/redis/info` | Redis 信息 |
| GET | `/v1/redis/health` | Redis 健康状态 |
| GET | `/v1/stats` | 系统统计 |

服务启动前必须设置 `APP_API_TOKEN`（本地演示可用 `local-demo`，也可用 `AUTH_DISABLED=true` 在本地开发/测试时显式关闭鉴权）。设置后所有 `/v1/*` 请求需要携带 `Authorization: Bearer <token>` 或 `X-API-Token: <token>`。

## 网页供应商配置

网页的“供应商管理”页可以添加 OpenAI 兼容供应商（如 Kimi、智谱、DeepSeek 或自定义网关），无需修改 `.env`、无需重启服务：

1. 填写名称、Base URL、API Key、聊天模型和 Embedding 模型。
2. 点击“测试 Chat”或“测试 Embedding”验证连接。
3. 保存后，顶部“模型”下拉会合并内置与自定义供应商；“Embedding”下拉只列出配置了 Embedding 模型的供应商。
4. Agent 对话、流式对话、RAG 检索和文档上传会自动携带当前供应商配置。

切换 Embedding 模型时必须先重建索引，否则上传/检索会返回 409。入口在“供应商管理 -> 向量库 Embedding -> 重建索引”，输入 `rebuild` 确认；重建会清空全部向量。

安全说明：非敏感配置（名称、Base URL、模型）保存在浏览器 `localStorage`；API Key 仅保存在当前页面 `sessionStorage`，关闭页面后清除，不进入日志、历史记录和向量元数据。

## RAG 链路

```text
文档上传 -> 解析 -> 递归切分 -> Embedding -> Chroma 入库
用户问题 -> Embedding -> 向量检索 + BM25 -> 混合排序 -> 阈值过滤 -> 构造上下文 -> Agent 回答
```

检索结果会保留 `chunk_id`、`doc_id`、`metadata` 和来源文件名，回答阶段切换到 RAG System Prompt 并要求引用来源。

## 文档

- [docs/theory.md](docs/theory.md)：RAG 完整链路、工具调用与普通函数调用区别、Agent 状态管理等理论知识
- [docs/prompt_records.md](docs/prompt_records.md)：Prompt 结构、设计理由、Few-shot、输出格式控制、异常处理与 3 组对比
- [docs/系统设计.md](docs/系统设计.md)：系统架构与模块设计
- [docs/实施计划.md](docs/实施计划.md)：实施计划与验收记录
- [docs/problem_log.md](docs/problem_log.md)：问题记录与改造决策

## Prompt 工程

- 完整记录见 [docs/prompt_records.md](docs/prompt_records.md)，包含角色/任务/上下文/输入/输出格式/约束拆解、System Prompt 设计理由、Few-shot、输出格式控制、不确定/越界处理和 3 组 Prompt 对比
- Agent System Prompt：动态写入服务器当前日期，明确助手角色、任务边界、可用工具与思考规则
- RAG System Prompt：检索成功后在下一轮注入检索上下文，只基于文档回答、引用来源、未找到时明确拒答
- 工具参数 JSON 解析失败时，把原始参数和错误作为 `role="tool"` 消息回灌给模型重试
- ReAct 循环最多 5 轮，达到上限时返回明确的兜底回答并记录 warning
- 评估问题集见 `data/test_questions/测试问题集.md`

## 效果评估

评估问题集与逐题记录见 [data/test_questions/测试问题集.md](data/test_questions/测试问题集.md)：共 25 条，覆盖知识检索、计算、搜索、时间、混合、拒绝与综合场景；当前统计为 20 成功 / 3 部分成功 / 2 失败，主要失败原因为 `web_search` 为模拟实现，真实搜索需接入搜索 API。

## 测试

```bash
python -m pytest tests/ -v
python -m ruff check backend/ tests/
```

当前 60 个测试覆盖：API 端点、鉴权开关、健康降级、限流（含内存兜底）、文档解析、文本切分、混合检索、Agent 工具循环、流式 SSE、上传成功/部分入库与 Embedding 失败回滚、供应商测试、向量库 profile 与重建、API Key 脱敏。测试使用 Mock，不依赖真实模型 API 或外网。

## 项目结构

```text
backend/             FastAPI 后端
frontend/index.html  单页前端
tests/               自动化测试
data/                文档、Chroma、日志、评估问题
docs/                系统设计、实施计划、理论知识、Prompt 记录
examples/            使用示例
Dockerfile           Docker 镜像
docker-compose.yml   容器编排
.github/workflows/   CI
```
## 历史说明

本仓库早期历史中存在机器化提交形态：2026-08-17 19:38/19:39 两分钟 11+38 个 commit（逐文件提交规程产物）。
该形态源于当时执行的"逐文件提交"自动化规程，不代表真实开发节奏，也不反映代码来源的全部事实；
自 2026-09-29 起已改为功能分支 + 逻辑分组提交 + squash 合并，并以 CI 门禁（测试/lint/格式/构建）作为合并前提。

## 修改记录

- 2026-09-29：
  - requirements.txt：补充 python-multipart 与 tzdata，修复全部 API 测试 RuntimeError 与镜像内上传链路
  - .github/workflows/ci.yml：新增 docker job（hashFiles 守卫），CI 内验证 Dockerfile 可构建
  - README.md：新增 CI badge、历史说明与修改记录小节
- 2026-10-01：Embedding 供应商解耦（EMBEDDING_PROVIDER）+ fitz→pymupdf 迁移 + 测试清理重试
- 2026-10-01：移除内置模型名默认值；README 新增「供应商配置示例」（RAG/Embedding 三种组合与网页配置说明）
