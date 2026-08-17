# RAG + Tool Calling Agent - 理论知识学习笔记

## 一、RAG (检索增强生成) 完整链路

### 1.1 什么是 RAG
RAG = Retrieval-Augmented Generation，检索增强生成。
核心思想: 在 LLM 回答之前，先从外部知识库中检索相关信息，将检索结果作为上下文注入到 prompt 中，让模型基于真实信息回答，从而解决:
- 知识过时问题 (模型训练数据有截止日期)
- 私有知识不可见问题 (企业内部文档未被训练)
- 模型幻觉问题 (模型编造不存在的知识)

### 1.2 RAG 完整链路

文档加载 -> 文本切分 -> Embedding -> 向量存储 -> 检索 -> (可选重排) -> 上下文注入 -> 生成回答

(1) 文档加载 (Document Loading)
    - 不同文档格式需要不同的解析器
    - PDF: pymupdf / pdfplumber / PyPDF2
    - Markdown: 用解析器提取结构化内容 (标题、列表、代码块)
    - Word: python-docx
    - HTML: beautifulsoup4

(2) 文本切分 (Chunking) [关键]
    - 目标: 把长文档切成适合检索的短片段
    - 策略1: 固定长度切分 (Fixed-size) - 简单但可能切断语义
    - 策略2: 递归字符切分 (RecursiveCharacterTextSplitter) - 按分隔符优先级切分
    - 策略3: 语义切分 (Semantic Chunking) - 按段落/句子边界
    - 策略4: 结构化切分 - 按 Markdown 标题、代码结构
    - 关键参数: chunk_size (片段大小), chunk_overlap (重叠量)
    - Overlap 的作用: 避免在切分边界处丢失关键信息

(3) Embedding (文本向量化)
    - 将文本转换为固定维度的浮点数向量
    - 常见模型: OpenAI text-embedding-3-small / ada-002, bge, sentence-transformers
    - 语义相近的文本在向量空间中的距离也更近
    - 归一化: 将向量缩放到单位长度 (L2 norm)

(4) 向量存储 (Vector Store)
    - 存储文档片段的向量表示，支持高效相似度搜索
    - 常见选择: Chroma (轻量级，适合本地开发), FAISS (高性能), Pinecone (托管), Qdrant
    - 索引算法: FLAT (暴力搜索), IVF (倒排文件), HNSW (分层可导航小世界图)
    - 本项目选择 Chroma: 轻量、Python原生、本地持久化

(5) 检索 (Retrieval)
    - 向量检索: 计算用户问题向量与文档片段向量的余弦相似度，取 Top-K
    - 混合检索: 结合关键词检索 (BM25) + 向量检索，提升召回率
    - MMR (最大边际相关性): 在相关性和多样性之间平衡，避免重复
    - 元数据过滤: 按文档来源、时间等维度过滤

(6) 重排序 (Reranking) [可选但推荐]
    - 粗排 + 精排二阶段架构
    - Cross-Encoder 模型对 (query, doc) 对进行精确打分
    - 常见: bge-reranker, Cohere Rerank
    - 效果: 显著提升 Top-K 结果的相关性

(7) 上下文注入与生成
    - 将检索到的片段拼接为上下文
    - 控制模型只基于检索内容回答 (system prompt 约束)
    - 引用溯源: 标注回答中引用的具体片段和来源
    - 不确定性处理: 当检索结果与问题不相关时拒绝回答

### 1.3 RAG 评估
- 检索质量: 准确率 (Precision), 召回率 (Recall), MRR, NDCG
- 生成质量: 答案忠实度 (Faithfulness), 相关性 (Relevance), 幻觉率
- 工具: RAGAS 框架, 人工评估
- 需要准备测试集持续验证

### 1.4 Advanced RAG
- Query Rewrite: 改写用户查询以提升检索效果
- HyDE: 先生成假设性回答，再用回答去检索
- Self-Query: 让模型从查询中提取结构化过滤条件
- Agentic RAG: Agent 自主决定何时检索、检索什么
- GraphRAG: 基于知识图谱的 RAG (Microsoft)

## 二、Agent 与工具调用

### 2.1 什么是 Agent
Agent 是一种能自主决策、调用工具、管理状态来完成复杂任务的 AI 系统。
- Chatbot: 纯对话，不执行操作
- Workflow: 预定流程，确定步骤
- Agent: 动态决策，根据环境反馈调整行动

### 2.2 Agent 核心循环 (Agent Loop)
1. 接收用户输入
2. 模型根据当前状态 (对话历史+工具结果) 决定下一步行动
3. 执行行动 (工具调用 / 检索 / 直接回答)
4. 将行动结果反馈给模型
5. 重复 2-4 直到任务完成或达到终止条件

### 2.3 工具调用 vs 普通函数调用 [面试重点]
- 普通函数调用: 开发者在代码中硬编码调用哪个函数、传什么参数
- 工具调用 (Tool Calling / Function Calling):
  - 模型决定是否调用工具、调用哪个工具、传什么参数
  - 模型只生成工具调用意图 (tool_call_id, name, arguments)
  - 真正执行工具的是业务系统
  - 工具结果再返回给模型进行下一步推理
  - 关键: 模型是决策者，业务系统是执行者

### 2.4 ReAct 范式
Reasoning + Acting
- Thought: 分析当前情况，决定下一步
- Action: 执行工具或检索
- Observation: 观察执行结果
- 循环直到可以给出最终答案

### 2.5 工具设计原则
- 清晰的名称和描述 (影响模型选择工具的准确性)
- 精确的参数 Schema (类型、枚举、必需字段)
- 健壮的错误处理 (工具返回错误时模型能理解并重试)
- 超时和重试机制

### 2.6 状态管理
- 对话历史: 维护完整的消息列表
- 工具调用记录: 记录每次工具调用的参数和结果
- 上下文窗口管理: 超过 Token 限制时进行摘要或裁剪

### 2.7 记忆系统
- 短期记忆: 当前对话历史
- 长期记忆: 跨会话的重要信息
- 情景记忆: 具体的交互经历
- 语义记忆: 从经验中提取的知识

### 2.8 Agent 安全
- Prompt Injection 防护: 用户输入中可能包含恶意指令
- 工具权限控制: 敏感操作需要权限检查
- 沙箱执行: 不可信代码在隔离环境执行
- 人工审批 (Human-in-the-Loop): 高风险操作需要人工确认

## 三、本项目技术选型理由

| 选项 | 选择 | 理由 |
|------|------|------|
| 后端框架 | FastAPI | 异步、高性能、自动文档，与项目1一致 |
| 向量数据库 | Chroma | 轻量无服务器，Python 原生，本地持久化 |
| Embedding | Dashscope / OpenAI 兼容 | 通过供应商兼容接口调用，DeepSeek 不提供 Embedding |
| 文档解析 | pymupdf | Python 最快的 PDF 库 |
| 切分工具 | LangChain Text Splitters | 成熟的切分策略实现 |
| 前端 | HTML+JS | 轻量，单页面应用 |
| Agent 框架 | 自定义 ReAct | 自行实现推理循环与工具调用，LangChain 仅用于文本切分 |
