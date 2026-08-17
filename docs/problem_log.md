# 问题记录与改造决策

## 2026-08-16 供应商管理 + 动态客户端

### 背景与原因
- 此前切换供应商必须修改 `.env` 并重启服务，网页无法添加 Kimi、智谱等 OpenAI 兼容供应商。
- 需求是不修改环境变量、不重启服务，就能在网页中配置 Base URL、API Key、聊天模型和 Embedding 模型，并在多个供应商之间切换对比。
- DeepSeek 原生不支持 Embedding，因此聊天供应商与 Embedding 供应商必须分开管理。

### 设计决策
- 新增 `ProviderConfig` 统一供应商模型，请求级携带，不写入全局配置。
- 内置供应商仍由 `LLM_CONFIGS` 提供，未配置自定义供应商时原行为不回归。
- 前端非敏感配置存 `localStorage`，API Key 存 `sessionStorage`，关闭页面后清除。
- 向量库集合 metadata 记录 Embedding profile；模型或维度不一致时返回 409，避免用旧索引匹配新向量。

### 风险与边界
- 自定义供应商由用户输入，Base URL/模型名错误时连接失败，由 `/v1/providers/test` 提前校验。
- API Key 不得出现在日志、历史、向量 metadata 和响应；后端对错误文本脱敏，前端不落盘。
- 切换 Embedding 模型后必须重建索引，否则上传/检索会被明确阻止。
- 测试连接会真实调用供应商 API，可能产生少量 Token/费用。
