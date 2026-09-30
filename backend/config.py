# 应用配置
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


DOCUMENTS_DIR = _env_path("DOCUMENTS_DIR", BASE_DIR / "data" / "documents")
CHROMA_DB_DIR = _env_path("CHROMA_DB_DIR", BASE_DIR / "data" / "chroma_db")
DATABASE_PATH = _env_path("DATABASE_PATH", BASE_DIR / "data" / "agent_logs.db")

# 创建目录
for d in [DOCUMENTS_DIR, CHROMA_DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM 配置 - 支持多供应商
LLM_CONFIGS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    },
    "dashscope": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "chat_model": os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        "embedding_model": os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "chat_model": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        "embedding_model": "",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        "chat_model": os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-20250514"),
        "embedding_model": "",
    },
}

# 默认供应商
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "dashscope")

# 切分参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索参数
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.7"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# 服务器配置
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8081"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 安全配置
APP_ENV = os.getenv("APP_ENV", "development")
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "")
# AUTH_DISABLED=true 时显式关闭鉴权，仅用于本地开发或自动化测试
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "false").lower() == "true"
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8001,http://127.0.0.1:8001,http://localhost:8081,http://127.0.0.1:8081",
).split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"

# Rate Limiting
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_TRUST_XFF = os.getenv("RATE_LIMIT_TRUST_XFF", "false").lower() == "true"

# Embedding
EMBEDDING_MAX_BATCH = int(os.getenv("EMBEDDING_MAX_BATCH", "32"))
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "60"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
# 修改记录：
#   2026-09-30 deepseek profile 默认 base_url 补 /v1（修复 OpenAI SDK 缺 /v1 导致 404）
