# 数据库操作模块 - 使用 SQLite 存储 Agent 日志和文档元数据
import json
import sqlite3
from datetime import datetime

from backend.config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
    """为已存在的表补充缺失列（轻量迁移）"""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    """初始化数据库表"""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                server_filename TEXT DEFAULT '',
                source TEXT,
                doc_type TEXT,
                title TEXT,
                content_length INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                chunks_added INTEGER DEFAULT 0,
                ingest_status TEXT DEFAULT 'pending',
                uploaded_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT DEFAULT '',
                tool_calls TEXT DEFAULT '[]',
                tool_call_id TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                question TEXT NOT NULL,
                reasoning TEXT DEFAULT '',
                tool_calls TEXT DEFAULT '[]',
                final_answer TEXT DEFAULT '',
                model TEXT DEFAULT '',
                total_tokens INTEGER DEFAULT 0,
                elapsed_ms REAL DEFAULT 0,
                success INTEGER DEFAULT 1,
                error TEXT DEFAULT '',
                timestamp TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_agent_logs_ts ON agent_logs(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_logs_conv ON agent_logs(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_conv_messages ON conversation_messages(conversation_id);
        """)
        _ensure_column(conn, "documents", "server_filename", "TEXT DEFAULT ''")
        _ensure_column(conn, "documents", "chunks_added", "INTEGER DEFAULT 0")
        _ensure_column(conn, "documents", "ingest_status", "TEXT DEFAULT 'pending'")
        conn.commit()
    finally:
        conn.close()


def save_document(doc: dict) -> bool:
    """保存文档元数据"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(id, filename, server_filename, source, doc_type, title, content_length, "
            "chunk_count, chunks_added, ingest_status, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc["id"], doc.get("filename", ""), doc.get("server_filename", ""),
             doc.get("source", ""), doc.get("doc_type", ""),
             doc.get("title", doc.get("filename", "")), doc.get("content_length", 0),
             doc.get("chunk_count", 0), doc.get("chunks_added", 0),
             doc.get("ingest_status", "pending"),
             doc.get("uploaded_at", datetime.now().isoformat()))
        )
        conn.commit()
        return True
    finally:
        conn.close()


def delete_document_record(doc_id: str) -> bool:
    """删除文档记录，返回是否删除了记录"""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_document_record(doc_id: str) -> dict | None:
    """获取单个文档记录"""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_documents() -> list[dict]:
    """获取所有文档列表"""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_agent_log(log_data: dict) -> int:
    """保存 Agent 调用日志"""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO agent_logs (conversation_id, question, reasoning, tool_calls, final_answer, "
            "model, total_tokens, elapsed_ms, success, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                log_data.get("conversation_id", ""),
                log_data.get("question", ""),
                log_data.get("reasoning", ""),
                json.dumps(log_data.get("tool_calls", []), ensure_ascii=False),
                log_data.get("final_answer", ""),
                log_data.get("model", ""),
                log_data.get("total_tokens", 0),
                log_data.get("elapsed_ms", 0),
                1 if log_data.get("success", True) else 0,
                log_data.get("error", ""),
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_agent_logs(limit: int = 50) -> list[dict]:
    """获取 Agent 日志列表"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                d["tool_calls"] = []
            result.append(d)
        return result
    finally:
        conn.close()


# === SQLite 会话兜底 ===

def save_conversation_sqlite(conv_id: str, messages: list[dict]) -> bool:
    """将会话消息写入 SQLite（Redis 不可用时的兜底）"""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conv_id,))
        for m in messages:
            conn.execute(
                "INSERT INTO conversation_messages (conversation_id, role, content, tool_calls, tool_call_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    conv_id,
                    m.get("role", "user"),
                    m.get("content", "") or "",
                    json.dumps(m.get("tool_calls", []), ensure_ascii=False),
                    m.get("tool_call_id", "") or "",
                )
            )
        conn.execute(
            "INSERT INTO conversations (id, title, message_count, updated_at) VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(id) DO UPDATE SET message_count = excluded.message_count, updated_at = datetime('now')",
            (conv_id, "", len(messages)),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_conversation_sqlite(conv_id: str) -> list[dict] | None:
    """从 SQLite 读取会话消息"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM conversation_messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        if not rows:
            return None
        messages = []
        for r in rows:
            msg = {"role": r["role"], "content": r["content"]}
            if r["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(r["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = []
            if r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            messages.append(msg)
        return messages
    finally:
        conn.close()