# Redis cache module - session storage and rate limiting
import json
import logging
import threading
import time

from backend.config import REDIS_ENABLED, REDIS_URL
from backend.database import get_conversation_sqlite, save_conversation_sqlite

logger = logging.getLogger(__name__)

# Lazy Redis connection
_redis = None
_redis_checked = False

# Redis 不可用时的进程内限流兜底
_memory_counts: dict[str, list[int]] = {}
_memory_lock = threading.Lock()


def get_redis():
    global _redis, _redis_checked
    if not REDIS_ENABLED:
        return None
    if _redis is None and not _redis_checked:
        try:
            import redis as _redis_mod
            _redis = _redis_mod.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
            _redis.ping()
        except Exception as e:
            logger.warning("Redis 不可用，将使用 SQLite 会话兜底: %s", e)
            _redis = None
        finally:
            _redis_checked = True
    return _redis


def is_available() -> bool:
    try:
        r = get_redis()
        return r is not None and r.ping()
    except Exception:
        return False


# === Session Storage ===

def save_conversation(conv_id: str, messages: list[dict], ttl: int = 3600) -> bool:
    r = get_redis()
    if r:
        try:
            key = f"conv:{conv_id}"
            r.setex(key, ttl, json.dumps(messages, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning("Redis 会话写入失败: %s", e)
    return save_conversation_sqlite(conv_id, messages)


def get_conversation(conv_id: str) -> list[dict] | None:
    r = get_redis()
    if r:
        try:
            data = r.get(f"conv:{conv_id}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning("Redis 会话读取失败: %s", e)
    return get_conversation_sqlite(conv_id)


# === Rate Limiting ===

def check_rate_limit(key: str, max_requests: int = 60, window: int = 60) -> bool:
    now = int(time.time())
    r = get_redis()
    if r:
        try:
            window_key = f"ratelimit:{key}:{now // window}"
            count = r.incr(window_key)
            if count == 1:
                r.expire(window_key, window + 1)
            return count <= max_requests
        except Exception as e:
            logger.warning("Redis 限流失败，改用进程内限流: %s", e)
    return _check_memory_rate_limit(key, max_requests, window, now)


def _check_memory_rate_limit(key: str, max_requests: int, window: int, now: int) -> bool:
    with _memory_lock:
        cutoff = now - window
        stamps = [s for s in _memory_counts.get(key, []) if s > cutoff]
        if len(stamps) >= max_requests:
            _memory_counts[key] = stamps
            return False
        stamps.append(now)
        _memory_counts[key] = stamps
        return True


def get_redis_info() -> dict:
    r = get_redis()
    if not r:
        return {"status": "unavailable", "reason": "Redis not connected"}
    try:
        info = r.info()
        return {
            "status": "connected",
            "version": info.get("redis_version", "?"),
            "used_memory_human": info.get("used_memory_human", "?"),
            "connected_clients": info.get("connected_clients", 0),
            "uptime_days": info.get("uptime_in_days", 0),
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}