import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rag_agent_test_"))
os.environ["DATABASE_PATH"] = str(_TMP / "agent_logs.db")
os.environ["DOCUMENTS_DIR"] = str(_TMP / "documents")
os.environ["CHROMA_DB_DIR"] = str(_TMP / "chroma_db")
os.environ["REDIS_ENABLED"] = "false"
os.environ["APP_API_TOKEN"] = ""
os.environ["APP_ENV"] = "development"
os.environ["AUTH_DISABLED"] = "true"


def _rmtree_retry(path: Path) -> None:
    """带重试的目录清理：Windows 下 Chroma/SQLite 句柄未及时释放时会 PermissionError"""
    for _ in range(5):
        try:
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                return
        except PermissionError:
            pass
        time.sleep(0.2)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    _rmtree_retry(_TMP)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c