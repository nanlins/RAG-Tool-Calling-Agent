import os
import tempfile
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



@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c