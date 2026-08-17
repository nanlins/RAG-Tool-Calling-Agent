import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.document_loader import load_document


class TestDocumentLoader:
    def test_parse_text(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("hello agent\nsecond line", encoding="utf-8")
        doc = load_document(str(p))
        assert doc.doc_type == "txt"
        assert "hello agent" in doc.content

    def test_parse_markdown(self, tmp_path):
        p = tmp_path / "guide.md"
        p.write_text("# Title\n\ncontent body", encoding="utf-8")
        doc = load_document(str(p))
        assert doc.doc_type == "md"
        assert doc.title == "Title"

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError):
            load_document(str(tmp_path / "test.docx"))