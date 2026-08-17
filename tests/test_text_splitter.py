import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.text_splitter import recursive_split, semantic_split


class TestTextSplitter:
    def test_recursive_split(self):
        text = "Hello. " * 1000
        chunks = recursive_split(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(len(c) <= 220 for c in chunks)

    def test_semantic_split(self):
        text = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        chunks = semantic_split(text, chunk_size=5)
        assert len(chunks) > 1
