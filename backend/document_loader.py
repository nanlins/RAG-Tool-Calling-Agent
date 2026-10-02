# 文档解析模块 - 支持 PDF / Markdown / TXT 格式
import os
import re
from datetime import datetime
from uuid import uuid4

import pymupdf


class Document:
    """统一文档结构"""
    def __init__(self, doc_id: str, filename: str, source: str, doc_type: str,
                 title: str, content: str, metadata: dict | None = None):
        self.id = doc_id
        self.filename = filename
        self.source = source
        self.doc_type = doc_type
        self.title = title
        self.content = content
        self.metadata = metadata or {}
        self.uploaded_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "source": self.source,
            "doc_type": self.doc_type,
            "title": self.title,
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "uploaded_at": self.uploaded_at,
            "content_length": len(self.content),
        }


def generate_id() -> str:
    return uuid4().hex[:12]


def parse_pdf(filepath: str) -> Document:
    """解析 PDF 文档"""
    filepath = str(filepath)
    filename = os.path.basename(filepath)
    doc_id = generate_id()
    
    doc = pymupdf.open(filepath)
    content_parts = []
    metadata = {"pages": len(doc), "page_ranges": []}
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            content_parts.append(f"[第{page_num}页]\n{text}")
            metadata["page_ranges"].append(page_num)
    
    content = "\n\n".join(content_parts)
    title = os.path.splitext(filename)[0]
    
    doc.close()
    return Document(doc_id, filename, filepath, "pdf", title, content, metadata)


def parse_markdown(filepath: str) -> Document:
    """解析 Markdown 文档"""
    filepath = str(filepath)
    filename = os.path.basename(filepath)
    doc_id = generate_id()
    
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    
    # 提取标题
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else os.path.splitext(filename)[0]
    
    # 提取所有标题作为元数据
    headers = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
    metadata = {
        "headers": [{"level": len(h[0]), "text": h[1]} for h in headers],
    }
    
    return Document(doc_id, filename, filepath, "md", title, content, metadata)


def parse_text(filepath: str) -> Document:
    """解析纯文本文档"""
    filepath = str(filepath)
    filename = os.path.basename(filepath)
    doc_id = generate_id()
    
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    
    title = os.path.splitext(filename)[0]
    return Document(doc_id, filename, filepath, "txt", title, content)


def load_document(filepath: str) -> Document | None:
    """自动识别格式并加载文档"""
    ext = os.path.splitext(filepath)[1].lower()
    parsers = {
        ".pdf": parse_pdf,
        ".md": parse_markdown,
        ".txt": parse_text,
        ".markdown": parse_markdown,
    }
    parser = parsers.get(ext)
    if not parser:
        raise ValueError(f"不支持的文档格式: {ext}，仅支持 PDF/Markdown/TXT")
    return parser(filepath)

# 修改记录：
#   2026-10-01 fitz 弃用别名迁移为 pymupdf，page.get_text() 显式传 'text'（兼容 pymupdf>=1.23）
