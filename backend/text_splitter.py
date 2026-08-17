# 文本切分模块 - 多种切分策略
import re
from uuid import uuid4

from backend.document_loader import Document


def generate_chunk_id() -> str:
    return uuid4().hex[:12]


def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """递归字符切分 - 按优先级使用分隔符"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def markdown_header_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """基于 Markdown 标题结构的切分 - 保持语义完整性"""
    # 按标题分割
    lines = text.split("\n")
    sections = []
    current_section = []
    current_headers = []
    
    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            if current_section:
                sections.append((list(current_headers), "\n".join(current_section)))
                current_section = []
            # 更新标题栈
            level = len(header_match.group(1))
            title = header_match.group(2)
            # 清除同级及以下标题
            current_headers = [h for h in current_headers if h[0] < level]
            current_headers.append((level, title))
        current_section.append(line)
    
    if current_section:
        sections.append((list(current_headers), "\n".join(current_section)))
    
    # 如果内容太长，对每个 section 进一步切分
    chunks = []
    for headers, section_text in sections:
        header_path = " > ".join(h[1] for h in headers)
        if len(section_text) <= chunk_size:
            chunks.append({"text": section_text, "headers": header_path})
        else:
            sub_chunks = recursive_split(section_text, chunk_size, chunk_overlap)
            for sc in sub_chunks:
                chunks.append({"text": sc, "headers": header_path})
    
    return chunks


def semantic_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """语义切分 - 按段落/自然边界"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < chunk_size:
            if current:
                current += "\n\n" + para
            else:
                current = para
        else:
            if current:
                chunks.append(current)
            current = para
    
    if current:
        chunks.append(current)
    
    return chunks


def split_document(doc: Document, strategy: str = "recursive",
                   chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """对文档执行切分，返回切分后的片段列表"""
    doc_id = doc.id
    filename = doc.filename
    
    if strategy == "recursive":
        raw_chunks = recursive_split(doc.content, chunk_size, chunk_overlap)
        chunks = [{"text": c, "headers": ""} for c in raw_chunks]
    elif strategy == "markdown":
        chunks = markdown_header_split(doc.content, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        raw_chunks = semantic_split(doc.content, chunk_size, chunk_overlap)
        chunks = [{"text": c, "headers": ""} for c in raw_chunks]
    else:
        raise ValueError(f"未知切分策略: {strategy}")
    
    # 统一输出格式
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "id": generate_chunk_id(),
            "doc_id": doc_id,
            "text": chunk["text"],
            "metadata": {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "headers": chunk.get("headers", ""),
            }
        })
    
    return result
