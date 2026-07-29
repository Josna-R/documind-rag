"""Document loading and chunking utilities."""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

from .vectorstore import Chunk


def load_documents(folder: str) -> List[Tuple[str, str]]:
    """Return a list of (source_name, full_text) for every .txt/.md file in folder."""
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.md")))
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Simple sliding-window word chunker with overlap to preserve context across boundaries."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(folder: str, chunk_size: int = 500, overlap: int = 80) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for source, text in load_documents(folder):
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append(Chunk(id=cid, text=piece, source=source))
            cid += 1
    return chunks
