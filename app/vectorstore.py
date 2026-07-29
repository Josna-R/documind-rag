"""FAISS-backed vector store for chunked document embeddings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import faiss
import numpy as np


@dataclass
class Chunk:
    id: int
    text: str
    source: str


@dataclass
class VectorStore:
    dim: int
    chunks: List[Chunk] = field(default_factory=list)
    _index: faiss.IndexFlatIP = field(init=False, repr=False)

    def __post_init__(self):
        # Inner product on L2-normalized vectors == cosine similarity.
        self._index = faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings/chunks length mismatch")
        self._index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[Tuple[Chunk, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        scores, idxs = self._index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
