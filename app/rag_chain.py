"""Ties embeddings + vector store + LLM into a single ingest/query pipeline."""
from __future__ import annotations

from typing import List

from .embeddings import EmbeddingProvider, get_embedder
from .ingest import build_chunks
from .llm import LLM, get_llm
from .vectorstore import VectorStore


class RagPipeline:
    def __init__(self, embedder_kind: str = "tfidf", llm_kind: str = "fake"):
        self.embedder: EmbeddingProvider = get_embedder(embedder_kind)
        self.llm: LLM = get_llm(llm_kind)
        self.store: VectorStore | None = None

    def ingest_folder(self, folder: str, chunk_size: int = 500, overlap: int = 80) -> int:
        chunks = build_chunks(folder, chunk_size, overlap)
        if not chunks:
            raise ValueError(f"No .txt/.md documents found in {folder}")
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)
        self.store = VectorStore(dim=vectors.shape[1])
        self.store.add(vectors, chunks)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> dict:
        if self.store is None or len(self.store) == 0:
            raise RuntimeError("Call ingest_folder() before query().")
        q_vec = self.embedder.embed([question])[0]
        hits = self.store.search(q_vec, k=k)
        context_chunks: List[str] = [chunk.text for chunk, _ in hits]
        answer = self.llm.generate(question, context_chunks)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": chunk.source, "score": round(score, 4), "excerpt": chunk.text[:220]}
                for chunk, score in hits
            ],
        }
