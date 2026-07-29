"""FastAPI service exposing the RAG pipeline as a JSON API."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag_chain import RagPipeline

app = FastAPI(
    title="DocuMind RAG API",
    description="Retrieval-Augmented Q&A over your own documents.",
    version="1.0.0",
)

_pipeline = RagPipeline(
    embedder_kind=os.environ.get("DOCUMIND_EMBEDDER", "tfidf"),
    llm_kind=os.environ.get("DOCUMIND_LLM", "fake"),
)


class IngestRequest(BaseModel):
    folder: str = "data/sample_docs"
    chunk_size: int = 500
    overlap: int = 80


class QueryRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        n_chunks = _pipeline.ingest_folder(req.folder, req.chunk_size, req.overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "chunks_indexed": n_chunks}


@app.post("/query")
def query(req: QueryRequest):
    try:
        return _pipeline.query(req.question, req.k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(_pipeline.store) if _pipeline.store else 0}
