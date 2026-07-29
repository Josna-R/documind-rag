# DocuMind RAG

Retrieval-Augmented Generation (RAG) engine that answers questions over your
own documents, with a FastAPI backend and a Streamlit UI. Runs fully offline
out of the box (no API key required) and is designed to drop in a real
embedding model or LLM for production use with a one-line change.

## Why this project

Demonstrates the core RAG architecture pattern used across the industry
(chunking → embedding → vector search → grounded generation) plus the
engineering practice of designing swappable providers instead of hard-coding
a single vendor's API — the same pattern used in the author's CliniMatch
clinical-trial-matching project, generalized into a reusable service.

## Architecture

```
documents (.txt/.md)
       │
       ▼
  chunk_text()  ── sliding window, configurable size/overlap
       │
       ▼
EmbeddingProvider  ── TfidfEmbedder (default, offline)  or  OpenAIEmbedder
       │
       ▼
   VectorStore  ── FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)
       │
       ▼
     query()  ── retrieve top-k chunks
       │
       ▼
       LLM  ── FakeLLM (default, offline extractive answerer)  or  OpenAILLM / AnthropicLLM
       │
       ▼
  answer + cited sources
```

Every stage is an interface (`EmbeddingProvider`, `LLM`) so swapping in a real
embedding model or a hosted LLM is a one-line change in `app/embeddings.py` /
`app/llm.py` — no changes to the ingestion, retrieval, or API layers.

## Quickstart

```bash
pip install -r requirements.txt

# Run the test suite (fully offline, no API key needed)
pytest tests/ -v

# Try the API
uvicorn app.api:app --reload
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"folder": "data/sample_docs"}'
curl -X POST localhost:8000/query  -H "Content-Type: application/json" -d '{"question": "How many days of PTO do employees get?"}'

# Or use the UI
streamlit run app/streamlit_app.py
```

## Upgrading to production models

```bash
pip install openai   # or: pip install anthropic
export OPENAI_API_KEY=sk-...
export DOCUMIND_LLM=openai
uvicorn app.api:app --reload
```

## Tech stack

Python · FastAPI · Streamlit · FAISS · scikit-learn (TF-IDF) · pydantic ·
pluggable OpenAI / Anthropic generation

## Possible extensions

- Swap `TfidfEmbedder` for `sentence-transformers` for semantic (not just
  lexical) retrieval.
- Add a reranker stage (cross-encoder) before generation.
- Persist the FAISS index to disk instead of rebuilding on every restart.
- Add streaming responses via FastAPI's `StreamingResponse`.
