# DocuMind RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Tests](https://img.shields.io/badge/tests-4%2F4%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey) ![Offline](https://img.shields.io/badge/runs-offline%2C%20no%20API%20key-success)

Retrieval-Augmented Generation (RAG) engine that answers questions over your
own documents, with a FastAPI backend and a Streamlit UI. Runs fully offline
out of the box (no API key required) and is designed to drop in a real
embedding model or LLM for production use with a one-line change.

**TL;DR:** point it at a folder of `.txt`/`.md` files, ask a question in
plain English, get back an answer plus the exact source chunks it was
grounded in. Try it in under 2 minutes — see [Get the code](#get-the-code) below.

## What it does

1. **Ingests** a folder of documents and splits them into overlapping chunks.
2. **Embeds** each chunk (TF-IDF by default, or a real embedding model/API).
3. **Indexes** the vectors in FAISS for fast cosine-similarity search.
4. On a query, **retrieves** the most relevant chunks and **generates** a
   grounded answer, citing which document(s) it came from.

Both the FastAPI JSON API and the Streamlit UI sit on top of the same
`RagPipeline` class, so anything you can do with one you can do with the other.

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

## Get the code

```bash
git clone https://github.com/Josna-R/documind-rag.git
cd documind-rag
```

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

Open `http://localhost:8501` for the Streamlit UI, or `http://localhost:8000/docs`
for the interactive FastAPI/Swagger docs. Two sample documents (an employee
handbook and a product FAQ) are bundled in `data/sample_docs/` so there's
something to query immediately — no setup needed.

## Project structure

```
documind-rag/
├── app/
│   ├── embeddings.py    # EmbeddingProvider interface + TF-IDF / OpenAI implementations
│   ├── vectorstore.py   # FAISS-backed vector store
│   ├── ingest.py        # document loading + chunking
│   ├── llm.py            # LLM interface + FakeLLM / OpenAI / Anthropic implementations
│   ├── rag_chain.py      # wires embeddings + vector store + LLM into one pipeline
│   ├── api.py             # FastAPI service (/ingest, /query, /health)
│   └── streamlit_app.py  # Streamlit UI
├── data/sample_docs/      # example documents to query out of the box
├── tests/test_pipeline.py # end-to-end pipeline tests
└── requirements.txt
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

## Author

Built by [Josna Deepa Rayana](https://github.com/Josna-R), AI/ML Engineer &
Data Scientist.
