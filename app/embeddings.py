"""
Embedding providers for DocuMind RAG.

The pipeline is written against the `EmbeddingProvider` interface so the
default, dependency-light TF-IDF embedder can be swapped for a real
sentence-transformer or an OpenAI/Cohere embedding API in production
without touching any other module.
"""
from __future__ import annotations

import abc
import os
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingProvider(abc.ABC):
    """Common interface every embedder must implement."""

    @abc.abstractmethod
    def fit(self, documents: List[str]) -> None:
        """Fit/learn a vocabulary or model from the corpus (no-op for API embedders)."""

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 matrix of embeddings."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        ...


class TfidfEmbedder(EmbeddingProvider):
    """
    Zero-dependency, zero-API-key embedder used as the default so the whole
    pipeline runs offline out of the box. TF-IDF + L2 normalization gives
    cosine-similarity retrieval that is surprisingly competitive on small,
    domain-specific corpora (the kind a portfolio/demo project deals with).
    """

    def __init__(self, max_features: int = 4096):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2)
        )
        self._fitted = False

    def fit(self, documents: List[str]) -> None:
        self._vectorizer.fit(documents)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        mat = self._vectorizer.transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    @property
    def dim(self) -> int:
        return len(self._vectorizer.vocabulary_) if self._fitted else 4096


class OpenAIEmbedder(EmbeddingProvider):
    """
    Production-grade embedder using OpenAI's `text-embedding-3-small`.
    Requires OPENAI_API_KEY. Not used by default so the project stays
    runnable without any paid API key; swap it in via `get_embedder("openai")`.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai to use OpenAIEmbedder") from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._dim = 1536

    def fit(self, documents: List[str]) -> None:
        pass  # API embedders need no local fitting

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim


def get_embedder(kind: str = "tfidf") -> EmbeddingProvider:
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedder kind: {kind}")
