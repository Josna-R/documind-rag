"""
LLM providers for answer generation.

DocuMind ships with a `FakeLLM` so `demo.py` and the test suite run fully
offline with zero cost and zero API keys -- useful for CI and for anyone
cloning the repo to try it immediately. Set OPENAI_API_KEY or
ANTHROPIC_API_KEY and pass --llm openai / --llm anthropic to upgrade to a
real model; no other code changes are needed because every provider
implements the same `LLM.generate()` interface.
"""
from __future__ import annotations

import abc
import os
import re
from typing import List


class LLM(abc.ABC):
    @abc.abstractmethod
    def generate(self, question: str, context_chunks: List[str]) -> str:
        ...


class FakeLLM(LLM):
    """
    Deterministic, extractive "generator": no external calls at all.
    It picks the most relevant sentences out of the retrieved context using
    lexical overlap with the question, so the RAG pipeline is genuinely
    end-to-end testable (retrieval -> synthesis -> answer) without an LLM.
    """

    def generate(self, question: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "I couldn't find anything relevant in the indexed documents."

        q_terms = set(re.findall(r"[a-zA-Z']+", question.lower()))
        sentences = []
        for chunk in context_chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                sent = sent.strip()
                if len(sent) > 15:
                    sentences.append(sent)

        def overlap_score(sent: str) -> int:
            terms = set(re.findall(r"[a-zA-Z']+", sent.lower()))
            return len(terms & q_terms)

        ranked = sorted(sentences, key=overlap_score, reverse=True)
        top = [s for s in ranked[:3] if overlap_score(s) > 0] or ranked[:2]
        if not top:
            return "The indexed documents don't appear to contain an answer to that question."
        return " ".join(top)


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4.1-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai to use OpenAILLM") from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def generate(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()


class AnthropicLLM(LLM):
    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic to use AnthropicLLM") from exc
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def generate(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


def get_llm(kind: str = "fake") -> LLM:
    if kind == "fake":
        return FakeLLM()
    if kind == "openai":
        return OpenAILLM()
    if kind == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Unknown LLM kind: {kind}")
