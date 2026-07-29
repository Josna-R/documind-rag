import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import chunk_text
from app.rag_chain import RagPipeline

SAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")


def test_chunk_text_respects_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # consecutive chunks should share the overlapping words
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_ingest_indexes_all_documents():
    pipeline = RagPipeline(embedder_kind="tfidf", llm_kind="fake")
    n = pipeline.ingest_folder(SAMPLE_FOLDER)
    assert n > 0
    assert len(pipeline.store) == n


def test_query_retrieves_relevant_chunk_and_answers():
    pipeline = RagPipeline(embedder_kind="tfidf", llm_kind="fake")
    pipeline.ingest_folder(SAMPLE_FOLDER)
    result = pipeline.query("How many days of PTO do employees get?")
    assert result["sources"], "expected at least one retrieved source"
    # the PTO chunk should be retrieved from the handbook, not the FAQ
    assert result["sources"][0]["source"] == "employee_handbook.txt"
    assert "15 days" in result["answer"] or "20 days" in result["answer"]


def test_query_on_empty_index_raises():
    pipeline = RagPipeline(embedder_kind="tfidf", llm_kind="fake")
    try:
        pipeline.query("anything")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
